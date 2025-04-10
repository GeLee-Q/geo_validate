import pandas as pd
from PIL import Image
import io
import numpy as np
import base64
from io import BytesIO
import re
import json
from mathruler.grader import extract_boxed_content, grade_answer
from loguru import logger
import time
from tqdm import tqdm
import argparse
import sys
from datetime import datetime
import sglang as sgl
from torch.distributed.device_mesh import init_device_mesh
import os
from typing import List, Tuple, Dict, Any, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info


# Constants
PARQUET_FILE_PATH = "/workspace/geo3k/test.parquet"
LLM_MODEL = "/workspace/Qwen2.5-VL-7B-Instruct"
MAX_TOKENS = 4096
ACC_REWARD_WEIGHT = 0.9
FORMAT_REWARD_WEIGHT = 0.1
BATCH_SIZE = 8 # Define batch size for processing

# Configure logging
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"geo3k_validation_engine_{timestamp}.log"
    
    # Remove default logger
    logger.remove()
    
    # Add file logger with rotation
    logger.add(
        log_file,
        rotation="100 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True
    )
    
    # Add console logger
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")

# ------------------- Utility Functions -------------------
def format_reward(predict_str: str) -> float:
    """
    Checks if the prediction string matches the expected format with <think> and \\boxed{}.
    """
    if predict_str is None:
        return 0.0
    
    pattern = re.compile(r'<think>.*</think>.*\\boxed\{.*\}.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    Grades the answer extracted from the prediction string against the ground truth.
    """
    if predict_str is None:
        return 0.0
    
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str) -> float:
    """
    Computes a weighted score based on the accuracy and format of the prediction.
    """
    accuracy_reward = acc_reward(predict_str, ground_truth)
    format_reward_value = format_reward(predict_str)
    return ACC_REWARD_WEIGHT * accuracy_reward + FORMAT_REWARD_WEIGHT * format_reward_value

def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

def prepare_batch_inputs(batch_rows: List[Tuple[int, pd.Series]]) -> List[Dict[str, Any]]:
    """
    Prepare input data for a batch of rows
    
    Args:
        batch_rows: List of (index, row) tuples
        
    Returns:
        List of dictionaries with prepared input data for each row
    """
    batch_inputs = []
    
    for idx, row in batch_rows:
        try:
            # Extract data from row
            choices = row['choices']
            ground_truth_data = row['ground_truth']
            correct_index = ord(ground_truth_data.upper()) - ord('A')
            ground_truth = choices[correct_index]
            problem_text = row.get('prompt')[0].get('content')
            
            # Process image data
            pil_image = [process_image(row['images'][0])]
            
            # Process prompt
            prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + problem_text
            prompt = prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            prompt = prompt + '<|im_start|>assistant\n'
            
            # Prepare input dict
            batch_inputs.append({
                'index': idx,
                'prompt': prompt,
                'image_data': pil_image,
                'ground_truth': ground_truth
            })
            
        except Exception as e:
            import traceback
            error_msg = f"Error preparing row {idx}:\n"
            error_msg += f"Error type: {type(e).__name__}\n"
            error_msg += f"Error message: {str(e)}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # Add placeholder for failed row prep
            batch_inputs.append({
                'index': idx,
                'error': str(e),
                'ground_truth': "Error"
            })
            
    return batch_inputs

def process_batch(llm, tokenizer, processor, batch_inputs: List[Dict[str, Any]]) -> List[Tuple[int, float, Optional[str], str]]:
    """
    Process a batch of inputs using the SGLang engine
    
    Args:
        llm: The VerlEngine instance
        batch_inputs: List of input dictionaries
        tokenizer: The tokenizer
        
    Returns:
        List of (index, score, response, ground_truth) tuples
    """
    results = []
    
    # Setup sampling parameters
    sampling_params = {
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        'max_new_tokens': 2048,
        'ignore_eos': False,
        'skip_special_tokens': True,
    }
    
    # Create a list to hold valid requests and their positions
    valid_requests = []
    valid_indices = []
    
    # Filter out error cases
    for i, input_data in enumerate(batch_inputs):
        if 'error' in input_data:
            # Add error result directly
            results.append((input_data['index'], 0.0, None, input_data['ground_truth']))
        else:
            # Add to valid requests
            valid_requests.append({
                'prompt': input_data['prompt'],
                'image_data': input_data['image_data'],
                'sampling_params': sampling_params
            })
            valid_indices.append(i)
    
    # If we have valid requests, process them as a batch
    if valid_requests:
        prompt_list = [input_data['prompt'] for input_data in valid_requests]
        image_list = [input_data['image_data'] for input_data in valid_requests]
        try:
            # Process batch with LLM
            # Note: For actual batched inference, you would use a batch API
            # This simulates batching by processing each valid request in sequence
            # breakpoint()
            inputs = processor(
                text=prompt_list,
                images=image_list,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")
            # import torch
            # device = next(llm.parameters()).device
            # for k, v in inputs.items():
            #     if isinstance(v, torch.Tensor):
            #         inputs[k] = v.to(device)
            # # breakpoint()
            generated_ids = llm.generate(
                    **inputs,
                    max_new_tokens=2048,
                    # temperature=0.0,
                    # top_p=1.0,
                    # min_p=0.0,
                    # do_sample=False,
                    # skip_special_tokens=True,
                    # spaces_between_special_tokens=True,
                    # presence_penalty=0.0,
                    # frequency_penalty=0.0,
                    # repetition_penalty=1.0,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed
            )
            # breakpoint()
            for i, response in enumerate(output_texts):
                batch_idx = valid_indices[i]
                input_data = batch_inputs[batch_idx]
                
                if response is None:
                    results.append((input_data['index'], 0.0, None, input_data['ground_truth']))
                else:
                    response_str = response
                    if response_str is None:
                        results.append((input_data['index'], 0.0, None, input_data['ground_truth']))
                    else:
                        # Compute score
                        cur_score = compute_score(response_str, input_data['ground_truth'])
                        logger.info(f"Row {input_data['index']}: score = {cur_score}")
                        logger.debug(f"debug:response_str: {response_str}")
                        results.append((input_data['index'], cur_score, response_str, input_data['ground_truth']))
        
        except Exception as e:
            import traceback
            error_msg = f"Error processing batch:\n"
            error_msg += f"Error type: {type(e).__name__}\n"
            error_msg += f"Error message: {str(e)}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # Add error results for all valid requests
            for i in valid_indices:
                input_data = batch_inputs[i]
                results.append((input_data['index'], 0.0, None, input_data['ground_truth']))
    
    return results


def process_dataframe_batched(df: pd.DataFrame, llm, tokenizer, processor, batch_size: int = BATCH_SIZE) -> list:
    """
    Processes the DataFrame in batches.
    
    Args:
        df: Input DataFrame
        llm: VerlEngine instance
        tokenizer: Tokenizer
        batch_size: Number of rows to process in each batch
        
    Returns:
        List of scores for each row
    """
    start_time = time.time()
    all_scores = [0.0] * len(df)
    all_results = []
    
    # Prepare batches
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    # Process rows in batches
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        # Get the current batch rows
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        
        # Prepare a list of (index, row) tuples for the current batch
        batch_rows = [(i, df.iloc[i]) for i in range(start_idx, end_idx)]
        
        # Prepare batch inputs
        batch_inputs = prepare_batch_inputs(batch_rows)
        
        # Process batch
        batch_results = process_batch(llm, tokenizer, processor, batch_inputs)
        
        # breakpoint()
        # assert(1 + 1 == 0)
        
        # Process results
        for index, score, response, ground_truth in batch_results:
            all_scores[index] = score
            all_results.append((index, score, response, ground_truth))
        
        # Log progress
        logger.info(f"Completed batch {batch_idx + 1}/{num_batches} ({end_idx}/{len(df)} rows)")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per row: {elapsed_time/len(df):.2f} seconds")
    logger.info(f"Average time per batch: {elapsed_time/num_batches:.2f} seconds")
    
    return all_scores


def main():
    parser = argparse.ArgumentParser(description='Run validation engine for geo3k dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for processing')
    args = parser.parse_args()
    
    # Setup logging with debug level if requested
    setup_logging()
    if args.debug:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.add(
            f".logs/geo3k_validation_transformers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="100 MB",
            retention="1 week",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            enqueue=True
        )
    
    logger.info(f"Starting batched processing job with Transformers (batch size: {args.batch_size})")

    llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/workspace/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="cuda:0"
    )

    tokenizer = AutoTokenizer.from_pretrained("/workspace/Qwen2.5-VL-7B-Instruct", device_map="cuda:0")
    processor = AutoProcessor.from_pretrained("/workspace/Qwen2.5-VL-7B-Instruct", device_map="cuda:0")

    # Load DataFrame
    df = pd.read_parquet(PARQUET_FILE_PATH)
    logger.info(f"Loaded dataframe with {len(df)} rows")
    if args.debug:
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"First row sample: {df.iloc[0].to_dict()}")

    # Process DataFrame in batches
    all_scores = process_dataframe_batched(df, llm, tokenizer, processor, args.batch_size)

    # Calculate and print the mean score
    scores = np.array(all_scores)
    mean_score = np.mean(scores)
    logger.info(f"The mean score is: {mean_score}")
    
    # Save detailed results to CSV
    try:
        results_df = pd.DataFrame({
            'score': all_scores
        })
        results_df.to_csv('evaluation_results_transformers.csv')
        logger.info("Results saved to evaluation_transformers.csv")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()