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

from sglang.srt.entrypoints.verl_engine import VerlEngine

# Constants
PARQUET_FILE_PATH = "/workspace/geo3k/test.parquet"
LLM_MODEL = "/workspace/Qwen2.5-VL-7B-Instruct"
MAX_TOKENS = 4096
ACC_REWARD_WEIGHT = 0.9
FORMAT_REWARD_WEIGHT = 0.1

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

def process_row(llm, index, row, tokenizer):
    """Process a single row using the SGLang engine"""
    try:
        # Extract data from row with detailed logging
        choices = row['choices']
        ground_truth_data = row['ground_truth']
        correct_index = ord(ground_truth_data.upper()) - ord('A')
        ground_truth = choices[correct_index]
        problem_text = row.get('prompt')[0].get('content')
        
        # 处理图片数据
        pil_image = [process_image(row['images'][0])]
        
        # 处理prompt
        prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + problem_text
        prompt = prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        prompt = prompt + '<|im_start|>assistant\n'
        
        #Prepare sampling parameters
        sampling_params = {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            'max_new_tokens':2048,
            'ignore_eos': False,
            'skip_special_tokens': True,
            'spaces_between_special_tokens': True,
            'n' : 1,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'repetition_penalty': 1.0,
        }

        # sampling_params = {
        #     "temperature": 0.0,
        #     "top_p": 1.0,
        #     "top_k": -1,
        #     "min_p": 0.0,
        #     "max_new_tokens": MAX_TOKENS,
        # }

        # breakpoint()
        # Generate response using the engine - synchronous version
        response = llm.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            image_data=pil_image,
            return_logprob=False,
        )
        
        # breakpoint()
        # Extract text from response
        if response is None:
            raise ValueError("Engine returned None response")
        response_str = response.get('text')
        if response_str is None:
            raise ValueError(f"Response missing 'text' key. Response keys: {response.keys()}")

        # Compute score
        cur_score = compute_score(response_str, ground_truth)
        logger.info(f"Row {index}: score = {cur_score}")
        logger.debug(f"debug:response_str: {response_str}")
        return index, cur_score, response_str, ground_truth
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing row {index}:\n"
        error_msg += f"Error type: {type(e).__name__}\n"
        error_msg += f"Error message: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return index, 0.0, None, "Error"


def process_dataframe_sequential(df: pd.DataFrame, llm, tokenizer) -> list:
    """
    Processes each row of the DataFrame sequentially.
    """
    start_time = time.time()
    all_scores = [0.0] * len(df)
    results = []

    # Process rows one by one
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        index, score, response, ground_truth = process_row(llm, i, row, tokenizer)
        all_scores[index] = score
        results.append((index, score, response, ground_truth))
        
        # Log progress periodically
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{len(df)} tasks")
        # if i >= 0:
        #     break

    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per row: {elapsed_time/len(df):.2f} seconds")
    
    return all_scores


def main():
    parser = argparse.ArgumentParser(description='Run validation engine for geo3k dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
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
            f"geo3k_validation_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="100 MB",
            retention="1 week",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            enqueue=True
        )
    
    logger.info("Starting sequential processing job with SGLang engine")
    
    # Initialize the SGLang engine
    logger.info("Initializing verl-SGLang engine...")
    os.environ.update({
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost", "MASTER_PORT": "12345"
    })
    tp_size = 1
    dp_size = 1
    device_mesh_kwargs = dict(
        mesh_shape=(tp_size, dp_size, 1), mesh_dim_names=["tp", "dp", "pp"]
    )    
    device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

    model_name, mem_fraction_static = LLM_MODEL, 0.6

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(model_name, trust_remote_code=True)

    llm = VerlEngine(
        model_path=model_name,
        mem_fraction_static=mem_fraction_static,
        device_mesh_cpu=device_mesh_cpu["tp"],
        base_gpu_id=0,
        gpu_id_step=1,
        port=30000,
        disable_cuda_graph=True,
        nnodes=1,
        log_level="INFO",
        log_requests=True,
        log_requests_level=2,
        max_running_requests=1,
    )

    try:
        # Load DataFrame
        df = pd.read_parquet(PARQUET_FILE_PATH)
        logger.info(f"Loaded dataframe with {len(df)} rows")
        if args.debug:
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"First row sample: {df.iloc[0].to_dict()}")

        # Process DataFrame sequentially
        all_scores = process_dataframe_sequential(df, llm, tokenizer)

        # Calculate and print the mean score
        scores = np.array(all_scores)
        mean_score = np.mean(scores)
        logger.info(f"The mean score is: {mean_score}")
        
        # Save detailed results to CSV
        try:
            results_df = pd.DataFrame({
                'score': all_scores
            })
            results_df.to_csv('evaluation_results_engine.csv')
            logger.info("Results saved to evaluation_results_engine.csv")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        logger.info("Processing completed successfully")
    
    finally:
        # Cleanup
        llm.shutdown()


if __name__ == "__main__":
    main()