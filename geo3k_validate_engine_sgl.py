"""
python geo3k_validate_engine.py --debug
"""

import pandas as pd
from PIL import Image
import io
import numpy as np
import base64
from io import BytesIO
import re
import json
import concurrent.futures
from mathruler.grader import extract_boxed_content, grade_answer
from loguru import logger
import time
from tqdm import tqdm
import argparse
import sys
from datetime import datetime
import sglang as sgl
import asyncio

# Constants
PARQUET_FILE_PATH = "/root/data/geo3k/test.parquet"
LLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_TOKENS = 4096
ACC_REWARD_WEIGHT = 0.9
FORMAT_REWARD_WEIGHT = 0.1
MAX_WORKERS = 32  # Number of parallel workers, adjust based on your system capabilities

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


def create_base64_image_uri(image_bytes: bytes) -> str:
    """
    Converts image bytes to a base64 data URI.
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    buffered = BytesIO()
    image_format = pil_image.format if pil_image.format else 'PNG'
    pil_image.save(buffered, format=image_format)
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode('utf-8')
    mime_type = f"image/{image_format.lower()}"
    return f"data:{mime_type};base64,{img_base64}"


async def process_row_with_engine(llm, row_data):
    """Process a single row using the SGLang engine"""
    index, row = row_data
    
    try:
        # Extract data from row with detailed logging
        choices = row['choices']
        ground_truth_data = row['ground_truth']
        correct_index = ord(ground_truth_data.upper()) - ord('A')
        ground_truth = choices[correct_index]
        problem_text = row.get('prompt')[0].get('content')
        
        image_bytes = row['images'][0]['bytes']
    
        # Create base64 image URI
        base64_image_uri = create_base64_image_uri(image_bytes)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": problem_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image_uri
                        },
                    },
                ],
            }
        ]

        formatted_prompt = llm.tokenizer_manager.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=None,
        )

        # Prepare sampling parameters
        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
        }

        # Generate response using the engine
        response = await llm.async_generate(
            prompt=formatted_prompt,
            sampling_params=sampling_params,
            image_data=base64_image_uri,
            return_logprob=False,
            logprob_start_len=None,
            top_logprobs_num=None,
            token_ids_logprob=None,
            lora_path=None,
            custom_logit_processor=None,
            stream=False
        )

        
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



async def process_dataframe_parallel(df: pd.DataFrame, llm) -> list:
    """
    Processes each row of the DataFrame in parallel using the SGLang engine.
    """
    start_time = time.time()
    all_scores = [0.0] * len(df)
    results = []

    # Create tasks for all rows
    tasks = [process_row_with_engine(llm, (i, row)) for i, row in df.iterrows()]
    
    # Process results as they complete
    for future in tqdm(asyncio.as_completed(tasks), total=len(df), desc="Processing rows"):
        try:
            index, score, response, ground_truth = await future
            all_scores[index] = score
            results.append((index, score, response, ground_truth))
            
            # Log progress periodically
            if len(results) % 10 == 0:
                logger.info(f"Completed {len(results)}/{len(df)} tasks")
        
        except Exception as e:
            logger.error(f"Error processing result: {e}")

    # Sort results by index for a clean log
    results.sort(key=lambda x: x[0])
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per row: {elapsed_time/len(df):.2f} seconds")
    
    return all_scores


async def main():
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
    
    logger.info("Starting parallel processing job with SGLang engine")
    
    # Initialize the SGLang engine
    logger.info("Initializing SGLang engine...")
    llm = sgl.Engine(model_path=LLM_MODEL, disable_radix_cache=True)
    
    try:
        # Load DataFrame
        df = pd.read_parquet(PARQUET_FILE_PATH)
        logger.info(f"Loaded dataframe with {len(df)} rows")
        if args.debug:
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"First row sample: {df.iloc[0].to_dict()}")

        # Process DataFrame in parallel
        all_scores = await process_dataframe_parallel(df, llm)

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
    asyncio.run(main()) 