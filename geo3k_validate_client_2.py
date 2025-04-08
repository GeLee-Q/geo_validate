import pandas as pd
from PIL import Image
import io
import numpy as np
import base64
from io import BytesIO
import re
import requests
import json
import concurrent.futures
from mathruler.grader import extract_boxed_content, grade_answer
from loguru import logger
import time
from tqdm import tqdm
import argparse
import sys
from datetime import datetime

# Constants
PARQUET_FILE_PATH = "/root/data/geo3k/test.parquet"
LLM_ENDPOINT = "http://localhost:39249/v1/chat/completions"
LLM_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
MAX_TOKENS = 4096
ACC_REWARD_WEIGHT = 0.9
FORMAT_REWARD_WEIGHT = 0.1
MAX_WORKERS =32  # Number of parallel workers, adjust based on your system capabilities
REQUEST_TIMEOUT = 300  # Timeout for each request in seconds

# Configure logging
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"geo3k_validation_{timestamp}.log"
    
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
    image_format = pil_image.format if pil_image.format else 'PNG'  # Preserve original format or default to PNG
    pil_image.save(buffered, format=image_format)
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode('utf-8')
    mime_type = f"image/{image_format.lower()}"
    return f"data:{mime_type};base64,{img_base64}"


def call_llm(problem_text: str, base64_image_uri: str) -> str:
    """
    Calls the LLM endpoint with the problem text and image.
    """
    data = {
        "model": LLM_MODEL,
        "messages": [
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
        ],
        "best_of": 1,
        'top_p': 1.0,
        'top_k': -1,
        'min_p': 0.0,
        'temperature': 0,
        'n': 1
    }
    
    response = requests.post(LLM_ENDPOINT, json=data, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    return response.text  # Or response.json() depending on the API


def extract_content(json_string: str) -> str | None:
    """Extracts the 'content' from the LLM's JSON response."""
    try:
        data = json.loads(json_string)
        return data['choices'][0]['message']['content'] if data['choices'] else None
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
        logger.error(f"Error extracting content: {e}, Response: {json_string[:100]}...")
        return None


# ------------------- Main Processing Function -------------------
def process_row(row_data):
    """Process a single row from the dataframe"""
    index, row = row_data
    
    try:
        # Extract data from row
        choices = row['choices']
        ground_truth_data = row['ground_truth']
        correct_index = ord(ground_truth_data.upper()) - ord('A')
        ground_truth = choices[correct_index]
        problem_array = row.get('prompt')  # Use prompt column if available
        problem_text = problem_array[0]['content']
        image_bytes = row['images'][0]['bytes']

        # Create base64 image URI
        base64_image_uri = create_base64_image_uri(image_bytes)

        # Call LLM
        try:
            response = call_llm(problem_text, base64_image_uri)
            response_str = extract_content(response)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LLM for row {index}: {e}")
            return index, 0.0, None, ground_truth
        except Exception as e:
            logger.error(f"Unexpected error for row {index}: {e}")
            return index, 0.0, None, ground_truth

        # Compute score
        cur_score = compute_score(response_str, ground_truth)
        
        logger.info(f"Row {index}: score = {cur_score}")
        logger.info(f"Row {index}: response_str = {response_str}")
        return index, cur_score, response_str, ground_truth
    
    except Exception as e:
        logger.error(f"Error processing row {index}: {e}")
        return index, 0.0, None, "Error"


def process_dataframe_parallel(df: pd.DataFrame) -> list:
    """
    Processes each row of the DataFrame in parallel, calls the LLM, and computes the score.
    """
    start_time = time.time()
    all_scores = [0.0] * len(df)  # Pre-allocate results list
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_row, (i, row)): i 
            for i, row in df.iterrows()
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(df), desc="Processing rows"):
            try:
                index, score, response, ground_truth = future.result()
                all_scores[index] = score
                results.append((index, score, response, ground_truth))
                
                # Log progress periodically
                if len(results) % 10 == 0:
                    logger.info(f"Completed {len(results)}/{len(df)} tasks")
            
            except Exception as e:
                idx = future_to_index[future]
                logger.error(f"Error processing result for row {idx}: {e}")
                all_scores[idx] = 0.0

    # Sort results by index for a clean log
    results.sort(key=lambda x: x[0])
    
    # # Log detailed results
    # for index, score, response, ground_truth in results:
    #     logger.info(f"Row {index} - Score: {score}")
    #     logger.debug(f"Row {index} - Response: {response[:100]}...")
    #     logger.debug(f"Row {index} - Ground truth: {ground_truth}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per row: {elapsed_time/len(df):.2f} seconds")
    
    return all_scores


# ------------------- Main Script -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run validation client for geo3k dataset')
    parser.add_argument('--port', type=int, default=8080, help='Port number for the LLM server')
    args = parser.parse_args()
    LLM_ENDPOINT = f"http://localhost:{args.port}/v1/chat/completions"
    
    # Setup logging
    setup_logging()
    logger.info("Starting parallel processing job")
    
    # Load DataFrame
    df = pd.read_parquet(PARQUET_FILE_PATH)
    logger.info(f"Loaded dataframe with {len(df)} rows")

    # Process DataFrame in parallel
    all_scores = process_dataframe_parallel(df)

    # Calculate and print the mean score
    scores = np.array(all_scores)
    mean_score = np.mean(scores)
    logger.info(f"The mean score is: {mean_score}")
    
    # Save detailed results to CSV
    try:
        results_df = pd.DataFrame({
            'score': all_scores
        })
        results_df.to_csv('evaluation_results.csv')
        logger.info("Results saved to evaluation_results.csv")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    logger.info("Processing completed successfully")