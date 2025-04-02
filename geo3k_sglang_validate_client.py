import pandas as pd
from PIL import Image
import io
import numpy as np
import base64
from io import BytesIO
import re
import requests
from mathruler.grader import extract_boxed_content, grade_answer

# Constants
PARQUET_FILE_PATH = "/workspace/geo3k/test.parquet"
LLM_ENDPOINT = "http://localhost:39125/v1/chat/completions"
LLM_MODEL = "/workspace/Qwen2.5-VL-7B-Instruct"
MAX_TOKENS = 2048
ACC_REWARD_WEIGHT = 0.9
FORMAT_REWARD_WEIGHT = 0.1
DEMO_ROW_LIMIT = 0  # Set to a small number for demonstration, e.g., 4 for the first 5 rows


# ------------------- Utility Functions -------------------
def format_reward(predict_str: str) -> float:
    """
    Checks if the prediction string matches the expected format with <think> and \\boxed{}.
    """
    pattern = re.compile(r'<think>.*</think>.*\\boxed\{.*\}.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    Grades the answer extracted from the prediction string against the ground truth.
    """
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
                        "type": "image_data",
                        "image_data": {
                            "url": base64_image_uri
                        },
                    },
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
    }

    response = requests.post(LLM_ENDPOINT, json=data)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    return response.text # Or response.json() depending on the API


# ------------------- Main Processing Function -------------------
def process_dataframe(df: pd.DataFrame) -> list:
    """
    Processes each row of the DataFrame, calls the LLM, and computes the score.
    """
    all_scores = []
    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}")

        # Extract data from row
        choices = row['choices']
        ground_truth_data = row['ground_truth']
        correct_index = ord(ground_truth_data.upper()) - ord('A')
        ground_truth = choices[correct_index]
        problem_text = row.get('prompt')  # Use prompt column if available
        image_bytes = row['images'][0]['bytes']

        # Print data for debugging
        print(f"Choices: {choices}")
        print(f"Choice answer is : {ground_truth_data}")
        print(f"the ground_truth is {ground_truth}")
        print(f"prompt: {problem_text}")

        # Create base64 image URI
        base64_image_uri = create_base64_image_uri(image_bytes)

        # Call LLM
        try:
            response = "TBD" #call_llm(problem_text, base64_image_uri)
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM: {e}")
            cur_score = 0.0 # or some other default value
        else:
            # Compute score
            cur_score = compute_score(response, ground_truth)
            print(f"score: {cur_score}")
        finally:
            all_scores.append(cur_score)

        if index >= DEMO_ROW_LIMIT:
            print(f"\nStopping after processing the first {DEMO_ROW_LIMIT + 1} rows (for demonstration).")
            break

    return all_scores


# ------------------- Main Script -------------------
if __name__ == "__main__":
    # Load DataFrame
    df = pd.read_parquet(PARQUET_FILE_PATH)

    # Process DataFrame
    all_scores = process_dataframe(df)

    # Calculate and print the mean score
    scores = np.array(all_scores)
    mean_score = np.mean(scores)
    print(f"The mean score is: {mean_score}")