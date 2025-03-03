import os
import argparse
from datetime import datetime
from huggingface_hub import snapshot_download
from pathlib import Path


log_file = "error_log.txt"

if __name__ == "__main__":
    # --- Config ---
    parser = argparse.ArgumentParser(description="Download a Hugging Face model.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--save_dir", type=str, default="./", help="Folder where you want the model to be stored (e.g. usr/models)")
    args = parser.parse_args()
    model_name = args.model_name
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    save_model_directory = str((Path(save_dir) / f"{model_name.replace('/', '-')}").resolve())
    huggingface_token = os.getenv("HF_TOKEN")  # HF token

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"Downloading {model_name} started at {start_time}")

    # --- Define save dir ---
    os.makedirs(save_model_directory, exist_ok=True)

    # --- Download ---
    try:
        snapshot_download(repo_id=model_name, local_dir=save_model_directory, token=huggingface_token)

        with open(log_file, 'a') as f:
            f.write(f"Model '{model_name}' downloaded successfully to '{save_model_directory}'\n")
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Error downloading model: {e}\n")
