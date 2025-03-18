import os
import argparse
from datetime import datetime
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil


log_file = "download_log.out"

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
        f.write(
            (
                "------------------------------------------------------------------------------------------------\n"
                f"{start_time}: Downloading {model_name} started."
            )
        )

    # --- Define save dir ---
    os.makedirs(save_model_directory, exist_ok=True)

    # --- Remove whatever existing stuff ---
    # shutil.rmtree(save_model_directory) # comment this out if setting resume_download=True

    # --- Download ---
    try:
        snapshot_download(repo_id=model_name, local_dir=save_model_directory, token=huggingface_token, resume_download=True)

        with open(log_file, 'a') as f:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{end_time}: Successfully downloaded '{model_name}' to '{save_model_directory}'\n")
    except Exception as e:
        with open(log_file, 'a') as f:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{end_time}: Error downloading model: {e}\n")
        raise

    finally:
        with open(log_file, 'a') as f:
            f.write("------------------------------------------------------------------------------------------------\n")
