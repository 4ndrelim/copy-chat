import os
import subprocess

# Read environment variables from powershell.env
with open("powershell.env") as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        os.environ[key] = value

save_dir = os.environ["SAVE_DIR"]
# print(f"{save_dir=}")
# save_dir = r"C:\Users\hcich\Documents\GitHub\copy-chat\models"
# os.makedirs(save_dir, exist_ok=True)

print("Running:")
print(
    f"python download.py --model_name Qwen/Qwen2.5-7B-Instruct --save_dir".split()
    + [save_dir]
)
subprocess.run(
    f"python download.py --model_name Qwen/Qwen2.5-7B-Instruct --save_dir".split()
    + [save_dir],
    check=True,
)
