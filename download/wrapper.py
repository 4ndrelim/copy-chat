import os
import subprocess

with open("download.env") as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        os.environ[key] = value

cmd = f"./download.sh"

print("Running:")
print(cmd)
subprocess.run(cmd.split(), check=True)

