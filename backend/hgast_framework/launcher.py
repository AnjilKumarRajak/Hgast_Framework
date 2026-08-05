import subprocess
import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, default="seamless_m4t")
args = parser.parse_args()

print(f"Launching full dataset batch evaluation for {args.backbone} in background...")

cmd = [
    "python3",
    "hgast_framework/run_batch_evaluation.py",
    "--backbone", args.backbone,
    "--limit", "0"
]

env = os.environ.copy()
env["PYTHONPATH"] = "."

with open("full_dataset_run.log", "w") as logfile:
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=logfile,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

print(f"Successfully launched with PID: {p.pid}")
print("You can check progress by reading full_dataset_run.log")
