"""Launch the pinned vLLM container on a Linux NVIDIA GPU host."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--print-command", action="store_true")
args = parser.parse_args()
config = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/vllm.json").read_text()
)
command = [
    "docker",
    "run",
    "--rm",
    "--gpus",
    "all",
    "--shm-size",
    "8g",
    "--publish",
    "127.0.0.1:8000:8000",
    "--env",
    "VLLM_API_KEY",
    "--env",
    "HF_HUB_DISABLE_TELEMETRY=1",
    "--volume",
    "inference-load-model-cache:/root/.cache/huggingface",
    config["image"],
    *config["arguments"],
]
if args.print_command:
    print(shlex.join(command))
else:
    if not os.environ.get("VLLM_API_KEY"):
        parser.error("Set VLLM_API_KEY; Docker receives it through the environment.")
    raise SystemExit(subprocess.run(command, check=False).returncode)
