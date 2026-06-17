import os
import time

import mlflow
import psutil
import requests


URL = "http://localhost:8000/v1/completions"
PROMPT = "def fibonacci(n):\n    "


def main() -> None:
    process = psutil.Process(os.getpid())
    payload = {
        "model": os.getenv("MODEL_NAME", "qwen-1.5b"),
        "prompt": PROMPT,
        "max_tokens": 50,
        "stream": False,
    }

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    start = time.perf_counter()
    response = requests.post(URL, json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    response.raise_for_status()

    usage = response.json()["usage"]
    completion_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]
    tokens_per_sec = completion_tokens / elapsed if elapsed else 0.0
    peak_ram_mb = process.memory_info().rss / 1024 / 1024

    with mlflow.start_run():
        mlflow.log_param("prompt_length", len(PROMPT))
        mlflow.log_metric("tokens_per_sec", tokens_per_sec)
        mlflow.log_metric("peak_ram_mb", peak_ram_mb)
        mlflow.log_metric("total_tokens", total_tokens)


if __name__ == "__main__":
    main()
