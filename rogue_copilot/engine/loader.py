import os
from pathlib import Path
from threading import Lock

from llama_cpp import Llama


_model: Llama | None = None
_model_lock = Lock()


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _load_model() -> Llama:
    model_path = Path(os.getenv("MODEL_PATH") or "./models/model.gguf")
    if not model_path.is_file():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    return Llama(
        model_path=str(model_path),
        n_ctx=_int_env("N_CTX", 4096),
        n_threads=_int_env("N_THREADS", os.cpu_count() or 1),
        n_gpu_layers=_int_env("N_GPU_LAYERS", 0),
    )


def get_model() -> Llama:
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = _load_model()

    return _model


def is_model_loaded() -> bool:
    return _model is not None
