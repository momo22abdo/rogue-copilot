from llama_cpp import Llama

from rogue_copilot.engine.loader import get_model


def get_model_dependency() -> Llama:
    return get_model()
