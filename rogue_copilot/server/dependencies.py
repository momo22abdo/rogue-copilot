import os
from pathlib import Path

import faiss
from llama_cpp import Llama

from rogue_copilot.engine.loader import get_model
from rogue_copilot.rag.chunker import CodeChunk
from rogue_copilot.rag.store import load_index


_rag_index: faiss.Index | None = None
_rag_metadata: list[CodeChunk] | None = None


def get_model_dependency() -> Llama:
    return get_model()


def load_rag_index() -> None:
    global _rag_index, _rag_metadata

    index_path = Path(os.getenv("RAG_INDEX_PATH") or "./rag_index")
    if not index_path.with_suffix(".faiss").is_file():
        return

    _rag_index, _rag_metadata = load_index(index_path)


def get_rag_dependency() -> tuple[faiss.Index, list[CodeChunk]] | None:
    if _rag_index is None or _rag_metadata is None:
        return None

    return _rag_index, _rag_metadata
