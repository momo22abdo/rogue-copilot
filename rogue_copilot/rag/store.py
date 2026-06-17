import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from rogue_copilot.rag.chunker import CodeChunk
from rogue_copilot.rag.embedder import get_embedder


VECTOR_DIM = 384


def build_index(chunks: list[CodeChunk], save_path: Path) -> None:
    texts = [chunk.source for chunk in chunks]
    index = faiss.IndexFlatIP(VECTOR_DIM)

    if texts:
        embeddings = np.asarray(get_embedder().encode(texts), dtype=np.float32)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_path.with_suffix(".faiss")))

    with save_path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump([asdict(chunk) for chunk in chunks], file, ensure_ascii=False)


def load_index(save_path: Path) -> tuple[faiss.Index, list[CodeChunk]]:
    index = faiss.read_index(str(save_path.with_suffix(".faiss")))

    with save_path.with_suffix(".json").open(encoding="utf-8") as file:
        metadata = [CodeChunk(**item) for item in json.load(file)]

    return index, metadata


def query(
    index: faiss.Index,
    metadata: list[CodeChunk],
    query_text: str,
    k: int = 3,
) -> list[CodeChunk]:
    embedding = np.asarray(get_embedder().encode([query_text]), dtype=np.float32)
    faiss.normalize_L2(embedding)

    _, indices = index.search(embedding, k)

    return [metadata[index_id] for index_id in indices[0] if index_id >= 0]
