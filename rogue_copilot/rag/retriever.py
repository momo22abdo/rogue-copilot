from rogue_copilot.rag.chunker import CodeChunk


def format_context(chunks: list[CodeChunk], max_chars: int) -> str:
    blocks = [
        f"### context\n# {chunk.file_path}::{chunk.name}\n{chunk.source}\n"
        for chunk in chunks
    ]
    return "\n\n".join(blocks)[:max_chars]
