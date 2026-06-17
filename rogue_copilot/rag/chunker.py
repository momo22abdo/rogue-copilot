import ast
import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeChunk:
    source: str
    file_path: str
    name: str
    kind: str
    lineno: int


def parse_file(path: Path) -> list[CodeChunk]:
    content = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        logging.warning("Failed to parse %s: %s", path, exc)
        return []

    chunks: list[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue

        source = ast.get_source_segment(content, node)
        if not source or len(source.splitlines()) < 3:
            continue

        chunks.append(
            CodeChunk(
                source=source,
                file_path=str(path),
                name=node.name,
                kind=kind,
                lineno=node.lineno,
            )
        )

    return chunks
