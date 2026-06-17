from collections.abc import Iterator

from llama_cpp import Llama


def generate(
    model: Llama,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None = None,
) -> str:
    response = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    return response["choices"][0]["text"]


def stream_generate(
    model: Llama,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None = None,
) -> Iterator[str]:
    chunks = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        stream=True,
    )

    for chunk in chunks:
        text = chunk["choices"][0].get("text")
        if text:
            yield text
