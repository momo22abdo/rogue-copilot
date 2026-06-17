import json
import os
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from rogue_copilot.engine import generate, stream_generate
from rogue_copilot.engine.loader import is_model_loaded
from rogue_copilot.server.dependencies import get_model_dependency
from rogue_copilot.server.schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
    ModelCard,
    ModelList,
)


router = APIRouter()


def _completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex}"


def _token_count(text: str) -> int:
    return len(text.split())


@router.get("/health")
async def health() -> dict[str, bool | str]:
    return {"status": "ok", "model_loaded": is_model_loaded()}


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    model_name = os.getenv("MODEL_NAME") or "qwen-1.5b"

    return ModelList(
        data=[
            ModelCard(
                id=model_name,
                created=int(time.time()),
            )
        ]
    )


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    model=Depends(get_model_dependency),
) -> CompletionResponse | StreamingResponse:
    completion_id = _completion_id()

    if not request.stream:
        text = generate(
            model=model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
        )
        prompt_tokens = _token_count(request.prompt)
        completion_tokens = _token_count(text)

        return CompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=text,
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def event_stream() -> AsyncIterator[str]:
        for chunk in stream_generate(
            model=model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
        ):
            payload = {
                "id": completion_id,
                "object": "text_completion",
                "choices": [{"text": chunk, "index": 0}],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
