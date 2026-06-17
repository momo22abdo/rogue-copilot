import os
import time

from fastapi import APIRouter

from rogue_copilot.server.schemas import ModelCard, ModelList


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, bool | str]:
    return {"status": "ok", "model_loaded": False}


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
