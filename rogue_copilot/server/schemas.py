from pydantic import BaseModel


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "library"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
