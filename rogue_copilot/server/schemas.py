from pydantic import BaseModel


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "library"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    stream: bool = False
    stop: list[str] | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None
