# app/schemas/chat.py
from pydantic import BaseModel, Field, field_validator


class SourceInfo(BaseModel):
    id: int
    source: str
    section: str
    subsection: str
    content: str
    chunk_id: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    session_id: str | None = None
    is_first_message: bool = True

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict
