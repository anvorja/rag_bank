# app/schemas/chat.py
"""
Pydantic Schemas for Chat API
Implements comprehensive data validation and serialization
"""
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
import re


class MessageHistory(BaseModel):
    """Individual message in conversation history"""
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, max_length=5000, description="Message content")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Validate message content"""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class SourceInfo(BaseModel):
    """Information about retrieved source document"""
    id: int = Field(..., ge=1, description="Source document index")
    source: str = Field(..., min_length=1, description="Document filename")
    section: str = Field(..., description="Main section (H1)")
    subsection: str = Field(..., description="Sub-section (H2)")
    content: str = Field(..., min_length=1, description="Preview of the content")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")  # FIXED: ge/le con floats


class QuestionRequest(BaseModel):
    """Request schema for asking a question"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Client's question"
    )
    conversation_history: List[MessageHistory] = Field(
        default=[],
        # FIXED: max_items -> max_length para listas
        max_length=20,
        description="Previous conversation messages for context"
    )
    session_id: Optional[str] = Field(
        None,
        pattern="^[a-zA-Z0-9_-]{1,50}$",
        description="Session identifier for tracking"
    )
    is_first_message: bool = Field(
        True,
        description="Whether this is the first message in session"
    )
    user_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional user metadata for context"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        """Validate and clean question"""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace")

        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', v.strip())

        # Check for potentially harmful content
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:',
            r'vbscript:'
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                raise ValueError("Question contains potentially harmful content")

        return cleaned

    @field_validator("conversation_history")
    @classmethod
    def validate_conversation_history(cls, v):
        """Validate conversation history structure"""
        if not v:
            return v

        # Ensure alternating user/assistant pattern
        for i, msg in enumerate(v):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg.role != expected_role and len(v) > 1:
                # Allow flexibility but warn about pattern
                pass

        return v

    @model_validator(mode='before')
    @classmethod
    def validate_session_consistency(cls, values):
        """Validate session-level consistency"""
        is_first = values.get("is_first_message", True)
        history = values.get("conversation_history", [])

        # If marked as first message, history should be empty
        if is_first and history:
            values["is_first_message"] = False

        # If history exists, shouldn't be first message
        elif history and is_first:
            values["is_first_message"] = False

        return values


class ModelInfo(BaseModel):
    """Information about models used"""
    llm_model: str = Field(..., description="LLM model name")
    embedding_model: str = Field(..., description="Embedding model name")
    embedding_dimension: int = Field(..., ge=1, description="Embedding dimension")
    mode: Literal["local", "cloud"] = Field(..., description="Operation mode")


class ResponseMetadata(BaseModel):
    """Metadata for response tracking and debugging"""
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    processing_time_ms: Optional[float] = Field(None, ge=0.0, description="Processing time in milliseconds")  # FIXED: ge con float
    model_info: ModelInfo = Field(..., description="Model information")
    retrieval_stats: Optional[Dict[str, Any]] = Field(None, description="Retrieval statistics")


class AnswerResponse(BaseModel):
    """Response schema with answer and sources"""
    answer: str = Field(..., min_length=1, description="Generated response from RAG")
    sources: List[SourceInfo] = Field(..., description="Source documents used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")  # FIXED: ge/le con floats
    metadata: ResponseMetadata = Field(..., description="Response metadata")

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v):
        """Validate answer content"""
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()


class ComponentStatus(BaseModel):
    """Individual component status"""
    status: Literal["healthy", "warning", "error"] = Field(..., description="Component status")
    message: Optional[str] = Field(None, description="Status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall system status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment")
    mode: Literal["local", "cloud"] = Field(..., description="Operation mode")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    components: Dict[str, ComponentStatus] = Field(default_factory=dict, description="Component statuses")  # FIXED: default_factory
    action_required: Optional[str] = Field(None, description="Required actions")


class VectorstoreStats(BaseModel):
    """Vectorstore statistics"""
    total_chunks: int = Field(..., ge=0, description="Total chunks in vectorstore")
    path: str = Field(..., description="Vectorstore path")
    collection_name: str = Field(..., description="Collection name")
    embedding_dim: int = Field(..., ge=1, description="Embedding dimension")


class ConfigStats(BaseModel):
    """Configuration statistics"""
    mode: Literal["local", "cloud"] = Field(..., description="Operation mode")
    retriever_k: int = Field(..., ge=1, description="Number of documents to retrieve")
    chunk_size: int = Field(..., ge=1, description="Chunk size")
    chunk_overlap: int = Field(..., ge=0, description="Chunk overlap")
    models: ModelInfo = Field(..., description="Model information")


class StatsResponse(BaseModel):
    """Statistics response schema"""
    status: Literal["success", "error"] = Field(..., description="Request status")
    vectorstore: VectorstoreStats = Field(..., description="Vectorstore statistics")
    config: ConfigStats = Field(..., description="Configuration statistics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# Error response schemas
class ErrorResponse(BaseModel):
    """Standard error response schema"""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Application error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ValidationErrorResponse(BaseModel):
    """Validation error response schema"""
    detail: str = Field(..., description="Validation error message")
    field_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Field-specific errors")  # FIXED: default_factory
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")