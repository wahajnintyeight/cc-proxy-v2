import logging
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, field_validator

from config import (
    BIG_MODEL,
    PREFERRED_PROVIDER,
    SMALL_MODEL,
    has_provider_prefix,
    resolve_model_name,
)

logger = logging.getLogger(__name__)


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None

    @field_validator("model")
    def validate_model_field(cls, v: str, info) -> str:
        original_model = v
        new_model, mapped = resolve_model_name(v)

        logger.debug(
            "📋 MODEL VALIDATION: Original='%s', Preferred='%s', BIG='%s', SMALL='%s'",
            original_model,
            PREFERRED_PROVIDER,
            BIG_MODEL,
            SMALL_MODEL,
        )

        if mapped:
            logger.debug("📌 MODEL MAPPING: '%s' ➡️ '%s'", original_model, new_model)
        else:
            if not has_provider_prefix(v):
                logger.warning("⚠️ No prefix or mapping rule for model: '%s'. Using as is.", original_model)
            new_model = v

        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model

        return new_model


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @field_validator("model")
    def validate_model_token_count(cls, v: str, info) -> str:
        original_model = v
        new_model, mapped = resolve_model_name(v)

        logger.debug(
            "📋 TOKEN COUNT VALIDATION: Original='%s', Preferred='%s', BIG='%s', SMALL='%s'",
            original_model,
            PREFERRED_PROVIDER,
            BIG_MODEL,
            SMALL_MODEL,
        )

        if mapped:
            logger.debug("📌 TOKEN COUNT MAPPING: '%s' ➡️ '%s'", original_model, new_model)
        else:
            if not has_provider_prefix(v):
                logger.warning(
                    "⚠️ No prefix or mapping rule for token count model: '%s'. Using as is.",
                    original_model,
                )
            new_model = v

        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model

        return new_model


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage
