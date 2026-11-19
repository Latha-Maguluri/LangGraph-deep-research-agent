from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.core.config import settings


class LLMOutputTokenLimitError(RuntimeError):
    """The provider exhausted its output budget without producing text."""


class AnthropicChatModel:
    def __init__(self, model_name: str, api_key: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("anthropic package is required when using the Anthropic provider.") from exc

        self.model_name = model_name
        self.client = Anthropic(api_key=api_key)

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list):
            prompt = "\n".join(
                item.get("content", str(item)) if isinstance(item, dict) else str(item) for item in messages
            )
        else:
            prompt = str(messages)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 1200),
            messages=[{"role": "user", "content": prompt}],
        )
        response_content = getattr(response, "content", None) or []
        content = "\n".join(
            block.text
            for block in response_content
            if getattr(block, "type", None) == "text"
        )
        if not content:
            block_types = ", ".join(
                str(getattr(block, "type", type(block).__name__)) for block in response_content
            ) or "none"
            stop_reason = getattr(response, "stop_reason", None)
            error_type = LLMOutputTokenLimitError if stop_reason == "max_tokens" else RuntimeError
            raise error_type(
                "Anthropic returned no text content "
                f"(stop_reason={stop_reason!r}; content_block_types={block_types})."
            )
        return AIMessage(
            content=content,
            response_metadata={"stop_reason": getattr(response, "stop_reason", None)},
        )


class GeminiChatModel:
    def __init__(self, model_name: str, api_key: str) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-genai package is required when using the Gemini provider.") from exc

        self.model_name = model_name
        self.genai = genai
        self.client = genai.Client(api_key=api_key)

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list):
            prompt = "\n".join(
                item.get("content", str(item)) if isinstance(item, dict) else str(item) for item in messages
            )
        else:
            prompt = str(messages)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.genai.types.GenerateContentConfig(
                max_output_tokens=kwargs.get("max_tokens", 1200),
            ),
        )
        content = getattr(response, "text", None)
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        finish_reason_name = (
            getattr(finish_reason, "name", str(finish_reason)).lower()
            if finish_reason
            else None
        )
        if not content:
            error_type = LLMOutputTokenLimitError if finish_reason_name == "max_tokens" else RuntimeError
            raise error_type(
                "Gemini returned no text content "
                f"(finish_reason={finish_reason!r})."
            )

        stop_reason = "max_tokens" if finish_reason_name == "max_tokens" else finish_reason_name
        return AIMessage(content=content, response_metadata={"stop_reason": stop_reason})


class LLMFactory:
    @staticmethod
    def create_model(provider: str, model_name: str):
        resolved_provider = provider
        resolved_model = model_name

        if resolved_provider == "anthropic":
            api_key = settings.anthropic_api_key
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required when using the Anthropic provider.")
            return AnthropicChatModel(model_name=resolved_model, api_key=api_key)

        if resolved_provider == "gemini":
            api_key = settings.gemini_api_key
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is required when using the Gemini provider.")
            return GeminiChatModel(model_name=resolved_model, api_key=api_key)

        raise ValueError(f"Unsupported LLM provider: {resolved_provider}")
