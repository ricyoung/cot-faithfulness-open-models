"""
API client for OpenRouter. All 14 models use the same client.
Uses the OpenAI SDK with OpenRouter's base URL.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cot_faithfulness.config import MODEL_REGISTRY, ModelConfig, Settings

_TRANSIENT_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


# ---------------------------------------------------------------------------
# Thinking extraction
# ---------------------------------------------------------------------------

# Patterns for extracting thinking from model output
THINK_TAG_PATTERN = re.compile(
    r"<think>(.*?)</think>", re.DOTALL
)
PIPE_THINK_PATTERN = re.compile(
    r"<\|thinking\|>(.*?)<\|/thinking\|>", re.DOTALL
)


def split_thinking(text: str) -> tuple[str, str]:
    """
    Extract thinking/reasoning from model output text.

    Models return reasoning in different formats:
    - <think>...</think> tags (DeepSeek, Qwen)
    - <|thinking|>...</|thinking|> tags (some models)
    - No tags (reasoning returned via API field instead)

    Returns (thinking_text, answer_text).
    """
    # Try <think>...</think>
    blocks = THINK_TAG_PATTERN.findall(text)
    if blocks:
        thinking = "\n\n".join(b.strip() for b in blocks)
        answer = THINK_TAG_PATTERN.sub("", text).strip()
        return thinking, answer

    # Try <|thinking|>...</|thinking|>
    blocks = PIPE_THINK_PATTERN.findall(text)
    if blocks:
        thinking = "\n\n".join(b.strip() for b in blocks)
        answer = PIPE_THINK_PATTERN.sub("", text).strip()
        return thinking, answer

    # No tags found — all text is the answer
    return "", text


# ---------------------------------------------------------------------------
# OpenRouter Client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """Client for all models via OpenRouter API."""

    def __init__(
        self,
        model_name: str,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or Settings()
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.config: ModelConfig = MODEL_REGISTRY[model_name]
        self.client = OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.api.base_url,
            timeout=self.settings.api.request_timeout,
        )

        _retry = retry(
            retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
            stop=stop_after_attempt(self.settings.api.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.settings.api.retry_min_wait,
                max=self.settings.api.retry_max_wait,
            ),
            reraise=True,
        )
        self.generate = _retry(self._generate_impl)

    def _generate_impl(self, user_message: str) -> dict:
        """
        Send a prompt to OpenRouter and return parsed response.

        Returns dict with keys:
        - full_response: str (complete text output)
        - thinking_text: str (reasoning/thinking portion)
        - answer_text: str (final answer portion)
        - latency_seconds: float
        - input_tokens: int
        - output_tokens: int
        - reasoning_tokens: int
        """
        start = time.monotonic()

        kwargs: dict = {
            "model": self.config.api_model_id,
            "messages": [
                {"role": "system", "content": self.settings.generation.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.settings.generation.temperature,
            "seed": self.settings.generation.seed,
            "max_tokens": self.config.max_completion_tokens,
        }

        # Request reasoning tokens for models that support it
        if self.config.supports_reasoning:
            kwargs["extra_body"] = {
                "reasoning": {"effort": "high"},
            }

        response = self.client.chat.completions.create(**kwargs)
        latency = time.monotonic() - start

        message = response.choices[0].message
        full_text = message.content or ""

        # Try to get reasoning from the API response field.
        # OpenRouter returns reasoning in `message.reasoning` (not `reasoning_content`)
        reasoning_field = (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
            or ""
        )

        if reasoning_field:
            thinking_text = reasoning_field
            answer_text = full_text
        else:
            # Fall back to parsing tags from the text
            thinking_text, answer_text = split_thinking(full_text)

        # Token counts — OpenRouter nests reasoning_tokens under completion_tokens_details
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        reasoning_tokens = 0
        if usage:
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd:
                reasoning_tokens = getattr(ctd, "reasoning_tokens", 0) or 0
            if not reasoning_tokens:
                reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0

        return {
            "full_response": full_text,
            "thinking_text": thinking_text,
            "answer_text": answer_text,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
        }


def get_client(
    model_name: str,
    settings: Optional[Settings] = None,
) -> OpenRouterClient:
    """Factory: create a client for the given model."""
    return OpenRouterClient(model_name, settings)
