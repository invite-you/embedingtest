"""Qwen/Qwen3-4B-Instruct-2507 호출을 담당하는 경량 래퍼."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol


logger = logging.getLogger(__name__)


class TransformerClientError(RuntimeError):
    """LLM 클라이언트 초기화/생성 오류."""


class SupportsGenerate(Protocol):
    def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class TransformerClientConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 128
    stop_strings: tuple[str, ...] = ()
    truncation_retry_max_new_tokens: int | None = None
    truncation_suffix: str = "(응답이 설정된 최대 토큰 {limit}에 도달하여 잘렸을 수 있습니다)"


class TransformerClient:
    """transformers 파이프라인을 통한 Qwen/Qwen3-4B-Instruct-2507 호출 래퍼."""

    def __init__(self, config: TransformerClientConfig | None = None):
        self.config = config or TransformerClientConfig()
        self._generator: SupportsGenerate | None = None
        self._tokenizer: Any | None = None

    def _load_pipeline(self) -> SupportsGenerate:
        if self._generator is not None:
            return self._generator
        try:
            from transformers import AutoTokenizer, pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise TransformerClientError(
                "transformers 라이브러리가 설치되어 있지 않습니다"
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception as exc:  # pragma: no cover - 토크나이저 초기화 오류
            raise TransformerClientError(
                f"토크나이저({self.config.model_name})를 초기화할 수 없습니다: {exc}"
            ) from exc

        try:
            self._generator = pipeline(
                "text-generation",
                model=self.config.model_name,
                device_map="auto",
                tokenizer=tokenizer,
            )
        except Exception as exc:  # pragma: no cover - 모델 초기화 오류
            raise TransformerClientError(
                f"모델({self.config.model_name})을 초기화할 수 없습니다: {exc}"
            ) from exc
        self._tokenizer = tokenizer
        return self._generator

    def generate(self, prompt: str) -> str:
        generator = self._load_pipeline()
        formatted_prompt = self._apply_chat_template_if_available(prompt)
        first_response, truncated = self._invoke_generator(
            generator, formatted_prompt, self.config.max_new_tokens
        )
        if truncated:
            logger.warning(
                "LLM 응답이 설정된 max_new_tokens=%s에 도달했습니다.",
                self.config.max_new_tokens,
            )
            retry_limit = self.config.truncation_retry_max_new_tokens
            if retry_limit and retry_limit > self.config.max_new_tokens:
                second_response, second_truncated = self._invoke_generator(
                    generator, formatted_prompt, retry_limit
                )
                if second_truncated:
                    return self._append_truncation_note(second_response, retry_limit)
                return second_response
            return self._append_truncation_note(
                first_response, self.config.max_new_tokens
            )
        return first_response

    def _invoke_generator(
        self, generator: SupportsGenerate, prompt: Any, max_new_tokens: int
    ) -> tuple[str, bool]:
        outputs = generator(prompt, **self._build_generation_kwargs(max_new_tokens))
        if not outputs:
            raise TransformerClientError("LLM 응답이 비어 있습니다")
        generated = outputs[0].get("generated_text", "")
        generated_text = generated if isinstance(generated, str) else str(generated)
        generated_text = generated_text.strip()
        token_count = self._extract_token_count(outputs[0], generated_text)
        truncated = max_new_tokens > 0 and token_count >= max_new_tokens
        return generated_text, truncated

    def _build_generation_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        if self._tokenizer is None:
            raise TransformerClientError("토크나이저가 초기화되지 않았습니다")

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
        }

        eos_token_ids = self._collect_eos_token_ids()
        if eos_token_ids:
            kwargs["eos_token_id"] = (
                eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]
            )
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        if self.config.stop_strings:
            kwargs["stop_strings"] = list(self.config.stop_strings)
        return kwargs

    def _collect_eos_token_ids(self) -> list[int]:
        if self._tokenizer is None:
            return []
        ids: list[int] = []
        for attr_name in ("eos_token_id", "im_end_id"):
            value = getattr(self._tokenizer, attr_name, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for token in value:
                    if isinstance(token, int) and token not in ids:
                        ids.append(token)
            elif isinstance(value, int) and value not in ids:
                ids.append(value)
        return ids

    def _apply_chat_template_if_available(self, prompt: str) -> Any:
        if self._tokenizer is None:
            return prompt
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {"add_generation_prompt": True}
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                **template_kwargs,
            )
        except TypeError:
            # 일부 토크나이저는 tokenize 인자를 지원하지 않는다.
            return apply_chat_template(messages, **template_kwargs)

    def _extract_token_count(self, output: dict[str, Any], text: str) -> int:
        generated_tokens = output.get("generated_tokens")
        if isinstance(generated_tokens, int):
            return generated_tokens
        if self._tokenizer is None:
            return 0
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def _append_truncation_note(self, text: str, limit: int) -> str:
        note = self.config.truncation_suffix.format(limit=limit)
        text = text.rstrip()
        if text and not text.endswith(" "):
            text = f"{text} "
        return f"{text}{note}".strip()
