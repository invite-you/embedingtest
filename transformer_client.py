"""Qwen/Qwen3-4B-Instruct-2507 호출을 담당하는 경량 래퍼."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class TransformerClientError(RuntimeError):
    """LLM 클라이언트 초기화/생성 오류."""


class SupportsGenerate(Protocol):
    def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class TransformerClientConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.95
    # Qwen3-2507 계열은 공식 문서 기준 256K 토큰(=262,144)을 처리할 수 있다.
    max_input_tokens: int = 262144


class TransformerClient:
    """transformers 파이프라인을 통한 Qwen/Qwen3-4B-Instruct-2507 호출 래퍼."""

    def __init__(self, config: TransformerClientConfig | None = None):
        self.config = config or TransformerClientConfig()
        self._generator: SupportsGenerate | None = None

    def _load_pipeline(self) -> SupportsGenerate:
        if self._generator is not None:
            return self._generator
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise TransformerClientError(
                "transformers 라이브러리가 설치되어 있지 않습니다"
            ) from exc

        try:
            self._generator = pipeline(
                "text-generation",
                model=self.config.model_name,
                device_map="auto",
            )
        except Exception as exc:  # pragma: no cover - 모델 초기화 오류
            raise TransformerClientError(
                f"모델({self.config.model_name})을 초기화할 수 없습니다: {exc}"
            ) from exc
        return self._generator

    def generate(self, prompt: str) -> str:
        generator = self._load_pipeline()
        tokenizer = getattr(generator, "tokenizer", None)
        eos_token_id = None
        pad_token_id = None
        if tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            # Qwen 시리즈는 `<|im_end|>` 토큰으로 응답을 닫는다. 토크나이저에 EOS/Pad가
            # 비어 있다면 수동으로 매핑해 모델이 해당 토큰을 생성했을 때 멈추도록 한다.
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if eos_token_id is None and convert is not None:
                for token in ("<|im_end|>", "</s>"):
                    token_id = convert(token)
                    if token_id is not None and token_id >= 0:
                        eos_token_id = token_id
                        break
            if pad_token_id is None:
                pad_token_id = eos_token_id
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": False,
        }
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        outputs = generator(prompt, **generation_kwargs)
        if not outputs:
            raise TransformerClientError("LLM 응답이 비어 있습니다")
        generated = outputs[0].get("generated_text", "")
        if generated.startswith(prompt):
            generated = generated[len(prompt) :]
        return generated.strip()
