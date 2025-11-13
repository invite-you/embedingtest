"""임베딩 모델 초기화와 텍스트 변환 도우미를 제공합니다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class EmbeddingConfig:
    """임베딩 모델 관련 설정입니다."""

    model_name: str = "qwen3-embedding-4b"
    device: str | None = None
    max_length: int = 512
    preview_values: int = 5


def _load_embedding_components(model_name: str):
    """토크나이저와 모델을 지연 로딩 방식으로 가져옵니다."""

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - 외부 의존성 검증
        raise RuntimeError(
            "transformers 패키지가 필요합니다. 'pip install transformers torch'로 설치해주세요."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


class TextEmbedder:
    """선택한 파일 텍스트를 임베딩하는 도우미입니다."""

    def __init__(
        self,
        config: EmbeddingConfig,
        loader: Callable[[str], Tuple[object, object]] = _load_embedding_components,
        torch_module: object | None = None,
    ) -> None:
        self.config = config
        self.tokenizer, self.model = loader(config.model_name)
        if torch_module is None:
            try:
                import torch as torch_module  # type: ignore
            except ImportError as exc:  # pragma: no cover - 외부 의존성 검증
                raise RuntimeError(
                    "torch 패키지가 필요합니다. 'pip install torch'로 설치해주세요."
                ) from exc

        self._torch = torch_module
        device = config.device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> list[float]:
        """단일 문자열을 임베딩 벡터로 변환합니다."""

        if not text.strip():
            raise ValueError("비어 있는 텍스트는 임베딩할 수 없습니다.")

        tokenizer = self.tokenizer
        torch = self._torch
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)

        if hasattr(outputs, "last_hidden_state"):
            hidden_state = outputs.last_hidden_state
        elif hasattr(outputs, "pooler_output"):
            hidden_state = outputs.pooler_output
        else:  # pragma: no cover - 정상 모델 경로에서는 발생하지 않음
            raise RuntimeError("임베딩 결과를 확인할 수 없습니다.")

        embedding = hidden_state.mean(dim=1).squeeze().detach().cpu().tolist()
        if isinstance(embedding, float):  # 단일 값이라면 리스트로 감쌉니다.
            return [embedding]
        return embedding
