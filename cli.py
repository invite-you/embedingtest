"""설정한 확장자만 읽어 출력하는 CLI입니다.

하나의 파일 경로를 전달하면 해당 파일을 검증 후 출력하고,
폴더 경로를 전달하면 재귀적으로 대상 확장자 파일을 찾아 사전순으로 출력합니다.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, List, Protocol, Sequence, Tuple


def _normalize_extensions(extensions: Sequence[str]) -> tuple[str, ...]:
    """확장자를 소문자로 통일하고 누락된 점(`.`)을 보완합니다."""

    normalized: list[str] = []
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(normalized)


def _default_device() -> str:
    """가용한 CUDA가 없으면 CPU를 기본값으로 선택합니다."""

    try:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - 환경 의존 코드
            return "cuda"
    except Exception:  # pragma: no cover - 방어적 코드
        pass
    return "cpu"


def _default_torch_dtype(device: str) -> Any | None:
    """디바이스 유형에 따라 적절한 기본 torch dtype을 고릅니다."""

    try:
        import torch

        if device.startswith("cuda"):
            return torch.float16
        return torch.float32
    except Exception:  # pragma: no cover - torch 미설치 환경 방어
        return None


@dataclass(frozen=True)
class CLIConfig:
    """CLI 동작에 필요한 설정 모음입니다."""

    allowed_extensions: tuple[str, ...]
    preferred_encodings: tuple[str, ...]
    embedding_model_name: str
    device: str
    torch_dtype: Any | None


_CLI_DEFAULT_DEVICE = _default_device()

CLI_CONFIG = CLIConfig(
    allowed_extensions=_normalize_extensions([".txt"]),
    preferred_encodings=("utf-8", "cp949"),
    embedding_model_name="qwen/qwen3-embedding-4b",
    device=_CLI_DEFAULT_DEVICE,
    torch_dtype=_default_torch_dtype(_CLI_DEFAULT_DEVICE),
)


def _validate_device_choice(device: str, torch_module: Any) -> str:
    """지정된 디바이스가 사용 가능한지 확인합니다."""

    if device.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA 디바이스를 요청했지만 torch.cuda.is_available()이 False입니다."
        )
    return device


def _torch_dtype_from_string(value: str) -> Any:
    """문자열 입력을 torch dtype 객체로 변환합니다."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - torch 미설치 방어
        raise argparse.ArgumentTypeError(
            "torch 모듈을 찾을 수 없어 dtype을 지정할 수 없습니다."
        ) from exc

    normalized = value.strip()
    if normalized.startswith("torch."):
        normalized = normalized.split(".", 1)[1]
    dtype = getattr(torch, normalized, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise argparse.ArgumentTypeError(
            f"지원하지 않는 torch dtype입니다: {value}"
        )
    return dtype


def _is_allowed_extension(path: Path, config: CLIConfig) -> bool:
    """파일 확장자가 허용 목록에 있는지 검사합니다."""

    return path.suffix.lower() in config.allowed_extensions


def find_target_files(target: Path, config: CLIConfig = CLI_CONFIG) -> List[Path]:
    """입력 경로에 해당하는 허용 확장자 파일 목록을 반환합니다."""

    if not target.exists():
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {target}")

    if target.is_file():
        if not _is_allowed_extension(target, config):
            allowed = ", ".join(config.allowed_extensions)
            raise ValueError(f"허용된 확장자가 아닙니다 ({allowed}): {target}")
        return [target]

    if not target.is_dir():
        raise ValueError(f"지원하지 않는 경로 형식입니다: {target}")

    files = [
        path
        for path in target.rglob("*")
        if path.is_file() and _is_allowed_extension(path, config)
    ]
    return sorted(files, key=lambda p: p.as_posix())


def read_file(path: Path, config: CLIConfig = CLI_CONFIG) -> str:
    """설정된 인코딩 후보를 순회하며 텍스트를 반환합니다."""

    last_error: UnicodeDecodeError | None = None
    for encoding in config.preferred_encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise UnicodeDecodeError("", b"", 0, 0, "사용 가능한 인코딩이 없습니다")


SegmentEmbedding = Tuple[str, Sequence[float]]


def _now_timestamp() -> str:
    """사람이 읽기 쉬운 현재 시각 문자열을 생성합니다."""

    return datetime.now().isoformat(timespec="milliseconds")


class StageTimer:
    """단계별 경과 시간을 추적해 병목을 파악할 수 있도록 돕습니다."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._last = self._start

    def mark(self, stage: str) -> None:
        """특정 단계가 끝났음을 타임스템프와 함께 출력합니다."""

        current = time.perf_counter()
        delta = current - self._last
        total = current - self._start
        self._last = current
        print(f"[{_now_timestamp()}] {stage} (Δ{delta:.3f}s, Σ{total:.3f}s)")


class EmbeddingServiceProtocol(Protocol):
    """임베딩 백엔드가 따라야 하는 최소 인터페이스입니다."""

    model_name: str

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """여러 텍스트를 한 번에 벡터로 변환합니다."""

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트를 벡터로 변환합니다."""


class EmbeddingService:
    """Transformers 기반 임베딩 서비스 구현체입니다."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        torch_dtype: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device_preference = (device or "cpu").lower()
        self._resolved_device: str | None = None
        self._autocast_device_type: str | None = None
        self._torch_dtype = torch_dtype

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer  # 지연 임포트
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }

        requested_device = self._device_preference
        device_map: Any | None = None
        resolved_device: str | None = None

        if requested_device == "auto":
            device_map = "auto"
        elif requested_device.startswith("cuda"):
            resolved_device = _validate_device_choice(requested_device, torch)
            device_map = {"": resolved_device}
        else:
            resolved_device = requested_device

        if device_map is not None:
            model_kwargs["device_map"] = device_map
            model_kwargs["low_cpu_mem_usage"] = True

        self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs)

        dtype = self._torch_dtype
        if device_map is None:
            to_kwargs: dict[str, Any] = {}
            if resolved_device is not None:
                to_kwargs["device"] = resolved_device
            if dtype is not None:
                to_kwargs["dtype"] = dtype
            if to_kwargs:
                self._model.to(**to_kwargs)
        else:
            if dtype is not None:
                self._model.to(dtype=dtype)

        def _infer_model_device() -> str | None:
            device_attr = getattr(self._model, "device", None)
            if device_attr is not None:
                return str(device_attr)
            hf_device_map = getattr(self._model, "hf_device_map", None)
            if isinstance(hf_device_map, dict) and hf_device_map:
                first_target = next(iter(hf_device_map.values()))
                if isinstance(first_target, int):
                    return f"cuda:{first_target}"
                return str(first_target)
            return resolved_device

        inferred_device = _infer_model_device()
        self._resolved_device = inferred_device
        if inferred_device and inferred_device.startswith("cuda"):
            self._autocast_device_type = "cuda"
        self._model.eval()
        self._torch = torch

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("임베딩할 텍스트가 비어 있습니다.")
        if any(not text.strip() for text in texts):
            raise ValueError("공백만 있는 텍스트는 임베딩할 수 없습니다.")

        self._ensure_model()
        assert self._tokenizer is not None
        assert self._model is not None
        torch = self._torch
        if torch is None:  # pragma: no cover - 방어적 코드
            raise RuntimeError("PyTorch 모듈이 초기화되지 않았습니다.")

        # 1) 토크나이저 전처리 (배치 입력)
        # - 여러 문장을 한 번에 토큰화해 GPU/CPU 호출 횟수를 줄입니다.
        # - `padding=True`로 가장 긴 문장에 맞춰 나머지에 패딩을 추가합니다.
        # - `truncation=True`, `max_length=512`는 각 문장의 토큰 수 상한을 보장합니다.
        inputs = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        if self._resolved_device is not None:
            inputs = inputs.to(self._resolved_device)

        # 2) 추론 전용 실행
        # - `torch.inference_mode()`로 감싸 그래디언트를 완전히 비활성화합니다.
        # - CUDA 디바이스라면 `torch.autocast`로 dtype을 줄여 메모리를 절약합니다.
        if self._autocast_device_type:
            autocast_kwargs: dict[str, Any] = {
                "device_type": self._autocast_device_type,
            }
            if self._torch_dtype is not None:
                autocast_kwargs["dtype"] = self._torch_dtype
            autocast_context = torch.autocast(**autocast_kwargs)
        else:
            autocast_context = nullcontext()
        with torch.inference_mode():
            with autocast_context:
                outputs = self._model(**inputs)

        # 3) 모델 출력 검증
        # - 신뢰할 수 없는 커스텀 모델도 사용할 수 있으므로 필수 필드가
        #   없으면 명시적으로 실패하게 만들어 조기에 문제를 드러냅니다.
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("모델 출력에 last_hidden_state가 없습니다.")

        # 4) 마스킹 평균 풀링
        # - `last_hidden_state`는 (배치, 토큰, 은닉차원) 형태의 텐서이며,
        #   `attention_mask`로 실제 토큰만 남긴 뒤 토큰별 벡터를 평균 내면
        #   문장 길이에 관계없는 대표 임베딩을 얻을 수 있습니다.
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked = last_hidden_state * attention_mask
        sum_embeddings = masked.sum(dim=1)
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_embeddings = sum_embeddings / token_counts

        # 5) L2 정규화
        # - 각 벡터를 단위 길이로 맞추면 코사인 유사도 계산 시 안정성이 높고
        #   문장 길이나 스케일 차이가 줄어듭니다.
        normalized = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
        return normalized.tolist()

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 입력을 위한 헬퍼입니다."""

        return self.embed_texts([text])[0]


def build_embedding_payload(
    file_path: Path,
    segment_embeddings: Sequence[SegmentEmbedding],
    model_name: str,
) -> dict:
    """분할된 텍스트 각각의 임베딩을 포함한 JSON 페이로드를 구성합니다."""

    segments = [
        {
            "index": index,
            "text": text,
            "vector": list(vector),
            "vector_length": len(vector),
        }
        for index, (text, vector) in enumerate(segment_embeddings)
    ]
    return {
        "source_path": str(file_path),
        "model_name": model_name,
        "segment_count": len(segments),
        "segments": segments,
    }


_SENTENCE_DELIMITER = re.compile(r"(?<=[.!?。？！])\s+")


def split_text_by_newline_or_sentence(text: str) -> list[str]:
    """줄바꿈 우선으로 분할하되 문장 부호를 기준으로도 세분화합니다."""

    segments: list[str] = []
    for line in text.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        # 문장 부호를 기준으로 추가 분리합니다. (마침표, 느낌표 등)
        parts = _SENTENCE_DELIMITER.split(cleaned_line)
        for part in parts:
            cleaned_part = part.strip()
            if cleaned_part:
                segments.append(cleaned_part)

    if not segments and text.strip():
        segments.append(text.strip())
    return segments


def print_file_contents(
    files: Iterable[Path],
    embedding_service: EmbeddingServiceProtocol,
    config: CLIConfig = CLI_CONFIG,
) -> None:
    """각 파일 이름, 텍스트, 임베딩 정보를 순서대로 출력합니다."""

    for file_path in files:
        print(f"===== 파일: {file_path} =====")
        timer = StageTimer()
        timer.mark("처리 시작")
        try:
            content = read_file(file_path, config=config)
        except UnicodeDecodeError as exc:  # pragma: no cover - CLI 도우미
            print(f"[디코딩 오류] {file_path}을(를) 읽을 수 없습니다: {exc}")
            timer.mark("파일 읽기 실패")
            continue
        timer.mark("파일 읽기 완료")
        print(content)
        if not content.endswith("\n"):
            print()

        segments = split_text_by_newline_or_sentence(content)
        timer.mark("텍스트 분할 완료")

        if not segments:
            print("[임베딩 생략] 비어 있는 파일입니다.")
            continue

        try:
            vectors = embedding_service.embed_texts(segments)
            segment_embeddings: list[SegmentEmbedding] = list(zip(segments, vectors))
        except ValueError as exc:
            print(f"[임베딩 오류] {exc}")
            timer.mark("임베딩 실패")
            continue
        except RuntimeError as exc:
            print(f"[임베딩 오류] {exc}")
            timer.mark("임베딩 실패")
            continue
        timer.mark("임베딩 계산 완료")

        payload = build_embedding_payload(
            file_path, segment_embeddings, embedding_service.model_name
        )
        timer.mark("페이로드 구성 완료")

        print("[임베딩 요약]")
        redacted_segments = [
            {
                "index": segment["index"],
                "text": segment["text"],
                "vector_length": segment["vector_length"],
            }
            for segment in payload["segments"]
        ]
        summary = {
            "source_path": payload["source_path"],
            "model_name": payload["model_name"],
            "segment_count": payload["segment_count"],
            "segments": redacted_segments,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        timer.mark("요약 출력 완료")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="선택한 .txt 파일을 출력하고 Qwen 임베딩을 생성합니다.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="단일 .txt 파일 경로 또는 .txt 파일을 포함한 폴더 경로",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="사용할 임베딩 모델 이름 (기본: config에 정의된 값)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="임베딩 모델을 올릴 디바이스 (예: cpu, cuda, auto)",
    )
    parser.add_argument(
        "--dtype",
        type=_torch_dtype_from_string,
        default=None,
        help="모델 및 autocast에 사용할 torch dtype (예: float16)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    updated_fields: dict[str, object] = {}
    if args.model:
        updated_fields["embedding_model_name"] = args.model
    if args.device:
        updated_fields["device"] = args.device
    if args.dtype is not None:
        updated_fields["torch_dtype"] = args.dtype
    effective_config = (
        replace(CLI_CONFIG, **updated_fields) if updated_fields else CLI_CONFIG
    )
    try:
        files = find_target_files(args.path, config=effective_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}")
        return 1

    if not files:
        allowed = ", ".join(effective_config.allowed_extensions)
        print(f"{args.path} 안에서 {allowed} 파일을 찾을 수 없습니다")
        return 1

    embedding_service = EmbeddingService(
        effective_config.embedding_model_name,
        device=effective_config.device,
        torch_dtype=effective_config.torch_dtype,
    )
    print_file_contents(
        files,
        embedding_service=embedding_service,
        config=effective_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
