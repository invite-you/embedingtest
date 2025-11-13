"""텍스트 파일을 읽고 문장 단위로 분리하는 유틸리티."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

_DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8", "cp949")


class TextDecodingError(UnicodeDecodeError):
    """텍스트 디코딩 실패 시 보다 구체적인 예외."""


def read_text_with_fallback(
    path: Path, encodings: Sequence[str] | None = None
) -> str:
    """여러 인코딩 후보를 시도해 텍스트를 반환합니다."""

    candidates = tuple(encodings) if encodings else _DEFAULT_ENCODINGS
    last_error: UnicodeDecodeError | None = None
    for encoding in candidates:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise UnicodeDecodeError("", b"", 0, 0, "사용 가능한 인코딩이 없습니다")


_SENTENCE_DELIMITER = re.compile(r"(?<=[.!?。？！])\s+")


def split_text_by_newline_or_sentence(text: str) -> list[str]:
    """줄바꿈 우선으로 분할하되 문장 부호 기준으로도 세분화합니다."""

    segments: list[str] = []
    for line in text.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        parts = _SENTENCE_DELIMITER.split(cleaned_line)
        for part in parts:
            cleaned_part = part.strip()
            if cleaned_part:
                segments.append(cleaned_part)

    if not segments and text.strip():
        segments.append(text.strip())
    return segments


__all__ = [
    "TextDecodingError",
    "read_text_with_fallback",
    "split_text_by_newline_or_sentence",
]
