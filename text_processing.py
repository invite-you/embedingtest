"""텍스트 파일을 읽고 문장 단위로 분리하는 유틸리티."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

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


def _iter_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            continue
        buffer.append(stripped)

    if buffer:
        paragraphs.append(" ".join(buffer).strip())

    if not paragraphs and text.strip():
        paragraphs.append(text.strip())
    return paragraphs


def _split_long_paragraph(paragraph: str, max_chunk_size: int) -> list[str]:
    if len(paragraph) <= max_chunk_size:
        return [paragraph]

    sentences = [part.strip() for part in _SENTENCE_DELIMITER.split(paragraph) if part.strip()]
    if not sentences:
        return [paragraph[i : i + max_chunk_size] for i in range(0, len(paragraph), max_chunk_size)]

    blocks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        projected_len = len(current) + 1 + len(sentence)
        if projected_len <= max_chunk_size:
            current = f"{current} {sentence}"
        else:
            blocks.append(current.strip())
            current = sentence

    if current:
        blocks.append(current.strip())
    return blocks


def _prepare_blocks(text: str, max_chunk_size: int) -> list[str]:
    paragraphs = _iter_paragraphs(text)
    blocks: list[str] = []
    for paragraph in paragraphs:
        blocks.extend(_split_long_paragraph(paragraph, max_chunk_size))
    return blocks


def _assemble_chunks(
    blocks: Iterable[str],
    *,
    min_chunk_size: int,
    max_chunk_size: int,
    overlap_size: int,
) -> list[str]:
    chunks: list[str] = []
    buffer = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if not buffer:
            buffer = block
            continue

        projected_len = len(buffer) + 1 + len(block)
        if projected_len <= max_chunk_size:
            buffer = f"{buffer} {block}"
            continue

        chunk = buffer.strip()
        if len(chunk) < min_chunk_size:
            if chunk:
                chunks.append(chunk)
            buffer = block
            continue

        if chunk:
            chunks.append(chunk)
        overlap = chunk[-overlap_size:] if overlap_size > 0 else ""
        candidate = (f"{overlap} {block}" if overlap else block).strip()
        if len(candidate) > max_chunk_size:
            allowed_overlap = max(0, max_chunk_size - len(block) - 1)
            overlap = chunk[-allowed_overlap:] if allowed_overlap > 0 else ""
            candidate = (f"{overlap} {block}" if overlap else block).strip()
        buffer = candidate

    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def split_text_by_newline_or_sentence(
    text: str,
    *,
    min_chunk_size: int = 1000,
    max_chunk_size: int = 1500,
    overlap_size: int = 250,
) -> list[str]:
    """빈 줄→문단→문장 순으로 나눈 뒤 1,000~1,500자 청크를 겹치게 만듭니다."""

    if max_chunk_size < min_chunk_size:
        raise ValueError("max_chunk_size는 min_chunk_size보다 작을 수 없습니다.")

    overlap_size = max(0, min(overlap_size, min_chunk_size))
    blocks = _prepare_blocks(text, max_chunk_size)
    if not blocks:
        return []
    return _assemble_chunks(
        blocks,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        overlap_size=overlap_size,
    )


__all__ = [
    "TextDecodingError",
    "read_text_with_fallback",
    "split_text_by_newline_or_sentence",
]
