from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import (
    CLIConfig,
    build_embedding_payload,
    find_target_files,
    read_file,
    split_text_by_newline_or_sentence,
    _torch_dtype_from_string,
    _validate_device_choice,
)


def create_file(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def test_find_target_files_returns_sorted_allowed_files(tmp_path: Path) -> None:
    config = CLIConfig(
        allowed_extensions=(".txt",),
        preferred_encodings=("utf-8", "cp949"),
        embedding_model_name="dummy",
        device="cpu",
        torch_dtype=None,
    )
    file_a = tmp_path / "b" / "파일1.txt"
    file_b = tmp_path / "a" / "파일2.txt"
    create_file(file_a, "첫 번째 파일")
    create_file(file_b, "두 번째 파일")

    files = find_target_files(tmp_path, config=config)

    assert files == sorted([file_a, file_b], key=lambda p: p.as_posix())


def test_find_target_files_rejects_disallowed_file(tmp_path: Path) -> None:
    config = CLIConfig(
        allowed_extensions=(".txt",),
        preferred_encodings=("utf-8",),
        embedding_model_name="dummy",
        device="cpu",
        torch_dtype=None,
    )
    file_path = tmp_path / "sample.md"
    create_file(file_path, "마크다운 파일")

    with pytest.raises(ValueError):
        find_target_files(file_path, config=config)


def test_read_file_retries_with_fallback_encodings(tmp_path: Path) -> None:
    config = CLIConfig(
        allowed_extensions=(".txt",),
        preferred_encodings=("utf-8", "cp949"),
        embedding_model_name="dummy",
        device="cpu",
        torch_dtype=None,
    )
    file_path = tmp_path / "cp949.txt"
    file_path.write_bytes("한글".encode("cp949"))

    content = read_file(file_path, config=config)

    assert content == "한글"


def test_build_embedding_payload_contains_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    create_file(file_path, "간단한 내용")

    payload = build_embedding_payload(
        file_path,
        [("첫 줄", [0.5, -0.5]), ("둘째 줄", [1.0, 2.0])],
        "dummy/model",
    )

    assert payload == {
        "source_path": str(file_path),
        "model_name": "dummy/model",
        "segment_count": 2,
        "segments": [
            {
                "index": 0,
                "text": "첫 줄",
                "vector": [0.5, -0.5],
                "vector_length": 2,
            },
            {
                "index": 1,
                "text": "둘째 줄",
                "vector": [1.0, 2.0],
                "vector_length": 2,
            },
        ],
    }


def test_split_text_by_newline_or_sentence() -> None:
    text = "첫 문장입니다. 두 번째 문장입니다!\n\n세 번째 줄"

    segments = split_text_by_newline_or_sentence(
        text, min_chunk_size=10, max_chunk_size=80, overlap_size=5
    )

    assert segments == ["첫 문장입니다. 두 번째 문장입니다! 세 번째 줄"]


def test_split_text_by_newline_or_sentence_builds_overlapping_chunks() -> None:
    sentence = "이 문장은 청킹 테스트를 위한 문장입니다."
    text = " ".join([sentence for _ in range(20)])

    segments = split_text_by_newline_or_sentence(
        text, min_chunk_size=50, max_chunk_size=80, overlap_size=10
    )

    assert len(segments) > 1
    assert all(0 < len(chunk) <= 80 for chunk in segments)
    for previous, current in zip(segments, segments[1:]):
        assert previous[-10:] in current


def test_torch_dtype_from_string_allows_prefixed_value() -> None:
    torch = pytest.importorskip("torch")

    dtype = _torch_dtype_from_string("torch.float16")

    assert dtype == torch.float16


def test_torch_dtype_from_string_rejects_invalid_value() -> None:
    pytest.importorskip("torch")

    with pytest.raises(argparse.ArgumentTypeError):
        _torch_dtype_from_string("not_a_dtype")


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available)


def test_validate_device_choice_accepts_cpu_even_without_cuda() -> None:
    torch_stub = _FakeTorch(available=False)

    assert _validate_device_choice("cpu", torch_stub) == "cpu"


def test_validate_device_choice_rejects_missing_cuda() -> None:
    torch_stub = _FakeTorch(available=False)

    with pytest.raises(RuntimeError):
        _validate_device_choice("cuda", torch_stub)
