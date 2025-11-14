from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import (
    CLIConfig,
    find_target_files,
    print_file_contents,
    read_file,
    stream_clean_lines,
)


def create_file(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def test_find_target_files_returns_sorted_allowed_files(tmp_path: Path) -> None:
    config = CLIConfig(allowed_extensions=(".txt",), preferred_encodings=("utf-8", "cp949"))
    file_a = tmp_path / "b" / "파일1.txt"
    file_b = tmp_path / "a" / "파일2.txt"
    create_file(file_a, "첫 번째 파일")
    create_file(file_b, "두 번째 파일")

    files = find_target_files(tmp_path, config=config)

    assert files == sorted([file_a, file_b], key=lambda p: p.as_posix())


def test_find_target_files_rejects_disallowed_file(tmp_path: Path) -> None:
    config = CLIConfig(allowed_extensions=(".txt",), preferred_encodings=("utf-8",))
    file_path = tmp_path / "sample.md"
    create_file(file_path, "마크다운 파일")

    with pytest.raises(ValueError):
        find_target_files(file_path, config=config)


def test_read_file_retries_with_fallback_encodings(tmp_path: Path) -> None:
    config = CLIConfig(allowed_extensions=(".txt",), preferred_encodings=("utf-8", "cp949"))
    file_path = tmp_path / "cp949.txt"
    file_path.write_bytes("한글".encode("cp949"))

    content = read_file(file_path, config=config)

    assert content == "한글"


def test_stream_clean_lines_normalizes_and_deduplicates(tmp_path: Path) -> None:
    config = CLIConfig(allowed_extensions=(".txt",), preferred_encodings=("utf-8",))
    file_path = tmp_path / "sample.txt"
    create_file(
        file_path,
        """
        첫 줄
        첫 줄

        둘째 줄
        셋째 줄
        셋째 줄
        """.strip(),
    )

    lines = list(stream_clean_lines(file_path, config=config))

    assert lines == ["첫 줄", "둘째 줄", "셋째 줄"]


def test_print_file_contents_streams_and_returns_chunks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = CLIConfig(allowed_extensions=(".txt",), preferred_encodings=("utf-8",))
    file_path = tmp_path / "sample.txt"
    create_file(file_path, "라인1\n\n라인1\n라인2")

    outputs = print_file_contents([file_path], config=config)
    captured = capsys.readouterr().out

    assert file_path in outputs
    assert outputs[file_path] == ["라인1", "라인2"]
    assert f"===== 파일: {file_path} =====" in captured
    assert "라인1\n라인2\n" in captured
