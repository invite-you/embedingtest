from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding import EmbeddingConfig
from cli import CLIConfig, find_target_files, print_file_contents, read_file


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


def test_print_file_contents_embeds_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = CLIConfig(
        allowed_extensions=(".txt",),
        preferred_encodings=("utf-8",),
        embedding=EmbeddingConfig(preview_values=2),
    )
    file_path = tmp_path / "sample.txt"
    create_file(file_path, "테스트 본문")

    class DummyEmbedder:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def embed_text(self, text: str) -> list[float]:
            self.calls.append(text)
            return [0.1, 0.2, 0.3]

    dummy = DummyEmbedder()

    print_file_contents([file_path], embedder=dummy, config=config)

    out = capsys.readouterr().out
    assert dummy.calls == ["테스트 본문"]
    assert "임베딩 길이: 3" in out
    assert "임베딩 앞 2개 값" in out
