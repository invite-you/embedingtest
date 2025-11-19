from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import (
    CLIConfig,
    FileStreamResult,
    PROMPT_INPUT_RATIO,
    _calculate_prompt_budget,
    build_classification_prompt,
    classify_file,
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

    outputs = print_file_contents([file_path], config=config, llm_client=None)
    captured = capsys.readouterr().out

    assert file_path in outputs
    result = outputs[file_path]
    assert isinstance(result, FileStreamResult)
    assert result.cleaned_chunks == ["라인1", "라인2"]
    assert result.classification is None
    assert f"===== 파일: {file_path} =====" in captured
    assert "라인1\n라인2\n" in captured


class DummyLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_classify_file_logs_prompt_when_debug_enabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file_path = tmp_path / "sample.txt"
    lines = ["첫 줄", "둘째 줄", "셋째 줄"]
    llm = DummyLLM("타임라인을 포함한 응답")

    classify_file(file_path, lines, llm, debug_llm_prompt=True)
    captured = capsys.readouterr().out

    assert llm.prompts and llm.prompts[0] in captured
    assert "[LLM 프롬프트" in captured


def test_classify_file_keeps_stdout_clean_when_debug_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file_path = tmp_path / "sample.txt"
    lines = ["첫 줄", "둘째 줄", "셋째 줄"]
    llm = DummyLLM("타임라인을 포함한 응답")

    classify_file(file_path, lines, llm)
    captured = capsys.readouterr().out

    assert captured == ""


def test_print_file_contents_invokes_llm_when_threshold_met(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = CLIConfig(
        allowed_extensions=(".txt",),
        preferred_encodings=("utf-8",),
        min_lines_for_classification=2,
    )
    file_path = tmp_path / "sample.txt"
    create_file(file_path, "첫 줄\n둘째 줄\n셋째 줄")
    llm = DummyLLM("보고서로 보이며 주요 요약이 포함됩니다.")

    outputs = print_file_contents([file_path], config=config, llm_client=llm)
    captured = capsys.readouterr().out

    result = outputs[file_path]
    assert result.classification == llm.response
    assert llm.prompts and llm.prompts[0].startswith(
        "주어진 텍스트를 읽고 어떤 파일인지 설명하세요."
    )
    assert "[[구간:" in llm.prompts[0]
    assert "중간에 끊기지 않도록" in llm.prompts[0]
    assert "타임라인" in llm.prompts[0]
    assert "HH:MM:SS" in llm.prompts[0]
    assert "첫 줄" in llm.prompts[0]
    assert f"[LLM 분류] {llm.response}" in captured


def test_build_classification_prompt_contains_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "a.txt"
    lines = [f"라인 {i}" for i in range(1, 10)]

    prompt = build_classification_prompt(file_path, lines, max_prompt_chars=30)

    assert str(file_path) in prompt
    assert "라인 1" in prompt
    assert "[[구간:파일_시작_구간]]" in prompt
    assert "[[구간:파일_중앙_구간]]" in prompt
    assert "[[구간:파일_끝_구간]]" in prompt


def test_build_classification_prompt_uses_full_text_when_within_budget(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "b.txt"
    lines = ["짧은", "본문"]

    prompt = build_classification_prompt(file_path, lines, max_prompt_chars=10_000)

    assert "[[구간:파일_전체_구간]]" in prompt
    assert "짧은" in prompt and "본문" in prompt


def test_calculate_prompt_budget_respects_ratio() -> None:
    max_tokens = 262_144
    budget = _calculate_prompt_budget(max_tokens)

    assert budget == int(max_tokens * PROMPT_INPUT_RATIO)
