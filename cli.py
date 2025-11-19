"""설정한 확장자만 읽어 출력하는 CLI입니다.

하나의 파일 경로를 전달하면 해당 파일을 검증 후 출력하고,
폴더 경로를 전달하면 재귀적으로 대상 확장자 파일을 찾아 사전순으로 출력합니다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from transformer_client import TransformerClient, TransformerClientError


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


@dataclass(frozen=True)
class CLIConfig:
    """CLI 동작에 필요한 설정 모음입니다."""

    allowed_extensions: tuple[str, ...]
    preferred_encodings: tuple[str, ...]
    min_lines_for_classification: int = 3


CLI_CONFIG = CLIConfig(
    allowed_extensions=_normalize_extensions([".txt"]),
    preferred_encodings=("utf-8", "cp949"),
    min_lines_for_classification=3,
)


@dataclass
class FileStreamResult:
    """LLM 응답을 포함한 파일 처리 결과."""

    cleaned_chunks: list[str]
    classification: str | None


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


def stream_clean_lines(
    path: Path, config: CLIConfig = CLI_CONFIG
) -> Iterator[str]:
    """파일을 스트리밍으로 읽으며 정제된 텍스트 라인을 제공합니다."""

    last_error: UnicodeDecodeError | None = None
    for encoding in config.preferred_encodings:
        try:
            with path.open("r", encoding=encoding) as file_obj:
                previous_line: str | None = None
                for raw_line in file_obj:
                    normalized = raw_line.strip()
                    if not normalized:
                        continue
                    if normalized == previous_line:
                        continue
                    previous_line = normalized
                    yield normalized
            return
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise UnicodeDecodeError("", b"", 0, 0, "사용 가능한 인코딩이 없습니다")


def _truncate_excerpt(lines: Sequence[str], max_chars: int = 1200) -> str:
    """LLM 프롬프트에 담을 텍스트를 지정 길이에 맞춰 축약합니다."""

    excerpt: list[str] = []
    total = 0
    for line in lines:
        if not line:
            continue
        if total + len(line) > max_chars:
            break
        excerpt.append(line)
        total += len(line)
    if not excerpt and lines:
        excerpt.append(lines[0][:max_chars])
    return "\n".join(excerpt)


def build_classification_prompt(path: Path, lines: Sequence[str]) -> str:
    """Qwen/Qwen3-4B-Instruct-2507에게 분류 프롬프트를 구성합니다."""

    excerpt = _truncate_excerpt(lines)
    return (
        "주어진 텍스트를 읽고 어떤 파일인지 설명하세요.\n"
        "주어진 텍스트는 파일의 첫 부분의 일부입니다.\n"
        "응답은 이 파일의 내용을 이해할 수 있는 설명이면 충분하며 형식은 고정되어 있지 않습니다.\n"
        "항상 완전한 문장으로 답변하고 자연스러운 종결어미와 문장부호로 마무리하세요.\n"
        "필요 시 추가 맥락을 한두 문장 이내로 덧붙이세요.\n"
        f"파일 경로: {path}\n\n"
        "[내용]\n"
        f"{excerpt}\n"
    )


def classify_file(
    path: Path,
    lines: Sequence[str],
    llm_client: TransformerClient,
) -> str:
    """LLM을 호출해 분류 결과를 반환합니다."""

    prompt = build_classification_prompt(path, lines)
    return llm_client.generate(prompt)


def print_file_contents(
    files: Iterable[Path],
    config: CLIConfig = CLI_CONFIG,
    llm_client: TransformerClient | None = None,
) -> dict[Path, FileStreamResult]:
    """각 파일 이름과 정제된 텍스트를 순서대로 출력하고 반환합니다."""

    streamed_outputs: dict[Path, FileStreamResult] = {}
    for file_path in files:
        print(f"===== 파일: {file_path} =====")
        cleaned_chunks: list[str] = []
        try:
            for chunk in stream_clean_lines(file_path, config=config):
                print(chunk)
                cleaned_chunks.append(chunk)
        except UnicodeDecodeError as exc:  # pragma: no cover - CLI 도우미
            print(f"[디코딩 오류] {file_path}을(를) 읽을 수 없습니다: {exc}")
            continue

        classification: str | None = None
        if (
            llm_client is not None
            and len(cleaned_chunks) >= config.min_lines_for_classification
        ):
            try:
                classification = classify_file(file_path, cleaned_chunks, llm_client)
            except TransformerClientError as exc:  # pragma: no cover - CLI 도우미
                print(f"[LLM 오류] {exc}")
            else:
                if classification:
                    print(f"[LLM 분류] {classification}")

        streamed_outputs[file_path] = FileStreamResult(
            cleaned_chunks=cleaned_chunks,
            classification=classification,
        )
        print()  # 파일 간 구분을 명확히 함

    return streamed_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="하나의 .txt 파일 또는 폴더 내 모든 .txt 파일을 출력합니다.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="단일 .txt 파일 경로 또는 .txt 파일을 포함한 폴더 경로",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm_client: TransformerClient | None = None
    try:
        llm_client = TransformerClient()
    except TransformerClientError as exc:  # pragma: no cover - 런타임 의존성
        print(
            "[경고] Qwen/Qwen3-4B-Instruct-2507를 초기화할 수 없어 분류를 건너뜁니다: "
            f"{exc}"
        )
    try:
        files = find_target_files(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}")
        return 1

    if not files:
        allowed = ", ".join(CLI_CONFIG.allowed_extensions)
        print(f"{args.path} 안에서 {allowed} 파일을 찾을 수 없습니다")
        return 1

    print_file_contents(files, llm_client=llm_client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
