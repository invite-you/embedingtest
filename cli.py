"""설정한 확장자만 읽어 출력하는 CLI입니다.

하나의 파일 경로를 전달하면 해당 파일을 검증 후 출력하고,
폴더 경로를 전달하면 재귀적으로 대상 확장자 파일을 찾아 사전순으로 출력합니다.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from transformer_client import TransformerClient, TransformerClientError


DEFAULT_PROMPT_CHAR_LIMIT = 1200
PROMPT_INPUT_RATIO = 0.7
SECTION_SPECS: tuple[tuple[str, float, float], ...] = (
    ("파일_시작_구간", 0.0, 0.34),
    ("파일_중앙_구간", 0.33, 0.67),
    ("파일_끝_구간", 0.66, 1.0),
)


def _log_with_ts(label: str, *, elapsed: float | None = None) -> None:
    """HH:MM:SS 타임스탬프와 선택적 경과 시간을 포함해 로그를 출력합니다."""

    timestamp = datetime.now().strftime("%H:%M:%S")
    duration = f" (elapsed: {elapsed:.3f}s)" if elapsed is not None else ""
    print(f"[{timestamp}] {label}{duration}")


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


def _truncate_excerpt(lines: Sequence[str], max_chars: int = DEFAULT_PROMPT_CHAR_LIMIT) -> str:
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


def _slice_lines_by_ratio(lines: Sequence[str], start_ratio: float, end_ratio: float) -> Sequence[str]:
    """주어진 비율 구간에 해당하는 라인 조각을 반환합니다."""

    if not lines:
        return []
    total = len(lines)
    start_index = min(int(start_ratio * total), total - 1)
    end_index = max(start_index + 1, min(total, int(end_ratio * total)))
    return lines[start_index:end_index]


def _build_excerpt_sections(lines: Sequence[str], max_chars: int) -> list[tuple[str, str]]:
    """처음/중간/끝 구간 또는 전체 텍스트 발췌를 생성합니다."""

    if not lines:
        return []

    full_text = "\n".join(lines)
    if max_chars >= len(full_text):
        return [("파일_전체_구간", full_text)] if full_text else []

    segments: list[tuple[str, Sequence[str]]] = []
    for label, start_ratio, end_ratio in SECTION_SPECS:
        segment_lines = _slice_lines_by_ratio(lines, start_ratio, end_ratio)
        if segment_lines:
            segments.append((label, segment_lines))

    if not segments:
        fallback = _truncate_excerpt(lines, max_chars=max_chars)
        return [("대표_구간", fallback)] if fallback else []

    prompt_budget = max(1, max_chars)
    per_section = prompt_budget // len(segments)
    remainder = prompt_budget % len(segments)

    sections: list[tuple[str, str]] = []
    for index, (label, segment_lines) in enumerate(segments):
        budget = per_section + (1 if index < remainder else 0)
        budget = max(1, budget)
        excerpt = _truncate_excerpt(segment_lines, max_chars=budget)
        if excerpt:
            sections.append((label, excerpt))

    if not sections:
        fallback = _truncate_excerpt(lines, max_chars=max_chars)
        if fallback:
            sections.append(("대표_구간", fallback))
    return sections


def _calculate_prompt_budget(max_input_tokens: int | None) -> int:
    """LLM 입력 한도를 받아 70% 이내 문자 수 한도를 계산합니다."""

    if max_input_tokens is None or max_input_tokens <= 0:
        return DEFAULT_PROMPT_CHAR_LIMIT
    return max(1, int(max_input_tokens * PROMPT_INPUT_RATIO))


def build_classification_prompt(
    path: Path,
    lines: Sequence[str],
    *,
    max_prompt_chars: int | None = None,
) -> str:
    """Qwen/Qwen3-4B-Instruct-2507에게 분류 프롬프트를 구성합니다."""

    prompt_budget = max_prompt_chars or DEFAULT_PROMPT_CHAR_LIMIT
    sections = _build_excerpt_sections(lines, prompt_budget)
    if not sections:
        excerpt = _truncate_excerpt(lines, max_chars=prompt_budget)
        sections = [("대표_구간", excerpt)] if excerpt else []

    section_text = "\n\n".join(
        f"[[구간:{label}]]\n{content}\n[[/구간:{label}]]"
        for label, content in sections
    )
    return (
        "주어진 텍스트를 읽고 어떤 파일인지 설명하세요.\n"
        "각 [[구간:...]] 표시는 파일 내 특정 위치에서 가져온 발췌이며, 표식 안쪽 텍스트만 해당 구간의 맥락입니다.\n"
        "항상 완전하고 자연스러운 문장으로 답하고 중간에 끊기지 않도록 하세요. 응답 전체가 하나의 단락처럼 읽히도록 문장부호와 종결어미를 명확히 남기세요.\n"
        "출력은 '타임라인' 섹션 아래에서 단계별로 나열하고, 각 단계마다 `- [HH:MM:SS] 설명` 형태의 촘촘한 타임스탬프를 붙여 몇 초 단위 흐름을 최대한 세밀하게 보여주세요.\n"
        "필요 시 한두 문장 이내에서 추가 맥락을 보충하되, 설명이 매끄럽게 이어지도록 작성하세요.\n"
        f"파일 경로: {path}\n\n"
        "[발췌]\n"
        f"{section_text}\n"
    )


def classify_file(
    path: Path,
    lines: Sequence[str],
    llm_client: TransformerClient,
    *,
    debug_llm_prompt: bool = False,
) -> str:
    """LLM을 호출해 분류 결과를 반환합니다."""

    max_input_tokens = getattr(getattr(llm_client, "config", None), "max_input_tokens", None)
    prompt_budget = _calculate_prompt_budget(max_input_tokens)
    prompt = build_classification_prompt(path, lines, max_prompt_chars=prompt_budget)
    if debug_llm_prompt:
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"[LLM 프롬프트 {timestamp}]\n{prompt}\n")
    return llm_client.generate(prompt)


def print_file_contents(
    files: Iterable[Path],
    config: CLIConfig = CLI_CONFIG,
    llm_client: TransformerClient | None = None,
    *,
    debug_llm_prompt: bool = False,
) -> dict[Path, FileStreamResult]:
    """각 파일 이름과 정제된 텍스트를 순서대로 출력하고 반환합니다.

    주요 단계는 HH:MM:SS 타임스탬프와 경과 시간을 포함한 로그로 안내됩니다.
    """

    streamed_outputs: dict[Path, FileStreamResult] = {}
    for file_path in files:
        overall_start = time.perf_counter()
        _log_with_ts(f"파일 처리 시작: {file_path}")
        print(f"===== 파일: {file_path} =====")

        cleaned_chunks: list[str] = []
        stream_start = time.perf_counter()
        _log_with_ts(f"텍스트 정제 단계 시작: {file_path}")
        try:
            for chunk in stream_clean_lines(file_path, config=config):
                print(chunk)
                cleaned_chunks.append(chunk)
        except UnicodeDecodeError as exc:  # pragma: no cover - CLI 도우미
            elapsed = time.perf_counter() - stream_start
            _log_with_ts(f"텍스트 정제 단계 실패: {file_path}", elapsed=elapsed)
            print(f"[디코딩 오류] {file_path}을(를) 읽을 수 없습니다: {exc}")
            _log_with_ts(
                f"파일 처리 종료(텍스트 정제 실패): {file_path}",
                elapsed=time.perf_counter() - overall_start,
            )
            continue
        else:
            elapsed = time.perf_counter() - stream_start
            _log_with_ts(
                f"텍스트 정제 단계 완료: {file_path} (라인 {len(cleaned_chunks)}개)",
                elapsed=elapsed,
            )

        classification: str | None = None
        if not cleaned_chunks:
            _log_with_ts(f"LLM 분류 건너뜀: 정제된 텍스트 없음 ({file_path})")
        elif llm_client is None:
            _log_with_ts(f"LLM 분류 건너뜀: 클라이언트 비활성화 ({file_path})")
        elif len(cleaned_chunks) < config.min_lines_for_classification:
            _log_with_ts(
                "LLM 분류 건너뜀: 라인 수 부족 "
                f"({len(cleaned_chunks)}/{config.min_lines_for_classification}) ({file_path})"
            )
        else:
            classify_start = time.perf_counter()
            _log_with_ts(f"LLM 분류 단계 시작: {file_path}")
            try:
                classification = classify_file(
                    file_path,
                    cleaned_chunks,
                    llm_client,
                    debug_llm_prompt=debug_llm_prompt,
                )
            except TransformerClientError as exc:  # pragma: no cover - CLI 도우미
                elapsed = time.perf_counter() - classify_start
                _log_with_ts(f"LLM 분류 단계 실패: {file_path}", elapsed=elapsed)
                print(f"[LLM 오류] {exc}")
            else:
                elapsed = time.perf_counter() - classify_start
                status = (
                    f"LLM 분류 단계 완료: {file_path}"
                    if classification
                    else f"LLM 분류 단계 완료: 결과 없음 ({file_path})"
                )
                _log_with_ts(status, elapsed=elapsed)
                if classification:
                    print(f"[LLM 분류] {classification}")

        streamed_outputs[file_path] = FileStreamResult(
            cleaned_chunks=cleaned_chunks,
            classification=classification,
        )
        _log_with_ts(
            f"파일 처리 완료: {file_path}", elapsed=time.perf_counter() - overall_start
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
    parser.add_argument(
        "--debug-llm-prompt",
        action="store_true",
        help="LLM 호출 전에 생성된 프롬프트 내용을 출력합니다.",
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

    print_file_contents(
        files,
        llm_client=llm_client,
        debug_llm_prompt=args.debug_llm_prompt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
