"""설정한 확장자만 읽어 출력하는 CLI입니다.

하나의 파일 경로를 전달하면 해당 파일을 검증 후 출력하고,
폴더 경로를 전달하면 재귀적으로 대상 확장자 파일을 찾아 사전순으로 출력합니다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, List, Sequence

from transformer_client import TransformerClient, TransformerClientError


StageLogSink = Callable[[str], None]


def _format_local_timestamp(dt: datetime | None = None) -> str:
    """로컬 타임존 기준 짧은 형식의 타임스탬프를 반환합니다."""

    if dt is None:
        dt = datetime.now()
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _emit_stage_log(payload: dict[str, Any], sink: StageLogSink | None = None) -> None:
    """공통 포맷의 JSON 로그를 출력합니다."""

    serialized = json.dumps(payload, ensure_ascii=False)
    (sink or print)(serialized)


def log_stage(
    stage: str,
    *,
    file_id: str | None = None,
    start_wall: datetime | None = None,
    start_perf: float | None = None,
    sink: StageLogSink | None = None,
    **context: Any,
) -> dict[str, Any]:
    """단일 단계의 종료 시점을 기록합니다."""

    end_wall = datetime.now(timezone.utc)
    local_end = _format_local_timestamp(end_wall)
    payload: dict[str, Any] = {
        "timestamp": end_wall.isoformat(),
        "local_timestamp": local_end,
        "stage": stage,
        "file": file_id,
    }
    if start_wall is not None:
        payload["start_at"] = start_wall.isoformat()
        payload["end_at"] = end_wall.isoformat()
        payload["local_start_at"] = _format_local_timestamp(start_wall)
        payload["local_end_at"] = local_end
    if start_perf is not None:
        payload["elapsed_ms"] = round((perf_counter() - start_perf) * 1000, 3)
    if context:
        payload.update(context)
    _emit_stage_log(payload, sink=sink)
    return payload


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
        print(f"[{_format_local_timestamp()}] ===== 파일: {file_path} =====")
        cleaned_chunks: list[str] = []
        file_id = file_path.as_posix()
        decode_error: UnicodeDecodeError | None = None
        decoding_start_wall = datetime.now(timezone.utc)
        decoding_start_perf = perf_counter()
        print(
            f"[{_format_local_timestamp(decoding_start_wall)}] [단계 시작] 디코딩: {file_id}"
        )
        decoding_log: dict[str, Any] | None = None
        try:
            for chunk in stream_clean_lines(file_path, config=config):
                cleaned_chunks.append(chunk)
        except UnicodeDecodeError as exc:  # pragma: no cover - CLI 도우미
            decode_error = exc
            decoding_log = log_stage(
                "decoding",
                file_id=file_id,
                start_wall=decoding_start_wall,
                start_perf=decoding_start_perf,
                status="error",
                error=str(exc),
            )
        else:
            decoding_log = log_stage(
                "decoding",
                file_id=file_id,
                start_wall=decoding_start_wall,
                start_perf=decoding_start_perf,
                status="ok",
                line_count=len(cleaned_chunks),
            )
        if decoding_log is not None:
            print(
                f"[{decoding_log['local_timestamp']}] [단계 종료] 디코딩"
                f"(status={decoding_log.get('status', 'unknown')})"
            )
        if decode_error is not None:
            print(f"[디코딩 오류] {file_path}을(를) 읽을 수 없습니다: {decode_error}")
            print()
            continue

        streaming_start_wall = datetime.now(timezone.utc)
        streaming_start_perf = perf_counter()
        print(
            f"[{_format_local_timestamp(streaming_start_wall)}] [단계 시작] 스트리밍: {file_id}"
        )
        for chunk in cleaned_chunks:
            print(chunk)
        streaming_log = log_stage(
            "streaming",
            file_id=file_id,
            start_wall=streaming_start_wall,
            start_perf=streaming_start_perf,
            status="ok",
            line_count=len(cleaned_chunks),
        )
        print(
            f"[{streaming_log['local_timestamp']}] [단계 종료] 스트리밍"
            f"(status={streaming_log.get('status', 'unknown')})"
        )

        classification: str | None = None
        if (
            llm_client is not None
            and len(cleaned_chunks) >= config.min_lines_for_classification
        ):
            llm_start_wall = datetime.now(timezone.utc)
            llm_start_perf = perf_counter()
            print(
                f"[{_format_local_timestamp(llm_start_wall)}] [단계 시작] LLM 분류: {file_id}"
            )
            try:
                classification = classify_file(
                    file_path, cleaned_chunks, llm_client
                )
            except TransformerClientError as exc:  # pragma: no cover - CLI 도우미
                llm_log = log_stage(
                    "llm_classification",
                    file_id=file_id,
                    start_wall=llm_start_wall,
                    start_perf=llm_start_perf,
                    status="error",
                    error=str(exc),
                )
                print(f"[LLM 오류] {exc}")
            else:
                llm_log = log_stage(
                    "llm_classification",
                    file_id=file_id,
                    start_wall=llm_start_wall,
                    start_perf=llm_start_perf,
                    status="ok",
                )
                if classification:
                    print(f"[LLM 분류] {classification}")
            print(
                f"[{llm_log['local_timestamp']}] [단계 종료] LLM 분류"
                f"(status={llm_log.get('status', 'unknown')})"
            )

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
    files: List[Path] = []
    discovery_error: Exception | None = None
    discovery_start_wall = datetime.now(timezone.utc)
    discovery_start_perf = perf_counter()
    target_path = args.path.as_posix()
    print(
        f"[{_format_local_timestamp(discovery_start_wall)}] [단계 시작] 파일 검색: {target_path}"
    )
    discovery_log: dict[str, Any] | None = None
    try:
        files = find_target_files(args.path)
    except (FileNotFoundError, ValueError) as exc:
        discovery_error = exc
        discovery_log = log_stage(
            "discovery",
            file_id=target_path,
            start_wall=discovery_start_wall,
            start_perf=discovery_start_perf,
            target=target_path,
            status="error",
            error=str(exc),
        )
    else:
        discovery_log = log_stage(
            "discovery",
            file_id=target_path,
            start_wall=discovery_start_wall,
            start_perf=discovery_start_perf,
            target=target_path,
            status="ok",
            file_count=len(files),
        )
    if discovery_log is not None:
        print(
            f"[{discovery_log['local_timestamp']}] [단계 종료] 파일 검색"
            f"(status={discovery_log.get('status', 'unknown')})"
        )

    if discovery_error is not None:
        print(f"오류: {discovery_error}")
        return 1

    if not files:
        allowed = ", ".join(CLI_CONFIG.allowed_extensions)
        print(f"{args.path} 안에서 {allowed} 파일을 찾을 수 없습니다")
        return 1

    print_file_contents(files, llm_client=llm_client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
