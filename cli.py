"""설정한 확장자만 읽어 출력하는 CLI입니다.

하나의 파일 경로를 전달하면 해당 파일을 검증 후 출력하고,
폴더 경로를 전달하면 재귀적으로 대상 확장자 파일을 찾아 사전순으로 출력합니다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


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


CLI_CONFIG = CLIConfig(
    allowed_extensions=_normalize_extensions([".txt"]),
    preferred_encodings=("utf-8", "cp949"),
)


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


def print_file_contents(files: Iterable[Path], config: CLIConfig = CLI_CONFIG) -> None:
    """각 파일 이름과 텍스트를 순서대로 출력합니다."""

    for file_path in files:
        print(f"===== 파일: {file_path} =====")
        try:
            content = read_file(file_path, config=config)
        except UnicodeDecodeError as exc:  # pragma: no cover - CLI 도우미
            print(f"[디코딩 오류] {file_path}을(를) 읽을 수 없습니다: {exc}")
            continue
        print(content)
        if not content.endswith("\n"):
            print()  # 빈 줄을 추가하여 파일 간 구분을 명확히 함


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
    try:
        files = find_target_files(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}")
        return 1

    if not files:
        allowed = ", ".join(CLI_CONFIG.allowed_extensions)
        print(f"{args.path} 안에서 {allowed} 파일을 찾을 수 없습니다")
        return 1

    print_file_contents(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
