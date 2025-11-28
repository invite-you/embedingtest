"""폴더 내 파일을 'mdbr-leaf-ir' 임베딩과 HDBSCAN으로 클러스터링하는 CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from hdbscan import HDBSCAN

from cli import (
    CLI_CONFIG,
    EmbeddingService,
    _normalize_extensions,
    _torch_dtype_from_string,
    find_target_files,
    read_file,
)


@dataclass
class FileRecord:
    """클러스터링 대상 파일의 정보를 담습니다."""

    path: Path
    content: str
    embedding: List[float] | None = None
    cluster_label: int = -1
    note: str | None = None


def _build_cli_config(extensions: Sequence[str], device: str, torch_dtype: str | None) -> CLI_CONFIG.__class__:
    normalized_ext = _normalize_extensions(extensions)
    dtype_value = None
    if torch_dtype is not None:
        dtype_value = _torch_dtype_from_string(torch_dtype)
    return replace(
        CLI_CONFIG,
        allowed_extensions=normalized_ext,
        device=device,
        torch_dtype=dtype_value,
    )


def _gather_files(target: Path, config) -> list[FileRecord]:
    files = find_target_files(target, config)
    records: list[FileRecord] = []
    for path in files:
        try:
            text = read_file(path, config=config)
        except UnicodeDecodeError as exc:
            records.append(
                FileRecord(path=path, content="", embedding=None, cluster_label=-1, note=f"디코딩 실패: {exc}")
            )
            continue
        if not text.strip():
            records.append(FileRecord(path=path, content="", embedding=None, cluster_label=-1, note="빈 파일"))
            continue
        records.append(FileRecord(path=path, content=text))
    return records


def _embed_records(records: list[FileRecord], model_name: str, device: str, torch_dtype: str | None) -> list[FileRecord]:
    embeddable = [record for record in records if record.content.strip()]
    if not embeddable:
        return records

    dtype_value = None
    if torch_dtype is not None:
        dtype_value = _torch_dtype_from_string(torch_dtype)

    service = EmbeddingService(
        model_name,
        device=device,
        torch_dtype=dtype_value,
    )
    embeddings = service.embed_texts([record.content for record in embeddable])
    for record, vector in zip(embeddable, embeddings):
        record.embedding = vector
    return records


def _cluster_embeddings(records: list[FileRecord], min_cluster_size: int, min_samples: int | None) -> list[FileRecord]:
    embeddable = [record for record in records if record.embedding is not None]
    if not embeddable:
        return records

    vectors = np.array([record.embedding for record in embeddable], dtype=float)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(vectors)
    for record, label in zip(embeddable, labels):
        record.cluster_label = int(label)
    return records


def _format_label(label: int) -> str:
    return "noise" if label == -1 else f"cluster-{label}"


def _print_clusters(records: Iterable[FileRecord]) -> None:
    clusters: Dict[int, list[FileRecord]] = {}
    for record in records:
        clusters.setdefault(record.cluster_label, []).append(record)

    def _cluster_sort_key(item: tuple[int, list[FileRecord]]) -> tuple[int, int]:
        label, members = item
        # 노이즈 클러스터는 항상 마지막에 출력합니다.
        if label == -1:
            return (1_000_000, len(members))
        return (label, len(members))

    for cluster_label, members in sorted(clusters.items(), key=_cluster_sort_key):
        print(f"=== {_format_label(cluster_label)} ({len(members)} files) ===")
        for record in members:
            suffix = f" - {record.note}" if record.note else ""
            print(f"- {record.path}{suffix}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="지정된 폴더에서 파일을 읽어 mdbr-leaf-ir 임베딩으로 HDBSCAN 클러스터링을 수행합니다.",
    )
    parser.add_argument("target", type=Path, help="클러스터링할 폴더 경로")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(CLI_CONFIG.allowed_extensions),
        help="검색할 파일 확장자 목록 (예: .txt .md)",
    )
    parser.add_argument(
        "--model",
        default="mdbr-leaf-ir",
        help="사용할 임베딩 모델 이름 (기본값: mdbr-leaf-ir)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="HDBSCAN의 min_cluster_size 값",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN의 min_samples 값 (미지정 시 기본 동작)",
    )
    parser.add_argument(
        "--device",
        default=CLI_CONFIG.device,
        help="임베딩 모델을 올릴 디바이스 (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="모델 로딩 및 autocast에 사용할 torch dtype 문자열",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_config = _build_cli_config(args.extensions, args.device, args.dtype)
    records = _gather_files(args.target, cli_config)
    records = _embed_records(records, args.model, args.device, args.dtype)
    records = _cluster_embeddings(records, args.min_cluster_size, args.min_samples)
    _print_clusters(records)


if __name__ == "__main__":
    main()
