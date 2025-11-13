"""문장 임베딩을 바탕으로 군집을 찾아 대표 문장을 추리는 도우미."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from pathlib import Path
from typing import Sequence

from agent_config import AgentConfig, load_agent_config
from cli import (
    CLI_CONFIG,
    EmbeddingService,
    EmbeddingServiceProtocol,
    find_target_files,
    read_file,
    split_text_by_newline_or_sentence,
)


def _euclidean_distance_sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    length = len(vectors[0])
    return [
        sum(vector[i] for vector in vectors) / len(vectors)
        for i in range(length)
    ]


@dataclass
class SentenceCluster:
    cluster_id: int
    representative_index: int
    representative_sentence: str
    member_indices: list[int]
    context_sentences: list[str]


class SentenceClusterer:
    """K-means 유사 알고리즘으로 문장 벡터를 그룹화합니다."""

    def __init__(self, config: AgentConfig) -> None:
        self._cluster_cfg = config.clustering
        self._context_cfg = config.context

    def build_clusters(
        self, sentences: Sequence[str], embeddings: Sequence[Sequence[float]]
    ) -> list[SentenceCluster]:
        if not sentences:
            return []
        if len(sentences) != len(embeddings):
            raise ValueError("문장 수와 임베딩 수가 일치하지 않습니다.")

        if len(sentences) < self._cluster_cfg.min_sentences:
            return self._build_sampled_blocks(sentences)

        return self._kmeans_clusters(sentences, embeddings)

    def _build_sampled_blocks(self, sentences: Sequence[str]) -> list[SentenceCluster]:
        max_clusters = max(1, self._cluster_cfg.max_clusters)
        target_count = min(len(sentences), max_clusters)
        if target_count == 0:
            return []
        indices: list[int] = []
        if target_count == len(sentences):
            indices = list(range(len(sentences)))
        else:
            step = (len(sentences) - 1) / max(target_count - 1, 1)
            indices = sorted({round(i * step) for i in range(target_count)})
            while len(indices) < target_count:
                indices.append(len(indices))
            indices = sorted(set(indices))[:target_count]

        clusters: list[SentenceCluster] = []
        for cluster_id, idx in enumerate(indices):
            clusters.append(
                SentenceCluster(
                    cluster_id=cluster_id,
                    representative_index=idx,
                    representative_sentence=sentences[idx],
                    member_indices=[idx],
                    context_sentences=self._gather_context(idx, sentences),
                )
            )
        return clusters

    def _kmeans_clusters(
        self, sentences: Sequence[str], embeddings: Sequence[Sequence[float]]
    ) -> list[SentenceCluster]:
        vector_list = [list(vector) for vector in embeddings]
        vector_count = len(vector_list)
        if vector_count == 0:
            return []
        k = min(
            self._cluster_cfg.max_clusters,
            max(1, math.ceil(math.sqrt(vector_count))),
        )
        k = min(k, vector_count)
        centroids = [vector_list[i][:] for i in range(k)]
        assignments = [0] * vector_count
        for _ in range(50):
            changed = False
            for idx, vector in enumerate(vector_list):
                distances = [
                    _euclidean_distance_sq(vector, centroid)
                    for centroid in centroids
                ]
                best_cluster = min(range(k), key=lambda c: distances[c])
                if assignments[idx] != best_cluster:
                    assignments[idx] = best_cluster
                    changed = True
            if not changed:
                break
            for cluster_id in range(k):
                members = [
                    vector_list[i]
                    for i in range(vector_count)
                    if assignments[i] == cluster_id
                ]
                if members:
                    centroids[cluster_id] = _mean_vector(members)
                else:
                    fallback_index = cluster_id % vector_count
                    centroids[cluster_id] = vector_list[fallback_index][:]

        clusters: list[SentenceCluster] = []
        for cluster_id in range(k):
            member_indices = [
                idx for idx, assignment in enumerate(assignments) if assignment == cluster_id
            ]
            if not member_indices:
                continue
            centroid = centroids[cluster_id]
            representative_index = min(
                member_indices,
                key=lambda idx: (
                    _euclidean_distance_sq(vector_list[idx], centroid),
                    len(sentences[idx]),
                    idx,
                ),
            )
            clusters.append(
                SentenceCluster(
                    cluster_id=cluster_id,
                    representative_index=representative_index,
                    representative_sentence=sentences[representative_index],
                    member_indices=sorted(member_indices),
                    context_sentences=self._gather_context(
                        representative_index, sentences
                    ),
                )
            )
        return clusters

    def _gather_context(self, index: int, sentences: Sequence[str]) -> list[str]:
        before = self._context_cfg.sentences_before
        after = self._context_cfg.sentences_after
        start = max(0, index - before)
        end = min(len(sentences), index + after + 1)
        return list(sentences[start:end])


def _detect_language(text: str) -> str | None:
    if not text.strip():
        return None
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return None


def _extract_keywords(text: str, limit: int = 5) -> list[str]:
    tokens = re.findall(r"[\w가-힣]+", text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) == limit:
            break
    return keywords


class DocumentClusterAgent:
    """텍스트 파일을 읽고 문장 군집 대표 블록을 생성합니다."""

    def __init__(
        self,
        agent_config: AgentConfig,
        embedding_service: EmbeddingServiceProtocol,
    ) -> None:
        self._config = agent_config
        self._embedding_service = embedding_service
        self._clusterer = SentenceClusterer(agent_config)

    def process_file(self, file_id: str, file_path: Path) -> dict:
        start = time.perf_counter()
        status = "OK"
        error_message: str | None = None
        retry_count = 0

        try:
            text = read_file(file_path)
        except UnicodeDecodeError as exc:
            status = "ERROR"
            error_message = str(exc)
            text = ""

        text_bytes = len(text.encode("utf-8")) if text else 0
        if status == "OK" and text_bytes > self._config.limits.max_text_bytes:
            status = "TEXT_TOO_LARGE"

        sentences = (
            split_text_by_newline_or_sentence(text) if status == "OK" else []
        )
        if status == "OK" and not sentences:
            status = "NO_CONTENT"

        embeddings: list[list[float]] = []
        clusters: list[SentenceCluster] = []
        if status == "OK":
            try:
                embeddings = self._embedding_service.embed_texts(sentences)
                clusters = self._clusterer.build_clusters(sentences, embeddings)
            except Exception as exc:  # pragma: no cover - 방어적 경로
                status = "ERROR"
                error_message = str(exc)

        short_summary: str | None = None
        long_summary: str | None = None
        tags: list[str] = []
        topics: list[str] = []
        language: str | None = None
        representative_blocks: list[dict[str, object]] = []
        final_embedding: list[float] | None = None

        if status == "OK":
            if clusters:
                representative_blocks = [
                    {
                        "cluster_id": cluster.cluster_id,
                        "representative_index": cluster.representative_index,
                        "representative_sentence": cluster.representative_sentence,
                        "context_sentences": cluster.context_sentences,
                        "member_indices": cluster.member_indices,
                    }
                    for cluster in clusters
                ]
                summary_sentences = [
                    " ".join(cluster.context_sentences) for cluster in clusters
                ]
                short_summary = " / ".join(
                    sentence for sentence in summary_sentences if sentence
                )
                long_summary = "\n".join(summary_sentences) or None
                language = _detect_language(short_summary)
                tags = _extract_keywords(short_summary)
            else:
                short_summary = ""
            if short_summary:
                final_embedding = self._embedding_service.embed_text(short_summary)

        processing_time_ms = int((time.perf_counter() - start) * 1000)

        return {
            "file_id": file_id,
            "status": status,
            "retry_count": retry_count,
            "processing_time_ms": processing_time_ms,
            "error_message": error_message,
            "short_summary": short_summary,
            "long_summary": long_summary,
            "tags": tags,
            "doc_type": None,
            "topics": topics,
            "language": language,
            "embedding": final_embedding,
            "raw_features": {
                "sentence_count": len(sentences),
                "chunk_count": len(representative_blocks),
                "text_bytes": text_bytes,
            },
            "representative_blocks": representative_blocks,
        }


def run_cluster_cli(target: Path) -> None:
    """간단한 CLI를 통해 문장 군집 결과를 출력합니다."""

    agent_config = load_agent_config()
    embedding_service = EmbeddingService(
        CLI_CONFIG.embedding_model_name,
        device=CLI_CONFIG.device,
        torch_dtype=CLI_CONFIG.torch_dtype,
    )
    processor = DocumentClusterAgent(agent_config, embedding_service)
    files = find_target_files(target)
    for index, file_path in enumerate(files, start=1):
        result = processor.process_file(f"file-{index}", file_path)
        print(f"===== {file_path} 결과 =====")
        for block in result.get("representative_blocks", []):
            print(
                f"[클러스터 {block['cluster_id']}] "
                f"{' | '.join(block['context_sentences'])}"
            )


__all__ = [
    "DocumentClusterAgent",
    "SentenceCluster",
    "SentenceClusterer",
    "run_cluster_cli",
]
