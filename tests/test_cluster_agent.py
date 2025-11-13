from pathlib import Path

import pytest

from agent_config import (
    AgentConfig,
    AgentConfigError,
    ClusteringConfig,
    ContextWindowConfig,
    EmbeddingRuntimeConfig,
    LimitsConfig,
    LLMConfig,
    TimeoutsConfig,
    load_agent_config,
)
from cluster_agent import DocumentClusterAgent, SentenceClusterer


class _DummyEmbeddingService:
    def __init__(self) -> None:
        self.model_name = "dummy/model"

    def embed_texts(self, texts):
        return [[float(len(text))] for text in texts]

    def embed_text(self, text):
        return [float(len(text))]


def create_agent_config(
    *,
    distance_ratio: float = 1.2,
    max_representatives: int = 3,
) -> AgentConfig:
    return AgentConfig(
        timeouts=TimeoutsConfig(per_file_minutes=50),
        limits=LimitsConfig(max_text_bytes=10_000),
        clustering=ClusteringConfig(
            max_clusters=4,
            min_sentences=2,
            min_cluster_size_for_output=2,
            representative_distance_ratio=distance_ratio,
            max_representative_sentences=max_representatives,
        ),
        context=ContextWindowConfig(sentences_before=1, sentences_after=1),
        llm=LLMConfig(max_tokens=2048),
        embedding=EmbeddingRuntimeConfig(batch_size=64),
    )


def test_load_agent_config_reads_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        """
{
  "timeouts": {"per_file_minutes": 60},
  "limits": {"max_text_bytes": 1000},
  "clustering": {"max_clusters": 5, "min_sentences": 3, "min_cluster_size_for_output": 2, "representative_distance_ratio": 1.5, "max_representative_sentences": 2},
  "context": {"sentences_before": 1, "sentences_after": 2},
  "llm": {"max_tokens": 4096},
  "embedding": {"batch_size": 32}
}
""".strip()
    )

    config = load_agent_config(config_path)

    assert config.timeouts.per_file_minutes == 60
    assert config.context.sentences_after == 2


def test_load_agent_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AgentConfigError):
        load_agent_config(tmp_path / "missing.yaml")


def test_sentence_clusterer_returns_representative_sentences() -> None:
    config = create_agent_config()
    clusterer = SentenceClusterer(config)
    sentences = ["첫 문장", "둘째 문장", "셋째 문장", "넷째 문장"]
    embeddings = [[float(i)] for i in range(len(sentences))]

    clusters = clusterer.build_clusters(sentences, embeddings)

    assert clusters
    assert all(cluster.representative_sentences for cluster in clusters)


def test_sentence_clusterer_skips_weak_clusters() -> None:
    config = create_agent_config()
    clusterer = SentenceClusterer(config)
    sentences = ["첫 문장", "둘째 문장"]
    embeddings = [[0.0], [10.0]]

    clusters = clusterer.build_clusters(sentences, embeddings)

    assert clusters == []


def test_rank_member_indices_enforces_distance_ratio() -> None:
    config = create_agent_config(distance_ratio=1.01)
    clusterer = SentenceClusterer(config)
    sentences = ["가", "나", "다"]
    centroid = [0.0]
    vectors = [[0.2], [0.25], [0.9]]

    indices = clusterer._rank_member_indices([0, 1, 2], centroid, sentences, vectors)

    assert indices == [0]


def test_select_unique_indices_prefers_denser_cluster() -> None:
    config = create_agent_config()
    clusterer = SentenceClusterer(config)
    sentences = ["중복", "중복", "유일"]
    candidates = [
        {"cluster_id": 0, "member_indices": [0, 2], "ranked_indices": [0, 2]},
        {"cluster_id": 1, "member_indices": [1], "ranked_indices": [1]},
    ]

    selection = clusterer._select_unique_indices(candidates, sentences)

    assert selection[0] == [0, 2]
    assert 1 not in selection


def test_document_cluster_agent_builds_representative_blocks(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "첫 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다. 네 번째 문장입니다.",
        encoding="utf-8",
    )
    agent = DocumentClusterAgent(create_agent_config(), _DummyEmbeddingService())

    result = agent.process_file("file-1", file_path)

    assert result["status"] == "OK"
    assert result["representative_blocks"]
    assert result["short_summary"]
    assert result["embedding"] is not None
