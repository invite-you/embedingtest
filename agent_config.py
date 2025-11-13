"""에이전트 동작을 제어하는 구성 로더입니다."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


try:  # pragma: no cover - PyYAML 미설치 환경 대비
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - CI 환경에 따라 달라짐
    yaml = None


class AgentConfigError(RuntimeError):
    """구성 파일 파싱이나 검증 실패를 나타냅니다."""


@dataclass(frozen=True)
class TimeoutsConfig:
    per_file_minutes: int


@dataclass(frozen=True)
class LimitsConfig:
    max_text_bytes: int


@dataclass(frozen=True)
class ClusteringConfig:
    max_clusters: int
    min_sentences: int


@dataclass(frozen=True)
class ContextWindowConfig:
    sentences_before: int
    sentences_after: int


@dataclass(frozen=True)
class LLMConfig:
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    batch_size: int


@dataclass(frozen=True)
class AgentConfig:
    timeouts: TimeoutsConfig
    limits: LimitsConfig
    clustering: ClusteringConfig
    context: ContextWindowConfig
    llm: LLMConfig
    embedding: EmbeddingRuntimeConfig


DEFAULT_CONFIG_PATH = Path("config/agent.yaml")


def _parse_yaml_or_json(text: str) -> Mapping[str, Any]:
    if yaml is not None:  # pragma: no branch - 분기 단순화
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise AgentConfigError("config/agent.yaml의 루트는 매핑이어야 합니다.")
    return data


def _require_positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or value <= 0:
        raise AgentConfigError(f"`{key}`는 양의 정수여야 합니다.")
    return value


def _load_section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = mapping.get(key)
    if not isinstance(section, Mapping):
        raise AgentConfigError(f"`{key}` 섹션이 누락되었거나 올바르지 않습니다.")
    return section


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """config/agent.yaml을 읽어 검증된 설정 객체를 반환합니다."""

    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise AgentConfigError(f"구성 파일을 찾을 수 없습니다: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    mapping = _parse_yaml_or_json(raw_text)

    timeouts_section = _load_section(mapping, "timeouts")
    limits_section = _load_section(mapping, "limits")
    clustering_section = _load_section(mapping, "clustering")
    context_section = _load_section(mapping, "context")
    llm_section = _load_section(mapping, "llm")
    embedding_section = _load_section(mapping, "embedding")

    timeouts = TimeoutsConfig(
        per_file_minutes=_require_positive_int(timeouts_section, "per_file_minutes"),
    )
    limits = LimitsConfig(
        max_text_bytes=_require_positive_int(limits_section, "max_text_bytes"),
    )
    clustering = ClusteringConfig(
        max_clusters=_require_positive_int(clustering_section, "max_clusters"),
        min_sentences=_require_positive_int(clustering_section, "min_sentences"),
    )
    context = ContextWindowConfig(
        sentences_before=_require_positive_int(
            context_section, "sentences_before"
        ),
        sentences_after=_require_positive_int(context_section, "sentences_after"),
    )
    llm = LLMConfig(max_tokens=_require_positive_int(llm_section, "max_tokens"))
    embedding = EmbeddingRuntimeConfig(
        batch_size=_require_positive_int(embedding_section, "batch_size"),
    )

    return AgentConfig(
        timeouts=timeouts,
        limits=limits,
        clustering=clustering,
        context=context,
        llm=llm,
        embedding=embedding,
    )


__all__ = [
    "AgentConfig",
    "AgentConfigError",
    "ClusteringConfig",
    "ContextWindowConfig",
    "EmbeddingRuntimeConfig",
    "LimitsConfig",
    "LLMConfig",
    "TimeoutsConfig",
    "load_agent_config",
]
