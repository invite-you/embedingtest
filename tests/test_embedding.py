from __future__ import annotations

from types import SimpleNamespace

import pytest

from embedding import EmbeddingConfig, TextEmbedder


class DummyTensor:
    def __init__(self, data):
        self.data = data

    def to(self, device):
        return self

    def mean(self, dim: int):
        if dim != 1:
            raise AssertionError("dim=1만 지원합니다")
        if not isinstance(self.data, list):
            raise AssertionError("리스트 입력만 지원합니다")
        means = [sum(row) / len(row) for row in self.data]
        return DummyTensor(means)

    def squeeze(self):
        if isinstance(self.data, list) and len(self.data) == 1:
            return DummyTensor(self.data[0])
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.data


class DummyTorch:
    class _CUDA:
        @staticmethod
        def is_available() -> bool:
            return False

    cuda = _CUDA()

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def no_grad(self):
        return self._NoGrad()


class DummyTokenizer:
    def __call__(self, text: str, **_: object):
        return {"input_ids": DummyTensor([[float(len(text)), 1.0]])}


class DummyModel:
    def __init__(self) -> None:
        self.device = None

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        pass

    def __call__(self, **_: object):
        return SimpleNamespace(last_hidden_state=DummyTensor([[0.2, 0.4, 0.6]]))


def dummy_loader(_: str):
    return DummyTokenizer(), DummyModel()


def test_text_embedder_returns_mean_vector() -> None:
    config = EmbeddingConfig(model_name="dummy", max_length=16)
    embedder = TextEmbedder(config=config, loader=dummy_loader, torch_module=DummyTorch())

    result = embedder.embed_text("테스트 문장")

    assert result == pytest.approx([0.4])
