from __future__ import annotations

from typing import Any

import pytest

from transformer_client import TransformerClient, TransformerClientConfig


class DummyTokenizer:
    eos_token_id = 11
    pad_token_id = 0
    im_start_id = 97
    im_end_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 단순화를 위해 문자 길이를 토큰 길이로 취급한다.
        return [ord(ch) for ch in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> Any:
        user_message = messages[-1]["content"]
        assistant_prefix = "<|im_start|>assistant\n" if add_generation_prompt else ""
        formatted = (
            "<|im_start|>user\n"
            f"{user_message}\n"
            "<|im_end|>\n"
            f"{assistant_prefix}"
        )
        if tokenize:
            return [ord(ch) for ch in formatted]
        return formatted


def setup_dummy_client(config: TransformerClientConfig) -> TransformerClient:
    client = TransformerClient(config)
    client._tokenizer = DummyTokenizer()  # type: ignore[attr-defined]
    return client


def test_generate_uses_tokenizer_settings_and_stop_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TransformerClientConfig(stop_strings=("###",))
    client = setup_dummy_client(config)
    captured_kwargs: dict[str, Any] = {}
    received_prompts: list[Any] = []

    def fake_generator(prompt: Any, **kwargs: Any) -> list[dict[str, Any]]:
        captured_kwargs.update(kwargs)
        received_prompts.append(prompt)
        return [{"generated_text": "응답"}]

    monkeypatch.setattr(client, "_load_pipeline", lambda: fake_generator)

    result = client.generate("질문")

    assert result == "응답"
    assert received_prompts and str(received_prompts[0]).startswith("<|im_start|>user")
    assert captured_kwargs["eos_token_id"] == [
        DummyTokenizer.eos_token_id,
        DummyTokenizer.im_end_id,
    ]
    assert captured_kwargs["pad_token_id"] == DummyTokenizer.pad_token_id
    assert captured_kwargs["return_full_text"] is False
    assert captured_kwargs["stop_strings"] == list(config.stop_strings)


def test_generate_warns_and_appends_note_when_truncated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    config = TransformerClientConfig(max_new_tokens=5, truncation_suffix="[잘림]")
    client = setup_dummy_client(config)

    def fake_generator(prompt: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"generated_text": "abcde"}]

    monkeypatch.setattr(client, "_load_pipeline", lambda: fake_generator)

    with caplog.at_level("WARNING"):
        result = client.generate("프롬프트")

    assert "max_new_tokens" in caplog.text
    assert result == "abcde [잘림]"


def test_generate_retries_with_higher_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TransformerClientConfig(
        max_new_tokens=2,
        truncation_retry_max_new_tokens=4,
        truncation_suffix="[잘림]",
    )
    client = setup_dummy_client(config)
    responses = ["ab", "abc"]
    call_index = {"value": 0}
    captured_limits: list[int] = []

    def fake_generator(prompt: str, **kwargs: Any) -> list[dict[str, Any]]:
        captured_limits.append(kwargs["max_new_tokens"])
        idx = call_index["value"]
        call_index["value"] += 1
        return [{"generated_text": responses[idx]}]

    monkeypatch.setattr(client, "_load_pipeline", lambda: fake_generator)

    result = client.generate("프롬프트")

    assert captured_limits == [2, 4]
    assert result == "abc"


def test_generate_retries_with_formatted_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TransformerClientConfig(
        max_new_tokens=2,
        truncation_retry_max_new_tokens=4,
    )
    client = setup_dummy_client(config)
    prompts: list[Any] = []

    def fake_generator(prompt: Any, **kwargs: Any) -> list[dict[str, Any]]:
        prompts.append(prompt)
        generated_tokens = kwargs["max_new_tokens"]
        if len(prompts) == 1:
            return [
                {
                    "generated_text": "첫 번째",  # 잘리도록 토큰 수를 제한값과 맞춤
                    "generated_tokens": generated_tokens,
                }
            ]
        return [{"generated_text": "두 번째", "generated_tokens": 1}]

    monkeypatch.setattr(client, "_load_pipeline", lambda: fake_generator)

    result = client.generate("프롬프트")

    assert result == "두 번째"
    assert len(prompts) == 2
    assert prompts[0] == prompts[1]
    assert str(prompts[0]).startswith("<|im_start|>user")
