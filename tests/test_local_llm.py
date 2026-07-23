from __future__ import annotations

import json

import pytest

from backend.local_llm import (
    LocalLLMConfig,
    LocalLLMError,
    OllamaLocalProvider,
    assert_loopback_provider,
)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body

    def geturl(self) -> str:
        return "http://127.0.0.1:11434/api/chat"


class _RecordingOpener:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls = 0

    def open(self, request: object, *, timeout: float) -> _FakeResponse:
        del request, timeout
        self.calls += 1
        return _FakeResponse(self.body)


def _envelope(
    content: dict[str, object],
    **metadata: object,
) -> bytes:
    value = {
        "message": {
            "content": json.dumps(
                content,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        },
        **metadata,
    }
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _provider_with_body(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    *,
    config: LocalLLMConfig | None = None,
) -> tuple[OllamaLocalProvider, _RecordingOpener]:
    opener = _RecordingOpener(body)
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    return OllamaLocalProvider(config), opener


def test_provider_policy_rejects_remote_declaration_and_endpoint() -> None:
    class DeclaredRemoteProvider:
        provider_id = "declared-remote"
        provider_version = "1"
        network_policy = "remote-allowed"

    class RemoteEndpointProvider:
        provider_id = "remote-endpoint"
        provider_version = "1"
        network_policy = "loopback-only"
        endpoint = "https://example.com/v1"

    with pytest.raises(LocalLLMError, match="loopback-only"):
        assert_loopback_provider(DeclaredRemoteProvider())
    with pytest.raises(LocalLLMError, match="endpoint"):
        assert_loopback_provider(RemoteEndpointProvider())


def test_context_preflight_accepts_exact_boundary_and_blocks_overflow_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LocalLLMConfig(
        model="boundary-model",
        context_tokens=1_024,
        output_tokens=128,
    )
    provider, opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": "ok"}, done=True, eval_count=4),
        config=config,
    )

    result = provider.generate_json(
        system_prompt="",
        user_prompt="a" * 2_496,
        model="boundary-model",
    )
    assert result == {"answer": "ok"}
    assert opener.calls == 1

    with pytest.raises(LocalLLMError, match="context window"):
        provider.generate_json(
            system_prompt="",
            user_prompt="a" * 2_497,
            model="boundary-model",
        )
    assert opener.calls == 1


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"done": False}, "incomplete generation"),
        ({"done": True, "done_reason": "length"}, "truncated"),
        ({"done": True, "eval_count": 128}, "token budget"),
        ({"done": "yes"}, "invalid done flag"),
        ({"done": True, "eval_count": -1}, "invalid eval count"),
    ],
)
def test_provider_generation_envelope_must_prove_non_truncation(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, object],
    message: str,
) -> None:
    provider, opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": "unsafe"}, **metadata),
        config=LocalLLMConfig(output_tokens=128),
    )

    with pytest.raises(LocalLLMError, match=message):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="qwen3.5:4b",
        )
    assert opener.calls == 1


def test_response_schema_mismatch_fails_closed_and_valid_output_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }
    invalid_provider, invalid_opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": 7}, done=True),
    )
    with pytest.raises(LocalLLMError, match="failed response schema"):
        invalid_provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="qwen3.5:4b",
            response_schema=schema,
        )
    assert invalid_opener.calls == 1

    valid_provider, valid_opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": "ok"}, done=True),
    )
    assert valid_provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model="qwen3.5:4b",
        response_schema=schema,
    ) == {"answer": "ok"}
    assert valid_opener.calls == 1


def test_invalid_response_schema_is_rejected_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": "unused"}, done=True),
    )

    with pytest.raises(LocalLLMError, match="response schema"):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="qwen3.5:4b",
            response_schema={"type": "not-a-json-schema-type"},
        )
    assert opener.calls == 0


def test_duplicate_provider_envelope_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        b'{"message":{"content":"{\\"answer\\":\\"first\\"}"},'
        b'"message":{"content":"{\\"answer\\":\\"second\\"}"}}'
    )
    provider, opener = _provider_with_body(monkeypatch, body)

    with pytest.raises(LocalLLMError, match="malformed JSON"):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="qwen3.5:4b",
        )
    assert opener.calls == 1
