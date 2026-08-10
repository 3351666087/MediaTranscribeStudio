from __future__ import annotations

import io
import json
import urllib.error

import pytest

from backend.local_llm import (
    LocalLLMConfig,
    LocalLLMContextWindowError,
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

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    def geturl(self) -> str:
        return "http://127.0.0.1:11434/api/chat"


class _RecordingOpener:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls = 0
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def open(self, request: object, *, timeout: float) -> _FakeResponse:
        self.calls += 1
        self.requests.append(request)
        self.timeouts.append(timeout)
        return _FakeResponse(self.body)


class _RejectingOpener:
    def open(self, request: object, *, timeout: float) -> _FakeResponse:
        del timeout
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(
                json.dumps(
                    {
                        "error": json.dumps(
                            {
                                "error": {
                                    "code": 400,
                                    "message": (
                                        "Failed to initialize samplers: "
                                        "failed to parse grammar"
                                    ),
                                    "type": "invalid_request_error",
                                }
                            }
                        )
                    }
                ).encode("utf-8")
            ),
        )


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
        _envelope(
            {"answer": "ok"},
            done=True,
            total_duration=120,
            load_duration=20,
            prompt_eval_count=30,
            prompt_eval_duration=40,
            eval_count=4,
            eval_duration=50,
        ),
        config=config,
    )

    result = provider.generate_json(
        system_prompt="",
        user_prompt="a" * 2_496,
        model="boundary-model",
    )
    assert result == {"answer": "ok"}
    assert opener.calls == 1
    assert provider.generation_metrics == {
        "completedCalls": 1,
        "totalDurationNanoseconds": 120,
        "loadDurationNanoseconds": 20,
        "promptEvalTokens": 30,
        "promptEvalDurationNanoseconds": 40,
        "outputTokens": 4,
        "outputEvalDurationNanoseconds": 50,
    }

    with pytest.raises(LocalLLMContextWindowError, match="context window"):
        provider.generate_json(
            system_prompt="",
            user_prompt="a" * 2_497,
            model="boundary-model",
        )
    assert opener.calls == 1


def test_expected_model_digest_is_verified_once_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64

    class Response:
        def __init__(self, body: bytes, url: str) -> None:
            self.body = body
            self.url = url

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return self.body if size < 0 else self.body[:size]

        def geturl(self) -> str:
            return self.url

    class Opener:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def open(self, request: object, *, timeout: float) -> Response:
            del timeout
            url = request.full_url
            self.urls.append(url)
            if url.endswith("/api/tags"):
                body = json.dumps(
                    {
                        "models": [
                            {
                                "name": "frozen-model:9b",
                                "digest": f"sha256:{digest}",
                            }
                        ]
                    }
                ).encode("utf-8")
            else:
                body = _envelope({"answer": "ok"}, done=True)
            return Response(body, url)

    opener = Opener()
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    provider = OllamaLocalProvider(
        LocalLLMConfig(
            model="frozen-model:9b",
            expected_model_digest=digest.upper(),
        )
    )

    for _ in range(2):
        assert provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="frozen-model:9b",
        ) == {"answer": "ok"}

    assert opener.urls == [
        "http://127.0.0.1:11434/api/tags",
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:11434/api/chat",
    ]


def test_expected_model_digest_mismatch_fails_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, opener = _provider_with_body(
        monkeypatch,
        json.dumps(
            {
                "models": [
                    {
                        "name": "frozen-model:9b",
                        "digest": "b" * 64,
                    }
                ]
            }
        ).encode("utf-8"),
        config=LocalLLMConfig(
            model="frozen-model:9b",
            expected_model_digest="a" * 64,
        ),
    )

    with pytest.raises(LocalLLMError, match="digest does not match") as captured:
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="frozen-model:9b",
        )

    assert opener.calls == 1
    assert captured.value.diagnostics["failureStage"] == "model-digest"
    assert captured.value.diagnostics["actualDigest"] == "sha256:" + "b" * 64


def test_stage_reload_reverifies_model_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_digest = "a" * 64

    class DigestChangingOpener:
        def __init__(self) -> None:
            self.inventory_calls = 0
            self.urls: list[str] = []

        def open(self, request: object, *, timeout: float) -> _FakeResponse:
            del timeout
            url = request.full_url
            self.urls.append(url)
            if url.endswith("/api/tags"):
                self.inventory_calls += 1
                digest = (
                    expected_digest
                    if self.inventory_calls == 1
                    else "b" * 64
                )
                body = json.dumps(
                    {
                        "models": [
                            {
                                "name": "frozen-model:9b",
                                "digest": digest,
                            }
                        ]
                    }
                ).encode("utf-8")
            elif url.endswith("/api/generate"):
                body = b'{"done":true,"done_reason":"unload"}'
            else:
                body = _envelope({"answer": "ok"}, done=True)
            return _FakeResponse(body)

    opener = DigestChangingOpener()
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    provider = OllamaLocalProvider(
        LocalLLMConfig(
            model="frozen-model:9b",
            expected_model_digest=expected_digest,
            release_on_close=True,
        )
    )

    assert provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model="frozen-model:9b",
    ) == {"answer": "ok"}
    provider.release_resources()

    with pytest.raises(LocalLLMError, match="digest does not match"):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="frozen-model:9b",
        )
    assert opener.urls == [
        "http://127.0.0.1:11434/api/tags",
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:11434/api/generate",
        "http://127.0.0.1:11434/api/tags",
    ]


def test_stage_scoped_provider_keeps_model_warm_then_unloads_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": "ok"}, done=True),
        config=LocalLLMConfig(
            model="stage-model",
            timeout_seconds=120,
            keep_alive="5m",
            release_on_close=True,
        ),
    )

    assert provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model="stage-model",
    ) == {"answer": "ok"}
    provider.release_resources()
    provider.release_resources()

    assert opener.calls == 2
    generation_request, release_request = opener.requests
    assert generation_request.full_url == "http://127.0.0.1:11434/api/chat"
    assert json.loads(generation_request.data)["keep_alive"] == "5m"
    assert release_request.full_url == "http://127.0.0.1:11434/api/generate"
    assert json.loads(release_request.data) == {
        "model": "stage-model",
        "keep_alive": 0,
        "stream": False,
    }
    assert opener.timeouts == [120, 30.0]


def test_stage_release_requires_strict_unload_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, opener = _provider_with_body(
        monkeypatch,
        b'{"done":false,"error":"model stayed loaded"}',
        config=LocalLLMConfig(
            model="stage-model",
            release_on_close=True,
        ),
    )

    with pytest.raises(LocalLLMError, match="was not acknowledged"):
        provider.release_resources()
    with pytest.raises(LocalLLMError, match="was not acknowledged"):
        provider.release_resources()
    assert opener.calls == 2


def test_worker_scoped_provider_release_does_not_unload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": "unused"}, done=True),
        config=LocalLLMConfig(keep_alive="10m", release_on_close=False),
    )

    provider.release_resources()

    assert opener.calls == 0


def test_stage_scoped_provider_can_reload_after_intermediate_unload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": "ok"}, done=True),
        config=LocalLLMConfig(
            model="round-model",
            keep_alive="5m",
            release_on_close=True,
        ),
    )

    for _ in range(2):
        assert provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="round-model",
        ) == {"answer": "ok"}
        provider.release_resources()
        provider.release_resources()

    assert [request.full_url for request in opener.requests] == [
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:11434/api/generate",
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:11434/api/generate",
    ]


def test_release_on_close_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="release_on_close"):
        LocalLLMConfig(release_on_close=1)  # type: ignore[arg-type]


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
            model=provider.config.model,
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
    with pytest.raises(
        LocalLLMError,
        match="failed response schema",
    ) as captured:
        invalid_provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=invalid_provider.config.model,
            response_schema=schema,
        )
    assert captured.value.diagnostics == {
        "failureStage": "response-schema",
        "responseFields": ["answer"],
        "decisionCount": None,
        "translationCount": None,
        "schemaErrorPath": "$.answer",
        "schemaValidator": "type",
        "responseContentPersisted": False,
    }
    assert invalid_opener.calls == 1

    valid_provider, valid_opener = _provider_with_body(
        monkeypatch,
        _envelope({"answer": "ok"}, done=True),
    )
    assert valid_provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model=valid_provider.config.model,
        response_schema=schema,
    ) == {"answer": "ok"}
    assert valid_opener.calls == 1


def test_http_schema_grammar_rejection_is_classified_without_raw_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _RejectingOpener()
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    provider = OllamaLocalProvider()

    with pytest.raises(
        LocalLLMError,
        match="request was rejected",
    ) as captured:
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=provider.config.model,
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        )

    assert captured.value.diagnostics == {
        "failureStage": "transport-http",
        "httpStatus": 400,
        "providerErrorCode": 400,
        "providerErrorType": "invalid_request_error",
        "providerErrorCategory": "schema-grammar-initialization",
        "responseContentPersisted": False,
    }


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
            model=provider.config.model,
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
            model=provider.config.model,
        )
    assert opener.calls == 1
