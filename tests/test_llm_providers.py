from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from backend.errors import WorkerError
from backend.local_llm import (
    AnthropicProvider,
    AnthropicProviderConfig,
    CustomJSONProvider,
    CustomJSONProviderConfig,
    GoogleGeminiProvider,
    GoogleGeminiProviderConfig,
    HuggingFaceRouterProvider,
    LLMProviderConfig,
    LocalLLMConfig,
    LocalLLMError,
    OllamaLocalProvider,
    OpenAICompatibleProvider,
    assert_loopback_provider,
    assert_provider_network_policy,
    create_llm_provider,
    provider_config_from_mapping,
)
from backend.paths import PathPolicy
from backend.service import WorkerService


class _Response:
    def __init__(self, body: bytes, url: str) -> None:
        self.body = body
        self.url = url

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    def geturl(self) -> str:
        return self.url


class _RecordingOpener:
    def __init__(self, body: dict[str, Any], url: str) -> None:
        self.body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.url = url
        self.requests: list[Any] = []
        self.timeouts: list[float] = []

    def open(self, request: Any, *, timeout: float) -> _Response:
        self.requests.append(request)
        self.timeouts.append(timeout)
        return _Response(self.body, self.url)


def _request_headers(request: urllib.request.Request) -> dict[str, str]:
    return {name.casefold(): value for name, value in request.header_items()}


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }


def test_network_policy_allows_https_and_loopback_but_rejects_remote_http() -> None:
    remote = LLMProviderConfig(endpoint="https://relay.example/v1")
    loopback = LLMProviderConfig(
        endpoint="http://127.0.0.1:8080/v1",
        network_policy="loopback-only",
        api_key_env=None,
    )
    assert remote.network_policy == "remote-explicit"
    assert loopback.network_policy == "loopback-only"

    with pytest.raises(ValueError, match="HTTPS"):
        LLMProviderConfig(endpoint="http://relay.example/v1")
    with pytest.raises(ValueError, match="credentials"):
        LLMProviderConfig(endpoint="https://user:secret@relay.example/v1")
    with pytest.raises(ValueError, match="query"):
        LLMProviderConfig(endpoint="https://relay.example/v1?key=secret")


def test_general_network_assertion_preserves_legacy_loopback_assertion() -> None:
    remote = OpenAICompatibleProvider(
        LLMProviderConfig(endpoint="https://relay.example/v1")
    )
    local = OpenAICompatibleProvider(
        LLMProviderConfig(
            endpoint="http://localhost:8080/v1",
            network_policy="loopback-only",
            api_key_env=None,
        )
    )

    assert assert_provider_network_policy(remote) == "remote-explicit"
    assert assert_provider_network_policy(local) == "loopback-only"
    assert assert_loopback_provider(local) == "loopback-only"
    with pytest.raises(LocalLLMError, match="loopback-only"):
        assert_loopback_provider(remote)

    class InvalidDeclaration:
        provider_id = "invalid"
        provider_version = "1"
        network_policy = "loopback-only"
        endpoint = "https://relay.example/v1"

    with pytest.raises(LocalLLMError, match="network policy"):
        assert_provider_network_policy(InvalidDeclaration())


def test_serializable_config_rejects_raw_keys_and_secret_headers() -> None:
    with pytest.raises(ValueError, match="raw API keys"):
        provider_config_from_mapping(
            {
                "provider": "openai-compatible",
                "apiKey": "must-not-be-stored",
            }
        )
    with pytest.raises(ValueError, match="raw secrets"):
        LLMProviderConfig(headers={"Authorization": "Bearer raw-secret"})
    with pytest.raises(ValueError, match="authentication material"):
        LLMProviderConfig(extra_body={"api_key": "raw-secret"})

    resolver_secret = "resolver-secret-value"
    config = LLMProviderConfig(
        secret_resolver=lambda name: resolver_secret if name else None
    )
    assert resolver_secret not in repr(config)


def test_openai_compatible_request_uses_env_key_proxy_and_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "https://relay.example/v1/chat/completions"
    opener = _RecordingOpener(
        {
            "choices": [
                {
                    "message": {"content": '{"answer":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
            },
        },
        endpoint,
    )
    handlers: list[Any] = []

    def build_opener(*values: Any) -> _RecordingOpener:
        handlers.extend(values)
        return opener

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        build_opener,
    )
    monkeypatch.setenv("RELAY_API_KEY", "env-secret")
    provider = OpenAICompatibleProvider(
        LLMProviderConfig(
            model="relay-model",
            endpoint="https://relay.example/v1",
            timeout_seconds=12.5,
            api_key_env="RELAY_API_KEY",
            require_api_key=True,
            proxy="http://127.0.0.1:7890",
        )
    )

    assert provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model="relay-model",
        response_schema=_schema(),
    ) == {"answer": "ok"}

    request = opener.requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    headers = _request_headers(request)
    assert request.full_url == endpoint
    assert opener.timeouts == [12.5]
    assert headers["authorization"] == "Bearer env-secret"
    assert "env-secret" not in request.data.decode("utf-8")
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "mts_structured_response",
            "strict": True,
            "schema": _schema(),
        },
    }
    proxy_handlers = [
        handler
        for handler in handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers[0].proxies == {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }
    assert provider.generation_metrics == {
        "completedCalls": 1,
        "promptTokens": 11,
        "outputTokens": 4,
        "totalTokens": 15,
    }


def test_secret_resolver_and_header_env_are_resolved_only_at_request_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "https://relay.example/v1/chat/completions"
    opener = _RecordingOpener(
        {
            "choices": [
                {
                    "message": {"content": {"answer": "ok"}},
                    "finish_reason": "stop",
                }
            ]
        },
        endpoint,
    )
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    monkeypatch.setenv("TENANT_HEADER", "tenant-a")
    calls: list[str] = []

    def resolver(name: str) -> str:
        calls.append(name)
        return "resolver-secret"

    provider = OpenAICompatibleProvider(
        LLMProviderConfig(
            model="relay-model",
            endpoint="https://relay.example/v1",
            api_key_env=None,
            secret_name="relay/main",
            require_api_key=True,
            header_env={"X-Tenant": "TENANT_HEADER"},
        ),
        secret_resolver=resolver,
    )
    assert calls == []

    assert provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model="relay-model",
    ) == {"answer": "ok"}
    headers = _request_headers(opener.requests[0])
    assert calls == ["relay/main"]
    assert headers["authorization"] == "Bearer resolver-secret"
    assert headers["x-tenant"] == "tenant-a"


def test_missing_required_key_fails_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "https://relay.example/v1/chat/completions"
    opener = _RecordingOpener({}, endpoint)
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    monkeypatch.delenv("MISSING_PROVIDER_KEY", raising=False)
    provider = OpenAICompatibleProvider(
        LLMProviderConfig(
            endpoint="https://relay.example/v1",
            api_key_env="MISSING_PROVIDER_KEY",
            require_api_key=True,
        )
    )

    with pytest.raises(LocalLLMError, match="unavailable") as captured:
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=provider.config.model,
        )
    assert opener.requests == []
    assert captured.value.diagnostics["failureStage"] == "authentication"


def test_http_error_diagnostics_do_not_persist_or_echo_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-key-value"

    class RejectingOpener:
        def open(self, request: Any, *, timeout: float) -> _Response:
            del timeout
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                {},
                io.BytesIO(
                    json.dumps(
                        {
                            "error": {
                                "type": "authentication_error",
                                "message": f"bad key {secret}",
                            }
                        }
                    ).encode("utf-8")
                ),
            )

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: RejectingOpener(),
    )
    provider = OpenAICompatibleProvider(
        LLMProviderConfig(
            endpoint="https://relay.example/v1",
            api_key_env=None,
            require_api_key=True,
        ),
        secret_resolver=lambda name: secret,
    )

    with pytest.raises(LocalLLMError, match="rejected") as captured:
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=provider.config.model,
        )
    rendered = str(captured.value) + json.dumps(captured.value.diagnostics)
    assert secret not in rendered
    assert captured.value.diagnostics == {
        "failureStage": "transport-http",
        "httpStatus": 401,
        "responseContentPersisted": False,
        "providerErrorType": "authentication_error",
    }


def test_custom_json_template_and_response_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://127.0.0.1:8080/invoke"
    opener = _RecordingOpener(
        {"result": {"data": {"answer": "custom"}}},
        endpoint,
    )
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    provider = CustomJSONProvider(
        CustomJSONProviderConfig(
            endpoint="http://127.0.0.1:8080",
            model="private-model",
            chat_path="/invoke",
            request_template={
                "engine": "${model}",
                "instruction": {
                    "system": "{{system_prompt}}",
                    "user": "${user_prompt}",
                },
                "limit": "${max_tokens}",
                "schema": "${response_schema}",
            },
            response_path="/result/data",
            finish_reason_path=None,
        )
    )

    assert provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model="private-model",
        response_schema=_schema(),
    ) == {"answer": "custom"}
    payload = json.loads(opener.requests[0].data.decode("utf-8"))
    assert payload == {
        "engine": "private-model",
        "instruction": {"system": "system", "user": "user"},
        "limit": 1024,
        "schema": _schema(),
    }


def test_anthropic_native_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "https://api.anthropic.com/v1/messages"
    opener = _RecordingOpener(
        {
            "content": [{"type": "text", "text": '{"answer":"claude"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 3},
        },
        endpoint,
    )
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    provider = AnthropicProvider(AnthropicProviderConfig())

    assert provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model=provider.config.model,
        response_schema=_schema(),
    ) == {"answer": "claude"}
    request = opener.requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    headers = _request_headers(request)
    assert request.full_url == endpoint
    assert headers["x-api-key"] == "anthropic-secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert payload["system"] == "system"
    assert payload["messages"] == [{"role": "user", "content": "user"}]
    assert payload["output_config"]["format"]["schema"] == _schema()
    assert provider.generation_metrics["totalTokens"] == 11


def test_google_gemini_native_request_keeps_key_out_of_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-pro:generateContent"
    )
    opener = _RecordingOpener(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"answer":"gemini"}'}]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 9,
                "candidatesTokenCount": 3,
                "totalTokenCount": 12,
            },
        },
        endpoint,
    )
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    provider = GoogleGeminiProvider(GoogleGeminiProviderConfig())

    assert provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model=provider.config.model,
        response_schema=_schema(),
    ) == {"answer": "gemini"}
    request = opener.requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    headers = _request_headers(request)
    assert request.full_url == endpoint
    assert "gemini-secret" not in request.full_url
    assert "?key=" not in request.full_url
    assert headers["x-goog-api-key"] == "gemini-secret"
    assert payload["systemInstruction"] == {"parts": [{"text": "system"}]}
    assert payload["generationConfig"]["responseJsonSchema"] == _schema()
    assert provider.generation_metrics["totalTokens"] == 12


@pytest.mark.parametrize(
    ("provider", "body", "message"),
    [
        (
            "anthropic",
            {
                "content": [{"type": "text", "text": '{"answer":"partial"}'}],
                "stop_reason": "max_tokens",
            },
            "truncated",
        ),
        (
            "gemini",
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"answer":"partial"}'}]
                        },
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            },
            "truncated",
        ),
    ],
)
def test_native_providers_fail_closed_on_truncation(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    body: dict[str, Any],
    message: str,
) -> None:
    if provider == "anthropic":
        config = AnthropicProviderConfig(require_api_key=False)
        endpoint = "https://api.anthropic.com/v1/messages"
        instance: Any = AnthropicProvider(config)
    else:
        config = GoogleGeminiProviderConfig(require_api_key=False)
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-pro:generateContent"
        )
        instance = GoogleGeminiProvider(config)
    opener = _RecordingOpener(body, endpoint)
    monkeypatch.setattr(instance, "_opener", opener)

    with pytest.raises(LocalLLMError, match=message):
        instance.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=config.model,
        )


def test_factory_covers_local_native_router_and_openai_presets() -> None:
    assert isinstance(create_llm_provider(LocalLLMConfig()), OllamaLocalProvider)
    assert isinstance(create_llm_provider("anthropic"), AnthropicProvider)
    assert isinstance(create_llm_provider("gemini"), GoogleGeminiProvider)
    assert isinstance(
        create_llm_provider("huggingface"),
        HuggingFaceRouterProvider,
    )
    deepseek = create_llm_provider("deepseek")
    assert isinstance(deepseek, OpenAICompatibleProvider)
    assert deepseek.config.endpoint == "https://api.deepseek.com/v1"
    assert deepseek.config.api_key_env == "DEEPSEEK_API_KEY"
    neutral_anthropic = create_llm_provider(
        LLMProviderConfig(
            provider="anthropic",
            model="claude-production",
            endpoint="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )
    )
    assert isinstance(neutral_anthropic, AnthropicProvider)
    assert neutral_anthropic.config.auth_header == "x-api-key"
    assert neutral_anthropic.config.chat_path == "/messages"
    neutral_gemini = create_llm_provider(
        LLMProviderConfig(
            provider="google-gemini",
            model="gemini-production",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
        )
    )
    assert isinstance(neutral_gemini, GoogleGeminiProvider)
    assert neutral_gemini.config.auth_header == "x-goog-api-key"
    assert neutral_gemini.config.chat_path == ""
    azure = create_llm_provider(
        LLMProviderConfig(
            provider="azure-enterprise",
            model="deployment-name",
            endpoint="https://example.openai.azure.com/openai/deployments/model",
            api_key_env="AZURE_OPENAI_API_KEY",
            require_api_key=True,
        )
    )
    assert isinstance(azure, OpenAICompatibleProvider)
    assert azure.provider_id == "azure-enterprise"

    local = provider_config_from_mapping(
        {
            "provider": "ollama",
            "model": "qwen3.5:27b-q4_K_M",
            "expectedDigest": "a" * 64,
        }
    )
    assert isinstance(local, LocalLLMConfig)
    assert local.expected_model_digest == "sha256:" + "a" * 64


def test_model_override_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    opener = _RecordingOpener(
        {
            "choices": [
                {
                    "message": {"content": {"answer": "ok"}},
                    "finish_reason": "stop",
                }
            ]
        },
        endpoint,
    )
    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: opener,
    )
    strict = OpenAICompatibleProvider(
        LLMProviderConfig(
            model="configured",
            endpoint="http://127.0.0.1:8080/v1",
            network_policy="loopback-only",
            api_key_env=None,
        )
    )
    with pytest.raises(LocalLLMError, match="does not match"):
        strict.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="override",
        )
    assert opener.requests == []

    flexible = OpenAICompatibleProvider(
        LLMProviderConfig(
            model="configured",
            endpoint="http://127.0.0.1:8080/v1",
            network_policy="loopback-only",
            api_key_env=None,
            allow_model_override=True,
        )
    )
    assert flexible.generate_json(
        system_prompt="system",
        user_prompt="user",
        model="override",
    ) == {"answer": "ok"}
    assert json.loads(opener.requests[-1].data)["model"] == "override"


def _provider_service(
    root: Path,
    *,
    config: LLMProviderConfig,
) -> WorkerService:
    input_root = root / "input"
    output_root = root / "output"
    input_root.mkdir()
    output_root.mkdir()
    (input_root / "source.wav").write_bytes(b"fixture")
    return WorkerService(
        path_policy=PathPolicy(
            allowed_input_roots=[input_root],
            allowed_output_root=output_root,
        ),
        semantic_model=config.model,
        llm_provider_config=config,
    )


def test_service_parses_remote_explicit_provider_without_accepting_raw_key(
    tmp_path: Path,
) -> None:
    config = LLMProviderConfig(
        provider="azure-enterprise",
        model="deployment-a",
        endpoint="https://example.openai.azure.com/openai/deployments/a",
        api_key_env="AZURE_OPENAI_API_KEY",
        require_api_key=True,
        allow_model_override=True,
    )
    service = _provider_service(tmp_path, config=config)
    try:
        request = service.parse_start_payload(
            {
                "jobId": "remote-provider",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "localLlmMode": "suggestion-only",
                "localLlmModel": "deployment-b",
                "llmProvider": "azure-enterprise",
                "llmEndpoint": (
                    "https://example.openai.azure.com/openai/deployments/b"
                ),
                "endpointPolicy": "remote-explicit",
                "llmApiKeyEnv": "AZURE_OPENAI_API_KEY",
                "llmProxyUrl": "http://127.0.0.1:7890",
                "llmAllowModelOverride": True,
            }
        )
    finally:
        service.shutdown()
    assert request.llm_provider == "azure-enterprise"
    assert request.llm_network_policy == "remote-explicit"
    assert request.local_llm_model == "deployment-b"
    assert request.llm_api_key_env == "AZURE_OPENAI_API_KEY"
    assert request.llm_proxy == "http://127.0.0.1:7890"
    assert isinstance(request.llm_provider_config, LLMProviderConfig)
    assert request.llm_provider_config.allow_model_override is True


def test_service_remote_profile_defaults_to_task_model_override(
    tmp_path: Path,
) -> None:
    """Remote relays may expose a different deployment id per task."""

    config = LLMProviderConfig(
        provider="enterprise-relay",
        model="configured-deployment",
        endpoint="https://relay.example/v1",
        api_key_env="RELAY_API_KEY",
        require_api_key=True,
    )
    service = _provider_service(tmp_path, config=config)
    try:
        request = service.parse_start_payload(
            {
                "jobId": "remote-default-override",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "localLlmMode": "suggestion-only",
                "localLlmModel": "another-deployment",
                "llmProvider": "enterprise-relay",
                "llmEndpoint": "https://relay.example/v1",
                "endpointPolicy": "remote-explicit",
                "llmApiKeyEnv": "RELAY_API_KEY",
            }
        )
    finally:
        service.shutdown()
    assert request.local_llm_model == "another-deployment"
    assert request.llm_allow_model_override is True
    assert request.llm_provider_config.allow_model_override is True


def test_service_remote_profile_honors_explicit_override_opt_out(
    tmp_path: Path,
) -> None:
    config = LLMProviderConfig(
        provider="enterprise-relay",
        model="configured-deployment",
        endpoint="https://relay.example/v1",
        api_key_env="RELAY_API_KEY",
        require_api_key=True,
    )
    service = _provider_service(tmp_path, config=config)
    try:
        with pytest.raises(WorkerError, match="match the configured production model"):
            service.parse_start_payload(
                {
                    "jobId": "remote-explicit-lock",
                    "sourcePath": "source.wav",
                    "outputDirectory": "job",
                    "speakerCountMode": "manual",
                    "speakerCount": 1,
                    "localLlmMode": "suggestion-only",
                    "localLlmModel": "another-deployment",
                    "llmProvider": "enterprise-relay",
                    "llmEndpoint": "https://relay.example/v1",
                    "endpointPolicy": "remote-explicit",
                    "llmApiKeyEnv": "RELAY_API_KEY",
                    "llmAllowModelOverride": False,
                }
            )
    finally:
        service.shutdown()


def test_service_requires_key_for_arbitrary_remote_relay_when_env_is_named(
    tmp_path: Path,
) -> None:
    config = LLMProviderConfig(
        provider="relay-default",
        model="deployment-a",
        endpoint="https://relay.example/v1",
        api_key_env=None,
        require_api_key=False,
        allow_model_override=True,
    )
    service = _provider_service(tmp_path, config=config)
    try:
        request = service.parse_start_payload(
            {
                "jobId": "remote-key-default",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "localLlmModel": "deployment-b",
                "llmProvider": "relay-default",
                "llmEndpoint": "https://relay.example/v1",
                "endpointPolicy": "remote-explicit",
                "llmApiKeyEnv": "RELAY_API_KEY",
                "llmAllowModelOverride": True,
            }
        )
        assert request.llm_require_api_key is True

        explicit_optional = service.parse_start_payload(
            {
                "jobId": "remote-key-optional",
                "sourcePath": "source.wav",
                "outputDirectory": "job-optional",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "localLlmModel": "deployment-b",
                "llmProvider": "relay-default",
                "llmEndpoint": "https://relay.example/v1",
                "endpointPolicy": "remote-explicit",
                "llmApiKeyEnv": "RELAY_API_KEY",
                "llmRequireApiKey": False,
                "llmAllowModelOverride": True,
            }
        )
        assert explicit_optional.llm_require_api_key is False
    finally:
        service.shutdown()


def test_service_does_not_allow_request_to_broaden_offline_network_policy(
    tmp_path: Path,
) -> None:
    config = LLMProviderConfig(
        provider="ollama-loopback",
        model="qwen3.5:27b-q4_K_M",
        endpoint="http://127.0.0.1:11434",
        network_policy="loopback-only",
        api_key_env=None,
        offline_only=True,
    )
    service = _provider_service(tmp_path, config=config)
    try:
        with pytest.raises(ValueError, match="loopback-only"):
            # The provider helper itself also rejects an inconsistent local
            # declaration before a request can be constructed.
            create_llm_provider(
                LLMProviderConfig(
                    provider="ollama-loopback",
                    model=config.model,
                    endpoint=config.endpoint,
                    network_policy="remote-explicit",
                    api_key_env=None,
                )
            )
        with pytest.raises(WorkerError, match="broaden"):
            service.parse_start_payload(
                {
                    "jobId": "remote-not-authorized",
                    "sourcePath": "source.wav",
                    "outputDirectory": "job",
                    "speakerCountMode": "manual",
                    "speakerCount": 1,
                    "localLlmModel": config.model,
                    "llmProvider": "openai-compatible",
                    "llmEndpoint": "https://relay.example/v1",
                    "endpointPolicy": "remote-explicit",
                }
            )
    finally:
        service.shutdown()
