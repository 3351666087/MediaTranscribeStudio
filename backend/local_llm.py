"""Strict, offline-first local language-model boundary.

The production worker deliberately treats a local model as an untrusted
semantic or business-output provider. A semantic model may rank hash-bound
candidate IDs spanning speech disposition, complete speaker timelines,
speaker assignments, language spans, and ASR text, or request bounded
challenger generation. It cannot mutate acoustic evidence, candidate identity,
human locks, or the immutable raw transcript.

The default provider is Ollama-compatible and is restricted to loopback
addresses.  This module uses only the Python standard library so that the
worker remains installable in the ``media-asr`` environment without adding a
network client dependency.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .errors import JobCancelled


class LocalLLMError(RuntimeError):
    """Raised when a local provider cannot produce a valid JSON response."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class LocalLLMContextWindowError(LocalLLMError):
    """Raised before transport when a request cannot fit the model context."""


class LocalLLMProvider(Protocol):
    """Minimal provider contract used by the business-processing layer."""

    provider_id: str
    provider_version: str
    network_policy: str

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
        cancellation_check: Any = None,
    ) -> Mapping[str, Any]:
        """Generate one strict JSON object and return it as a mapping."""

    def release_resources(self) -> None:
        """Release provider-owned resources when its configured scope ends."""


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configuration for an explicitly local provider."""

    model: str = "qwen3.5:27b-q4_K_M"
    endpoint: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 300.0
    temperature: float = 0.0
    top_p: float = 0.1
    context_tokens: int = 8192
    output_tokens: int = 1024
    keep_alive: str = "10m"
    release_on_close: bool = False
    offline_only: bool = True
    expected_model_digest: str | None = None

    def __post_init__(self) -> None:
        model = self.model.strip()
        endpoint = self.endpoint.strip()
        if not model:
            raise ValueError("local LLM model must not be empty")
        if not endpoint:
            raise ValueError("local LLM endpoint must not be empty")
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("local LLM timeout_seconds must be finite and positive")
        if self.temperature < 0 or not math.isfinite(self.temperature):
            raise ValueError("local LLM temperature must be finite and non-negative")
        if (
            self.top_p <= 0
            or self.top_p > 1
            or not math.isfinite(self.top_p)
        ):
            raise ValueError("local LLM top_p must be finite and in (0, 1]")
        if self.context_tokens < 1024 or self.context_tokens > 262_144:
            raise ValueError(
                "local LLM context_tokens must be between 1024 and 262144"
            )
        if self.output_tokens < 128 or self.output_tokens > self.context_tokens:
            raise ValueError(
                "local LLM output_tokens must be between 128 and context_tokens"
            )
        keep_alive = self.keep_alive.strip()
        if re.fullmatch(r"(?:0|-1|\d+(?:ms|s|m|h))", keep_alive) is None:
            raise ValueError(
                "local LLM keep_alive must be 0, -1, or a bounded duration"
            )
        if not isinstance(self.release_on_close, bool):
            raise ValueError("local LLM release_on_close must be boolean")
        if self.offline_only is not True:
            raise ValueError(
                "offline_only must remain enabled for the local LLM boundary"
            )
        expected_model_digest = self.expected_model_digest
        if expected_model_digest is not None:
            if not isinstance(expected_model_digest, str):
                raise ValueError(
                    "local LLM expected_model_digest must be SHA-256 text"
                )
            expected_model_digest = expected_model_digest.strip().casefold()
            if expected_model_digest.startswith("sha256:"):
                expected_model_digest = expected_model_digest.removeprefix(
                    "sha256:"
                )
            if re.fullmatch(r"[0-9a-f]{64}", expected_model_digest) is None:
                raise ValueError(
                    "local LLM expected_model_digest must be a SHA-256 digest"
                )
            expected_model_digest = f"sha256:{expected_model_digest}"
        _assert_loopback_endpoint(endpoint)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "keep_alive", keep_alive)
        object.__setattr__(
            self,
            "expected_model_digest",
            expected_model_digest,
        )


def _assert_loopback_endpoint(endpoint: str) -> None:
    """Fail closed unless the provider endpoint is loopback-only."""

    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("local LLM endpoint must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("local LLM endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("local LLM endpoint must not contain a query or fragment")
    host = (parsed.hostname or "").strip().casefold()
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            "local LLM endpoint must use localhost or a literal loopback address"
        ) from exc
    if not address.is_loopback:
        raise ValueError("local LLM endpoint must resolve to loopback only")


NETWORK_POLICY_LOOPBACK_ONLY = "loopback-only"
NETWORK_POLICY_REMOTE_EXPLICIT = "remote-explicit"
PROVIDER_NETWORK_POLICIES = frozenset(
    {NETWORK_POLICY_LOOPBACK_ONLY, NETWORK_POLICY_REMOTE_EXPLICIT}
)
# Descriptive compatibility name retained for early provider integrations.
NETWORK_POLICY_HTTPS_OR_LOOPBACK = NETWORK_POLICY_REMOTE_EXPLICIT

_NETWORK_POLICY_ALIASES = {
    NETWORK_POLICY_LOOPBACK_ONLY: NETWORK_POLICY_LOOPBACK_ONLY,
    "https-only": NETWORK_POLICY_REMOTE_EXPLICIT,
    "https-required": NETWORK_POLICY_REMOTE_EXPLICIT,
    "https-or-loopback": NETWORK_POLICY_REMOTE_EXPLICIT,
    "remote-allowed": NETWORK_POLICY_REMOTE_EXPLICIT,
    "remote-https": NETWORK_POLICY_REMOTE_EXPLICIT,
    NETWORK_POLICY_REMOTE_EXPLICIT: NETWORK_POLICY_REMOTE_EXPLICIT,
}
_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PROTECTED_REQUEST_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "content-type",
    "host",
    "proxy-authorization",
    "transfer-encoding",
    "x-api-key",
    "api-key",
}


SecretResolver = Callable[[str], str | None]


def _normalize_network_policy(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("LLM network_policy must be text")
    normalized = value.strip().casefold()
    try:
        return _NETWORK_POLICY_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            "LLM network_policy must be loopback-only or remote-explicit"
        ) from exc


def _is_literal_loopback_host(host: str) -> bool:
    normalized = host.strip().casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _assert_endpoint_for_network_policy(
    endpoint: str,
    network_policy: str,
) -> None:
    """Validate one endpoint without resolving or trusting a remote redirect.

    Remote transports must use HTTPS. Plain HTTP remains available for a
    literal loopback address or ``localhost`` so local OpenAI-compatible
    servers such as llama.cpp and LM Studio remain usable.
    """

    policy = _normalize_network_policy(network_policy)
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("LLM endpoint must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("LLM endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM endpoint must not contain a query or fragment")
    host = (parsed.hostname or "").strip().casefold()
    if not host:
        raise ValueError("LLM endpoint must contain a host")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("LLM endpoint contains an invalid port") from exc
    loopback = _is_literal_loopback_host(host)
    if policy == NETWORK_POLICY_LOOPBACK_ONLY:
        if not loopback:
            raise ValueError("LLM endpoint must resolve to loopback only")
        return
    if parsed.scheme == "https" or loopback:
        return
    raise ValueError("remote LLM endpoints must use HTTPS")


def _validate_proxy_url(proxy: str | None) -> str | None:
    if proxy is None:
        return None
    if not isinstance(proxy, str):
        raise ValueError("LLM proxy must be text")
    normalized = proxy.strip()
    if not normalized:
        return None
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LLM proxy must be an absolute http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("LLM proxy credentials must use an external secret store")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM proxy must not contain a query or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("LLM proxy contains an invalid port") from exc
    return normalized


def _normalize_header_name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"LLM {field_name} must be text")
    normalized = value.strip()
    if _HEADER_NAME_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"LLM {field_name} is not a valid HTTP header name")
    return normalized


def _normalize_public_headers(
    headers: Mapping[str, str],
    *,
    auth_header: str,
) -> dict[str, str]:
    if not isinstance(headers, Mapping):
        raise ValueError("LLM headers must be a mapping")
    result: dict[str, str] = {}
    forbidden = set(_PROTECTED_REQUEST_HEADERS)
    forbidden.add(auth_header.casefold())
    for raw_name, raw_value in headers.items():
        name = _normalize_header_name(raw_name, field_name="header name")
        if name.casefold() in forbidden:
            raise ValueError(
                "authentication and transport headers must not contain raw secrets"
            )
        if not isinstance(raw_value, str):
            raise ValueError("LLM header values must be text")
        value = raw_value.strip()
        if not value or "\r" in value or "\n" in value:
            raise ValueError("LLM header values must be non-empty single-line text")
        result[name] = value
    return result


def _normalize_json_path(
    value: Sequence[str | int] | str,
    *,
    field_name: str,
) -> tuple[str | int, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("/"):
            raw_parts = [
                part.replace("~1", "/").replace("~0", "~")
                for part in stripped.split("/")[1:]
            ]
        else:
            raw_parts = stripped.split(".")
        parts: Sequence[str | int] = raw_parts
    elif isinstance(value, Sequence):
        parts = value
    else:
        raise ValueError(f"LLM {field_name} must be a JSON path")
    normalized: list[str | int] = []
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, (str, int)):
            raise ValueError(f"LLM {field_name} contains an invalid path component")
        if isinstance(part, int):
            if part < 0:
                raise ValueError(f"LLM {field_name} array indexes must be non-negative")
            normalized.append(part)
            continue
        token = part.strip()
        if not token:
            raise ValueError(f"LLM {field_name} contains an empty path component")
        normalized.append(int(token) if token.isdecimal() else token)
    return tuple(normalized)


def _reject_embedded_secrets(value: Any, *, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).replace("_", "").replace("-", "").casefold()
            if normalized in {
                "apikey",
                "authorization",
                "accesstoken",
                "proxyauthorization",
            }:
                raise ValueError(
                    f"LLM {field_name} must not contain raw authentication material"
                )
            _reject_embedded_secrets(item, field_name=field_name)
    elif isinstance(value, list):
        for item in value:
            _reject_embedded_secrets(item, field_name=field_name)


@dataclass(frozen=True)
class LLMProviderConfig:
    """Transport-neutral configuration for HTTP JSON model providers.

    Raw API keys are deliberately absent. Authentication material is resolved
    at request time from ``api_key_env`` or ``secret_resolver`` and is never
    included in provider diagnostics or serializable configuration.
    """

    provider: str = "openai-compatible"
    model: str = "gpt-4o-mini"
    endpoint: str = "https://api.openai.com/v1"
    network_policy: str = NETWORK_POLICY_REMOTE_EXPLICIT
    timeout_seconds: float = 300.0
    temperature: float = 0.0
    top_p: float = 0.1
    context_tokens: int = 8_192
    output_tokens: int = 1_024
    api_key_env: str | None = "OPENAI_API_KEY"
    secret_name: str | None = None
    require_api_key: bool = False
    secret_resolver: SecretResolver | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    proxy: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    header_env: Mapping[str, str] = field(default_factory=dict)
    chat_path: str = "/chat/completions"
    response_format: str = "json-schema"
    max_tokens_field: str = "max_tokens"
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    request_template: Mapping[str, Any] | None = None
    response_path: Sequence[str | int] | str = (
        "choices",
        0,
        "message",
        "content",
    )
    finish_reason_path: Sequence[str | int] | str | None = (
        "choices",
        0,
        "finish_reason",
    )
    max_response_bytes: int = 16 * 1024 * 1024
    offline_only: bool = False
    expected_model_digest: str | None = None
    allow_model_override: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str):
            raise ValueError("LLM provider must be text")
        if not isinstance(self.model, str):
            raise ValueError("LLM model must be text")
        if not isinstance(self.endpoint, str):
            raise ValueError("LLM endpoint must be text")
        provider = self.provider.strip().casefold()
        model = self.model.strip()
        endpoint = self.endpoint.strip()
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", provider) is None:
            raise ValueError("LLM provider must be a stable provider identifier")
        if not model:
            raise ValueError("LLM model must not be empty")
        if not endpoint:
            raise ValueError("LLM endpoint must not be empty")
        policy = _normalize_network_policy(self.network_policy)
        if not isinstance(self.offline_only, bool):
            raise ValueError("LLM offline_only must be boolean")
        if self.offline_only and policy != NETWORK_POLICY_LOOPBACK_ONLY:
            raise ValueError("offline_only LLM providers must be loopback-only")
        _assert_endpoint_for_network_policy(endpoint, policy)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
            or not math.isfinite(self.timeout_seconds)
        ):
            raise ValueError("LLM timeout_seconds must be finite and positive")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or self.temperature < 0
            or not math.isfinite(self.temperature)
        ):
            raise ValueError("LLM temperature must be finite and non-negative")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or self.top_p <= 0
            or self.top_p > 1
            or not math.isfinite(self.top_p)
        ):
            raise ValueError("LLM top_p must be finite and in (0, 1]")
        if (
            isinstance(self.context_tokens, bool)
            or not isinstance(self.context_tokens, int)
            or self.context_tokens < 256
            or self.context_tokens > 1_048_576
        ):
            raise ValueError("LLM context_tokens must be between 256 and 1048576")
        if (
            isinstance(self.output_tokens, bool)
            or not isinstance(self.output_tokens, int)
            or self.output_tokens < 1
            or self.output_tokens > self.context_tokens
        ):
            raise ValueError("LLM output_tokens must be between 1 and context_tokens")
        if (
            isinstance(self.max_response_bytes, bool)
            or not isinstance(self.max_response_bytes, int)
            or self.max_response_bytes < 1_024
            or self.max_response_bytes > 64 * 1024 * 1024
        ):
            raise ValueError("LLM max_response_bytes must be between 1024 and 67108864")

        api_key_env = self.api_key_env
        if api_key_env is not None:
            if not isinstance(api_key_env, str):
                raise ValueError("LLM api_key_env must be text")
            api_key_env = api_key_env.strip()
            if not api_key_env:
                api_key_env = None
            elif _ENVIRONMENT_NAME_PATTERN.fullmatch(api_key_env) is None:
                raise ValueError("LLM api_key_env must be an environment variable name")
        secret_name = self.secret_name
        if secret_name is not None:
            if not isinstance(secret_name, str):
                raise ValueError("LLM secret_name must be text")
            secret_name = secret_name.strip()
            if not secret_name or len(secret_name) > 240:
                raise ValueError("LLM secret_name must be non-empty bounded text")
        if self.secret_resolver is not None and not callable(self.secret_resolver):
            raise ValueError("LLM secret_resolver must be callable")
        if not isinstance(self.require_api_key, bool):
            raise ValueError("LLM require_api_key must be boolean")

        auth_header = _normalize_header_name(
            self.auth_header,
            field_name="auth_header",
        )
        if not isinstance(self.auth_scheme, str):
            raise ValueError("LLM auth_scheme must be text")
        auth_scheme = self.auth_scheme.strip()
        if "\r" in auth_scheme or "\n" in auth_scheme:
            raise ValueError("LLM auth_scheme must be single-line text")
        headers = _normalize_public_headers(
            self.headers,
            auth_header=auth_header,
        )
        if not isinstance(self.header_env, Mapping):
            raise ValueError("LLM header_env must be a mapping")
        header_env: dict[str, str] = {}
        for raw_header, raw_environment_name in self.header_env.items():
            header_name = _normalize_header_name(
                raw_header,
                field_name="header_env name",
            )
            if header_name.casefold() in {
                "connection",
                "content-length",
                "content-type",
                "host",
                "proxy-authorization",
                "transfer-encoding",
            }:
                raise ValueError("LLM header_env contains a protected transport header")
            if not isinstance(raw_environment_name, str):
                raise ValueError("LLM header_env values must be environment names")
            environment_name = raw_environment_name.strip()
            if _ENVIRONMENT_NAME_PATTERN.fullmatch(environment_name) is None:
                raise ValueError("LLM header_env values must be environment names")
            if header_name.casefold() in {name.casefold() for name in headers}:
                raise ValueError("LLM headers and header_env must not overlap")
            if header_name.casefold() == auth_header.casefold():
                raise ValueError(
                    "LLM header_env must not override the API key header"
                )
            header_env[header_name] = environment_name
        proxy = _validate_proxy_url(self.proxy)

        expected_model_digest = self.expected_model_digest
        if expected_model_digest is not None:
            if not isinstance(expected_model_digest, str):
                raise ValueError("LLM expected_model_digest must be SHA-256 text")
            expected_model_digest = expected_model_digest.strip().casefold()
            if expected_model_digest.startswith("sha256:"):
                expected_model_digest = expected_model_digest.removeprefix("sha256:")
            if re.fullmatch(r"[0-9a-f]{64}", expected_model_digest) is None:
                raise ValueError("LLM expected_model_digest must be a SHA-256 digest")
            expected_model_digest = f"sha256:{expected_model_digest}"
        if not isinstance(self.allow_model_override, bool):
            raise ValueError("LLM allow_model_override must be boolean")

        if not isinstance(self.chat_path, str):
            raise ValueError("LLM chat_path must be text")
        chat_path = self.chat_path.strip()
        if chat_path:
            parsed_path = urllib.parse.urlparse(chat_path)
            if (
                parsed_path.scheme
                or parsed_path.netloc
                or parsed_path.query
                or parsed_path.fragment
                or not chat_path.startswith("/")
            ):
                raise ValueError("LLM chat_path must be an absolute URL path")
        response_format = self.response_format.strip().casefold()
        if response_format not in {"json-schema", "json-object", "none"}:
            raise ValueError(
                "LLM response_format must be json-schema, json-object, or none"
            )
        max_tokens_field = self.max_tokens_field.strip()
        if max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "LLM max_tokens_field must be max_tokens or max_completion_tokens"
            )

        extra_body = parse_strict_json_object(self.extra_body)
        _reject_embedded_secrets(extra_body, field_name="extra_body")
        request_template = self.request_template
        if request_template is not None:
            request_template = parse_strict_json_object(request_template)
            _reject_embedded_secrets(
                request_template,
                field_name="request_template",
            )
        response_path = _normalize_json_path(
            self.response_path,
            field_name="response_path",
        )
        finish_reason_path = self.finish_reason_path
        if finish_reason_path is not None:
            finish_reason_path = _normalize_json_path(
                finish_reason_path,
                field_name="finish_reason_path",
            )

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "network_policy", policy)
        object.__setattr__(self, "api_key_env", api_key_env)
        object.__setattr__(self, "secret_name", secret_name)
        object.__setattr__(self, "auth_header", auth_header)
        object.__setattr__(self, "auth_scheme", auth_scheme)
        object.__setattr__(self, "proxy", proxy)
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "header_env", header_env)
        object.__setattr__(self, "chat_path", chat_path)
        object.__setattr__(self, "response_format", response_format)
        object.__setattr__(self, "max_tokens_field", max_tokens_field)
        object.__setattr__(self, "extra_body", extra_body)
        object.__setattr__(self, "request_template", request_template)
        object.__setattr__(self, "response_path", response_path)
        object.__setattr__(self, "finish_reason_path", finish_reason_path)
        object.__setattr__(
            self,
            "expected_model_digest",
            expected_model_digest,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LLMProviderConfig":
        """Build a config from JSON-shaped settings while rejecting raw keys."""

        if not isinstance(raw, Mapping):
            raise ValueError("LLM provider configuration must be a mapping")
        aliases = {
            "providerId": "provider",
            "provider_id": "provider",
            "networkPolicy": "network_policy",
            "timeoutSeconds": "timeout_seconds",
            "topP": "top_p",
            "contextTokens": "context_tokens",
            "outputTokens": "output_tokens",
            "apiKeyEnv": "api_key_env",
            "secretName": "secret_name",
            "requireApiKey": "require_api_key",
            "authHeader": "auth_header",
            "authScheme": "auth_scheme",
            "proxyUrl": "proxy",
            "proxy_url": "proxy",
            "headerEnv": "header_env",
            "chatPath": "chat_path",
            "responseFormat": "response_format",
            "maxTokensField": "max_tokens_field",
            "extraBody": "extra_body",
            "requestTemplate": "request_template",
            "responsePath": "response_path",
            "finishReasonPath": "finish_reason_path",
            "maxResponseBytes": "max_response_bytes",
            "offlineOnly": "offline_only",
            "expectedDigest": "expected_model_digest",
            "expectedModelDigest": "expected_model_digest",
            "allowModelOverride": "allow_model_override",
        }
        if any(
            str(key).replace("_", "").casefold() == "apikey"
            for key in raw
        ):
            raise ValueError(
                "raw API keys are forbidden; use api_key_env or secret_resolver"
            )
        allowed = set(cls.__dataclass_fields__) - {"secret_resolver"}
        normalized: dict[str, Any] = {}
        for raw_key, value in raw.items():
            if not isinstance(raw_key, str):
                raise ValueError("LLM provider configuration keys must be text")
            key = aliases.get(raw_key, raw_key)
            if key not in allowed:
                raise ValueError(f"unsupported LLM provider configuration field: {raw_key}")
            if key in normalized:
                raise ValueError(f"duplicate LLM provider configuration field: {key}")
            normalized[key] = value
        return cls(**normalized)

    @property
    def proxy_url(self) -> str | None:
        """Compatibility alias for configuration surfaces using proxyUrl."""

        return self.proxy


@dataclass(frozen=True)
class OpenAICompatibleConfig(LLMProviderConfig):
    """OpenAI chat-completions defaults with a key resolved from the environment."""


@dataclass(frozen=True)
class HuggingFaceRouterConfig(LLMProviderConfig):
    """OpenAI-compatible Hugging Face Inference Router defaults."""

    provider: str = "huggingface-router"
    model: str = "Qwen/Qwen3-32B"
    endpoint: str = "https://router.huggingface.co/v1"
    api_key_env: str | None = "HF_TOKEN"
    require_api_key: bool = True


@dataclass(frozen=True)
class AnthropicProviderConfig(LLMProviderConfig):
    """Native Anthropic Messages API defaults."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    endpoint: str = "https://api.anthropic.com/v1"
    api_key_env: str | None = "ANTHROPIC_API_KEY"
    require_api_key: bool = True
    auth_header: str = "x-api-key"
    auth_scheme: str = ""
    headers: Mapping[str, str] = field(
        default_factory=lambda: {"anthropic-version": "2023-06-01"}
    )
    chat_path: str = "/messages"
    response_path: Sequence[str | int] | str = ("content",)
    finish_reason_path: Sequence[str | int] | str | None = ("stop_reason",)


@dataclass(frozen=True)
class GoogleGeminiProviderConfig(LLMProviderConfig):
    """Native Google Gemini generateContent API defaults."""

    provider: str = "google-gemini"
    model: str = "gemini-2.5-pro"
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str | None = "GEMINI_API_KEY"
    require_api_key: bool = True
    auth_header: str = "x-goog-api-key"
    auth_scheme: str = ""
    chat_path: str = ""
    response_path: Sequence[str | int] | str = (
        "candidates",
        0,
        "content",
        "parts",
    )
    finish_reason_path: Sequence[str | int] | str | None = (
        "candidates",
        0,
        "finishReason",
    )


@dataclass(frozen=True)
class CustomJSONProviderConfig(LLMProviderConfig):
    """Conservative defaults for a user-defined HTTP JSON endpoint."""

    provider: str = "custom-http-json"
    model: str = "custom-model"
    endpoint: str = "http://127.0.0.1:8000/v1"
    network_policy: str = NETWORK_POLICY_LOOPBACK_ONLY
    api_key_env: str | None = None


CustomJSONConfig = CustomJSONProviderConfig
AnthropicConfig = AnthropicProviderConfig
GeminiProviderConfig = GoogleGeminiProviderConfig
GoogleGeminiConfig = GoogleGeminiProviderConfig


def assert_loopback_provider(provider: LocalLLMProvider) -> str:
    """Validate the transport declaration used by a business-model provider.

    Endpoint-backed providers must expose either ``config.endpoint`` or
    ``endpoint`` so the boundary can validate the concrete destination.
    Endpoint-less providers are in-process adapters and therefore expose no
    network destination through this contract.  An explicit policy declaration
    always wins and any value other than ``loopback-only`` is rejected.
    """

    policy = getattr(provider, "network_policy", None)
    if policy is not None and policy != "loopback-only":
        raise LocalLLMError(
            "local LLM provider network policy must be loopback-only"
        )

    config = getattr(provider, "config", None)
    endpoint = getattr(config, "endpoint", None)
    if endpoint is None:
        endpoint = getattr(provider, "endpoint", None)
    if endpoint is not None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise LocalLLMError(
                "local LLM provider endpoint must be non-empty text"
            )
        try:
            _assert_loopback_endpoint(endpoint.strip())
        except ValueError as exc:
            raise LocalLLMError(
                "local LLM provider endpoint is not loopback-only"
            ) from exc

    return "loopback-only"


def assert_provider_network_policy(provider: LocalLLMProvider) -> str:
    """Validate a provider declaration and return its provenance policy.

    ``loopback-only`` remains the offline default. ``remote-explicit`` is an
    opt-in transport boundary: the configured destination must be HTTPS unless
    it is a literal loopback endpoint. Redirects are rejected by the concrete
    HTTP providers independently of this declaration check.
    """

    raw_policy = getattr(provider, "network_policy", None)
    if raw_policy is None:
        raw_policy = NETWORK_POLICY_LOOPBACK_ONLY
    try:
        policy = _normalize_network_policy(raw_policy)
    except ValueError as exc:
        raise LocalLLMError("LLM provider network policy is invalid") from exc

    config = getattr(provider, "config", None)
    endpoint = getattr(config, "endpoint", None)
    if endpoint is None:
        endpoint = getattr(provider, "endpoint", None)
    if endpoint is not None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise LocalLLMError("LLM provider endpoint must be non-empty text")
        try:
            _assert_endpoint_for_network_policy(endpoint.strip(), policy)
        except ValueError as exc:
            raise LocalLLMError(
                "LLM provider endpoint violates its declared network policy"
            ) from exc
    elif policy == NETWORK_POLICY_REMOTE_EXPLICIT:
        raise LocalLLMError(
            "remote-explicit LLM providers must expose their endpoint"
        )
    return policy


def parse_strict_json_object(raw: Any) -> dict[str, Any]:
    """Parse provider output with duplicate-key and non-finite-number checks."""

    if isinstance(raw, Mapping):
        value: Any = dict(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```") or text.endswith("```"):
            raise LocalLLMError("fenced JSON is forbidden in local LLM output")

        def reject_constant(token: str) -> Any:
            raise LocalLLMError(f"non-finite JSON number is forbidden: {token}")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise LocalLLMError(f"duplicate JSON object key: {key}")
                result[key] = item
            return result

        try:
            value = json.loads(
                text,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
        except LocalLLMError:
            raise
        except (TypeError, json.JSONDecodeError) as exc:
            raise LocalLLMError("local LLM output is not valid JSON") from exc
    else:
        raise LocalLLMError("local LLM output must be a JSON object or JSON string")

    _validate_json_value(value, "$")
    if not isinstance(value, dict):
        raise LocalLLMError("local LLM output root must be an object")
    return value


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalLLMError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LocalLLMError(f"{path} contains a non-string object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise LocalLLMError(f"{path} contains unsupported value type {type(value).__name__}")


def _check_cancelled(cancellation_check: Any) -> None:
    if cancellation_check is None:
        return
    if callable(cancellation_check):
        cancellation_check()
        return
    if getattr(cancellation_check, "is_set", lambda: False)():
        raise JobCancelled()


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect instead of trusting a provider-supplied target."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        status_code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        del request, file_pointer, message, headers, new_url
        raise LocalLLMError(
            f"local LLM redirects are forbidden (HTTP {status_code})"
        )


def _assert_loopback_response_url(response: Any, requested_url: str) -> None:
    """Ensure the transport did not silently escape the loopback boundary."""

    get_url = getattr(response, "geturl", None)
    final_url = get_url() if callable(get_url) else requested_url
    if not isinstance(final_url, str) or not final_url:
        raise LocalLLMError("local LLM response URL is unavailable")
    try:
        _assert_loopback_endpoint(final_url)
    except ValueError as exc:
        raise LocalLLMError(
            "local LLM response escaped the loopback boundary"
        ) from exc


def estimate_input_tokens(*values: str) -> int:
    """Return the conservative token estimate shared by planning and preflight."""

    encoded_size = sum(len(value.encode("utf-8")) for value in values)
    return max(1, math.ceil(encoded_size / 3)) + 64


def _validate_response_against_schema(
    value: Mapping[str, Any],
    *,
    validator: Draft202012Validator | None,
) -> None:
    if validator is None:
        return
    errors = sorted(
        validator.iter_errors(dict(value)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    location = "$"
    for part in error.absolute_path:
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    decisions = value.get("decisions")
    translations = value.get("translations")
    raise LocalLLMError(
        f"local LLM output failed response schema at {location}: {error.message}",
        diagnostics={
            "failureStage": "response-schema",
            "responseFields": sorted(str(key) for key in value),
            "decisionCount": (
                len(decisions) if isinstance(decisions, list) else None
            ),
            "translationCount": (
                len(translations) if isinstance(translations, list) else None
            ),
            "schemaErrorPath": location,
            "schemaValidator": str(error.validator),
            "responseContentPersisted": False,
        },
    )


def _assert_complete_generation(
    envelope: Mapping[str, Any],
    *,
    output_token_limit: int,
) -> None:
    done = envelope.get("done")
    if "done" in envelope and not isinstance(done, bool):
        raise LocalLLMError("local LLM provider returned an invalid done flag")
    if done is False:
        raise LocalLLMError("local LLM provider returned an incomplete generation")
    done_reason = envelope.get("done_reason")
    if "done_reason" in envelope and not isinstance(done_reason, str):
        raise LocalLLMError("local LLM provider returned an invalid done reason")
    if isinstance(done_reason, str) and done_reason.strip().casefold() in {
        "length",
        "limit",
        "max_length",
        "max_tokens",
        "token_limit",
    }:
        raise LocalLLMError("local LLM provider truncated the structured response")
    eval_count = envelope.get("eval_count")
    if "eval_count" in envelope and (
        isinstance(eval_count, bool)
        or not isinstance(eval_count, int)
        or eval_count < 0
    ):
        raise LocalLLMError("local LLM provider returned an invalid eval count")
    if (
        isinstance(eval_count, int)
        and not isinstance(eval_count, bool)
        and eval_count >= output_token_limit
    ):
        raise LocalLLMError(
            "local LLM provider exhausted the structured-output token budget"
        )


def _http_error_diagnostics(exc: urllib.error.HTTPError) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "failureStage": "transport-http",
        "httpStatus": int(exc.code),
        "responseContentPersisted": False,
    }
    try:
        body = exc.read(16_384).decode("utf-8", errors="strict")
        envelope = parse_strict_json_object(body)
        error: Any = envelope.get("error")
        if isinstance(error, str):
            try:
                error = parse_strict_json_object(error)
            except LocalLLMError:
                error = None
        if isinstance(error, Mapping) and isinstance(
            error.get("error"),
            Mapping,
        ):
            error = error["error"]
        if isinstance(error, Mapping):
            code = error.get("code")
            if (
                not isinstance(code, bool)
                and isinstance(code, int)
                and code >= 0
            ):
                diagnostics["providerErrorCode"] = code
            error_type = error.get("type")
            if isinstance(error_type, str) and error_type:
                diagnostics["providerErrorType"] = error_type[:100]
            message = error.get("message")
            if isinstance(message, str):
                normalized = message.casefold()
                if (
                    "initialize samplers" in normalized
                    and "parse grammar" in normalized
                ):
                    diagnostics["providerErrorCategory"] = (
                        "schema-grammar-initialization"
                    )
    except (LocalLLMError, UnicodeError, OSError):
        pass
    return diagnostics


class OllamaLocalProvider:
    """Ollama-compatible provider restricted to a loopback endpoint."""

    provider_id = "ollama-loopback"
    network_policy = "loopback-only"
    # Cache/provenance contract v3 covers bounded multi-segment batching and
    # hierarchical evidence-grounded summarization. Reliability hardening in
    # this module preserves that public provider contract; the business-layer
    # execution revision independently invalidates unsafe checkpoints.
    provider_version = "native-json-v3"
    # Business tasks may opt into bounded batching/hierarchical summarization.
    # These are capability hints, not trust signals; every response still goes
    # through the immutable identity and public-contract validators.
    business_batch_size = 8
    business_batch_character_limit = 7_000
    business_translation_segment_attempts = 3
    business_generation_attempts = 3
    business_summary_segment_limit = 40
    business_summary_character_limit = 8_000
    business_summary_reduce_size = 8

    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = config or LocalLLMConfig()
        # Input-only limits previously allowed a 7k-character translation
        # batch to compete for a 1k-token output budget. That can truncate the
        # structured response while still leaving syntactically plausible
        # partial content. Keep the immutable-segment batch bounded by both
        # provider output capacity and the hard business-layer ceiling.
        self.business_batch_character_limit = max(
            256,
            min(
                type(self).business_batch_character_limit,
                self.config.output_tokens // 2,
            ),
        )
        available_input_tokens = max(
            128,
            self.config.context_tokens - self.config.output_tokens - 256,
        )
        self.business_summary_character_limit = max(
            256,
            min(
                type(self).business_summary_character_limit,
                available_input_tokens * 2,
            ),
        )
        self._opener = urllib.request.build_opener(_RejectRedirectHandler())
        self._release_lock = threading.Lock()
        self._released = False
        self._model_digest_lock = threading.Lock()
        self._model_digest_verified = False
        self._generation_metrics = {
            "completedCalls": 0,
            "totalDurationNanoseconds": 0,
            "loadDurationNanoseconds": 0,
            "promptEvalTokens": 0,
            "promptEvalDurationNanoseconds": 0,
            "outputTokens": 0,
            "outputEvalDurationNanoseconds": 0,
        }

    @property
    def generation_metrics(self) -> dict[str, int]:
        """Return aggregate provider timings without exposing response content."""

        return dict(self._generation_metrics)

    def _record_generation_metrics(self, envelope: Mapping[str, Any]) -> None:
        self._generation_metrics["completedCalls"] += 1
        fields = {
            "total_duration": "totalDurationNanoseconds",
            "load_duration": "loadDurationNanoseconds",
            "prompt_eval_count": "promptEvalTokens",
            "prompt_eval_duration": "promptEvalDurationNanoseconds",
            "eval_count": "outputTokens",
            "eval_duration": "outputEvalDurationNanoseconds",
        }
        for source, target in fields.items():
            value = envelope.get(source)
            if (
                not isinstance(value, bool)
                and isinstance(value, int)
                and value >= 0
            ):
                self._generation_metrics[target] += value

    def _verify_expected_model_digest(self) -> None:
        expected = self.config.expected_model_digest
        if expected is None or self._model_digest_verified:
            return
        with self._model_digest_lock:
            if self._model_digest_verified:
                return
            endpoint = self.config.endpoint.rstrip("/") + "/api/tags"
            try:
                _assert_loopback_endpoint(endpoint)
            except ValueError as exc:
                raise LocalLLMError(
                    "local LLM model inventory endpoint is not loopback-only"
                ) from exc
            request = urllib.request.Request(
                endpoint,
                headers={"Accept": "application/json"},
                method="GET",
            )
            try:
                with self._opener.open(
                    request,
                    timeout=min(self.config.timeout_seconds, 30.0),
                ) as response:
                    _assert_loopback_response_url(response, endpoint)
                    body = response.read(4 * 1024 * 1024 + 1)
            except LocalLLMError:
                raise
            except (OSError, urllib.error.URLError) as exc:
                raise LocalLLMError(
                    "loopback local LLM model inventory request failed"
                ) from exc
            if len(body) > 4 * 1024 * 1024:
                raise LocalLLMError(
                    "local LLM model inventory exceeded the response limit"
                )
            try:
                inventory = parse_strict_json_object(
                    body.decode("utf-8", errors="strict")
                )
            except (UnicodeError, LocalLLMError) as exc:
                raise LocalLLMError(
                    "local LLM model inventory is invalid"
                ) from exc
            raw_models = inventory.get("models")
            if not isinstance(raw_models, list):
                raise LocalLLMError(
                    "local LLM model inventory has no models list"
                )
            actual: str | None = None
            for raw_model in raw_models:
                if not isinstance(raw_model, Mapping):
                    continue
                names = (raw_model.get("name"), raw_model.get("model"))
                if self.config.model not in names:
                    continue
                raw_digest = raw_model.get("digest")
                if not isinstance(raw_digest, str):
                    break
                normalized = raw_digest.strip().casefold()
                if not normalized.startswith("sha256:"):
                    normalized = f"sha256:{normalized}"
                if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized) is None:
                    break
                actual = normalized
                break
            if actual is None:
                raise LocalLLMError(
                    "configured local LLM model is absent from the local inventory",
                    diagnostics={
                        "failureStage": "model-digest",
                        "model": self.config.model,
                    },
                )
            if actual != expected:
                raise LocalLLMError(
                    "configured local LLM model digest does not match",
                    diagnostics={
                        "failureStage": "model-digest",
                        "model": self.config.model,
                        "expectedDigest": expected,
                        "actualDigest": actual,
                    },
                )
            self._model_digest_verified = True

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
        cancellation_check: Any = None,
    ) -> Mapping[str, Any]:
        _check_cancelled(cancellation_check)
        with self._release_lock:
            self._released = False
        selected_model = model.strip() or self.config.model
        if selected_model != self.config.model:
            raise LocalLLMError(
                "requested local LLM model does not match the configured model"
            )
        if temperature < 0 or not math.isfinite(temperature):
            raise LocalLLMError("temperature must be finite and non-negative")
        if response_schema is not None:
            try:
                schema = parse_strict_json_object(response_schema)
                Draft202012Validator.check_schema(schema)
                schema_validator: Draft202012Validator | None = (
                    Draft202012Validator(schema)
                )
            except (LocalLLMError, SchemaError) as exc:
                raise LocalLLMError("response schema must be strict JSON") from exc
        else:
            schema = None
            schema_validator = None
        schema_text = (
            json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            if schema is not None
            else ""
        )
        estimated_input_tokens = estimate_input_tokens(
            system_prompt,
            user_prompt,
            schema_text,
        )
        input_token_budget = self.config.context_tokens - self.config.output_tokens
        if estimated_input_tokens > input_token_budget:
            raise LocalLLMContextWindowError(
                "local LLM request exceeds the configured context window"
            )
        self._verify_expected_model_digest()
        payload = {
            "model": selected_model,
            "stream": False,
            "format": schema or "json",
            # Qwen3-family models otherwise may consume the entire output budget
            # in ``message.thinking`` and return an empty JSON content field.
            # Business processing needs deterministic, contract-bound output,
            # not hidden chain-of-thought.
            "think": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": float(temperature),
                "top_p": self.config.top_p,
                "num_ctx": self.config.context_tokens,
                "num_predict": self.config.output_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        endpoint = self.config.endpoint.rstrip("/") + "/api/chat"
        try:
            _assert_loopback_endpoint(endpoint)
        except ValueError as exc:
            raise LocalLLMError(
                "local LLM request endpoint is not loopback-only"
            ) from exc
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener.open(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                _assert_loopback_response_url(response, endpoint)
                body = response.read().decode("utf-8", errors="strict")
        except LocalLLMError:
            raise
        except urllib.error.HTTPError as exc:
            raise LocalLLMError(
                "loopback local LLM request was rejected",
                diagnostics=_http_error_diagnostics(exc),
            ) from exc
        except (OSError, urllib.error.URLError, UnicodeError) as exc:
            raise LocalLLMError("loopback local LLM request failed") from exc
        _check_cancelled(cancellation_check)
        try:
            envelope = parse_strict_json_object(body)
        except LocalLLMError as exc:
            raise LocalLLMError("local LLM provider returned malformed JSON") from exc
        _assert_complete_generation(
            envelope,
            output_token_limit=self.config.output_tokens,
        )
        message = envelope.get("message")
        content: Any
        if isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = envelope.get("response")
        if not isinstance(content, (str, Mapping)) or (
            isinstance(content, str) and not content.strip()
        ):
            raise LocalLLMError(
                "local LLM provider returned empty structured content"
            )
        result = parse_strict_json_object(content)
        _validate_response_against_schema(
            result,
            validator=schema_validator,
        )
        self._record_generation_metrics(envelope)
        return result

    def release_resources(self) -> None:
        """Explicitly unload a stage-scoped model through the loopback API."""

        if not self.config.release_on_close:
            return
        with self._release_lock:
            if self._released:
                return
            with self._model_digest_lock:
                self._model_digest_verified = False
            endpoint = self.config.endpoint.rstrip("/") + "/api/generate"
            try:
                _assert_loopback_endpoint(endpoint)
            except ValueError as exc:
                raise LocalLLMError(
                    "local LLM release endpoint is not loopback-only"
                ) from exc
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(
                    {
                        "model": self.config.model,
                        "keep_alive": 0,
                        "stream": False,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with self._opener.open(
                    request,
                    timeout=min(self.config.timeout_seconds, 30.0),
                ) as response:
                    _assert_loopback_response_url(response, endpoint)
                    body = response.read(1024 * 1024 + 1)
            except LocalLLMError:
                raise
            except (OSError, urllib.error.URLError) as exc:
                raise LocalLLMError(
                    "loopback local LLM resource release failed"
                ) from exc
            if len(body) > 1024 * 1024:
                raise LocalLLMError(
                    "local LLM resource release response exceeded the limit"
                )
            try:
                acknowledgement = parse_strict_json_object(
                    body.decode("utf-8", errors="strict")
                )
            except (UnicodeError, LocalLLMError) as exc:
                raise LocalLLMError(
                    "local LLM resource release acknowledgement is invalid"
                ) from exc
            if acknowledgement.get("done") is not True or "error" in acknowledgement:
                raise LocalLLMError(
                    "local LLM resource release was not acknowledged"
                )
            self._released = True


def _prepare_response_schema(
    response_schema: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, Draft202012Validator | None, str]:
    if response_schema is None:
        return None, None, ""
    try:
        schema = parse_strict_json_object(response_schema)
        Draft202012Validator.check_schema(schema)
    except (LocalLLMError, SchemaError) as exc:
        raise LocalLLMError("response schema must be strict JSON") from exc
    return (
        schema,
        Draft202012Validator(schema),
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
    )


def _build_http_json_opener(proxy: str | None) -> Any:
    proxies = {"http": proxy, "https": proxy} if proxy else {}
    return urllib.request.build_opener(
        urllib.request.ProxyHandler(proxies),
        _RejectRedirectHandler(),
    )


def _provider_request_url(config: LLMProviderConfig) -> str:
    endpoint = config.endpoint.rstrip("/")
    path = config.chat_path
    if not path:
        return endpoint
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.path.rstrip("/").endswith(path.rstrip("/")):
        return endpoint
    return endpoint + path


def _assert_provider_response_url(
    response: Any,
    *,
    requested_url: str,
    network_policy: str,
) -> None:
    get_url = getattr(response, "geturl", None)
    final_url = get_url() if callable(get_url) else requested_url
    if not isinstance(final_url, str) or not final_url:
        raise LocalLLMError("LLM provider response URL is unavailable")
    try:
        _assert_endpoint_for_network_policy(final_url, network_policy)
    except ValueError as exc:
        raise LocalLLMError(
            "LLM provider response violates its network policy"
        ) from exc
    if final_url.rstrip("/") != requested_url.rstrip("/"):
        raise LocalLLMError("LLM provider response URL changed unexpectedly")


def _read_bounded_response(response: Any, *, limit: int) -> bytes:
    try:
        body = response.read(limit + 1)
    except TypeError:
        body = response.read()
    if not isinstance(body, bytes):
        raise LocalLLMError("LLM provider response body must be bytes")
    if len(body) > limit:
        raise LocalLLMError("LLM provider response exceeded the configured limit")
    return body


def _extract_json_path(
    value: Any,
    path: Sequence[str | int],
    *,
    field_name: str,
) -> Any:
    current = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                raise LocalLLMError(
                    f"LLM provider response is missing {field_name}"
                )
            current = current[part]
        else:
            if not isinstance(current, Mapping) or part not in current:
                raise LocalLLMError(
                    f"LLM provider response is missing {field_name}"
                )
            current = current[part]
    return current


def _optional_json_path(
    value: Any,
    path: Sequence[str | int] | None,
) -> Any:
    if path is None:
        return None
    try:
        return _extract_json_path(value, path, field_name="completion status")
    except LocalLLMError:
        return None


def _coalesce_structured_content(value: Any) -> str | Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        if not value.strip():
            raise LocalLLMError("LLM provider returned empty structured content")
        return value
    if not isinstance(value, list) or not value:
        raise LocalLLMError("LLM provider returned empty structured content")
    text_parts: list[str] = []
    mapped_content: list[Mapping[str, Any]] = []
    for part in value:
        if isinstance(part, str):
            if part:
                text_parts.append(part)
            continue
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
        content = part.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
        raw_input = part.get("input")
        if isinstance(raw_input, Mapping):
            mapped_content.append(raw_input)
    if text_parts:
        return "".join(text_parts)
    if len(mapped_content) == 1:
        return mapped_content[0]
    raise LocalLLMError("LLM provider returned empty structured content")


_TEMPLATE_EXACT_PATTERN = re.compile(
    r"(?:\$\{([a-zA-Z][a-zA-Z0-9_]*)\}|\{\{([a-zA-Z][a-zA-Z0-9_]*)\}\})"
)


def _render_request_template(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _render_request_template(item, variables)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render_request_template(item, variables) for item in value]
    if not isinstance(value, str):
        return value
    exact = _TEMPLATE_EXACT_PATTERN.fullmatch(value)
    if exact is not None:
        name = exact.group(1) or exact.group(2)
        if name not in variables:
            raise LocalLLMError(f"unknown custom request template variable: {name}")
        return variables[name]

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in variables:
            raise LocalLLMError(f"unknown custom request template variable: {name}")
        replacement = variables[name]
        if isinstance(replacement, (Mapping, list)):
            return json.dumps(
                replacement,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return str(replacement)

    return _TEMPLATE_EXACT_PATTERN.sub(replace, value)


def _provider_error_diagnostics(envelope: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "failureStage": "provider-envelope",
        "responseContentPersisted": False,
    }
    error = envelope.get("error")
    if isinstance(error, Mapping):
        error_type = error.get("type") or error.get("status")
        if isinstance(error_type, str) and error_type:
            diagnostics["providerErrorType"] = error_type[:100]
        code = error.get("code")
        if isinstance(code, (str, int)) and not isinstance(code, bool):
            diagnostics["providerErrorCode"] = str(code)[:100]
    return diagnostics


class HTTPJSONLLMProvider:
    """Strict JSON provider over an explicitly authorized HTTP transport."""

    provider_id = "openai-compatible"
    provider_version = "http-json-v1"
    network_policy = NETWORK_POLICY_REMOTE_EXPLICIT
    business_batch_size = 8
    business_batch_character_limit = 7_000
    business_translation_segment_attempts = 3
    business_generation_attempts = 3
    business_summary_segment_limit = 40
    business_summary_character_limit = 8_000
    business_summary_reduce_size = 8

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        if not isinstance(config, LLMProviderConfig):
            raise TypeError("HTTP JSON providers require LLMProviderConfig")
        self.config = config
        self.provider_id = config.provider
        self.network_policy = config.network_policy
        self._secret_resolver = secret_resolver or config.secret_resolver
        self._opener = _build_http_json_opener(config.proxy)
        self.business_batch_character_limit = max(
            256,
            min(
                type(self).business_batch_character_limit,
                config.output_tokens // 2,
            ),
        )
        available_input_tokens = max(
            128,
            config.context_tokens - config.output_tokens - 256,
        )
        self.business_summary_character_limit = max(
            256,
            min(
                type(self).business_summary_character_limit,
                available_input_tokens * 2,
            ),
        )
        self._metrics_lock = threading.Lock()
        self._generation_metrics = {
            "completedCalls": 0,
            "promptTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
        }

    @property
    def endpoint(self) -> str:
        return self.config.endpoint

    @property
    def generation_metrics(self) -> dict[str, int]:
        with self._metrics_lock:
            return dict(self._generation_metrics)

    def _resolve_api_key(self) -> str | None:
        value: Any = None
        resolver = self._secret_resolver
        resolver_name = (
            self.config.secret_name
            or self.config.api_key_env
            or f"{self.provider_id}.api-key"
        )
        if resolver is not None:
            try:
                value = resolver(resolver_name)
            except Exception as exc:
                raise LocalLLMError(
                    "LLM provider secret resolution failed",
                    diagnostics={
                        "failureStage": "authentication",
                        "responseContentPersisted": False,
                    },
                ) from exc
        if value is None and self.config.api_key_env is not None:
            value = os.environ.get(self.config.api_key_env)
        if value is None:
            if self.config.require_api_key:
                raise LocalLLMError(
                    "LLM provider API key is unavailable",
                    diagnostics={
                        "failureStage": "authentication",
                        "responseContentPersisted": False,
                    },
                )
            return None
        if not isinstance(value, str):
            raise LocalLLMError("LLM provider secret resolver returned non-text")
        normalized = value.strip()
        if not normalized:
            if self.config.require_api_key:
                raise LocalLLMError("LLM provider API key is unavailable")
            return None
        if "\r" in normalized or "\n" in normalized or len(normalized) > 16_384:
            raise LocalLLMError("LLM provider API key is invalid")
        return normalized

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **dict(self.config.headers),
        }
        for header_name, environment_name in self.config.header_env.items():
            value = os.environ.get(environment_name)
            if value is None or not value.strip():
                raise LocalLLMError(
                    "LLM provider configured header secret is unavailable",
                    diagnostics={
                        "failureStage": "authentication",
                        "responseContentPersisted": False,
                    },
                )
            normalized = value.strip()
            if "\r" in normalized or "\n" in normalized:
                raise LocalLLMError("LLM provider configured header secret is invalid")
            headers[header_name] = normalized
        key = self._resolve_api_key()
        if key is not None:
            value = (
                f"{self.config.auth_scheme} {key}"
                if self.config.auth_scheme
                else key
            )
            headers[self.config.auth_header] = value
        return headers

    def _select_model(self, model: str) -> str:
        if not isinstance(model, str):
            raise LocalLLMError("requested LLM model must be text")
        selected = model.strip() or self.config.model
        if not self.config.allow_model_override and selected != self.config.model:
            raise LocalLLMError(
                "requested LLM model does not match the configured model"
            )
        return selected

    def _request_url(self, selected_model: str) -> str:
        del selected_model
        return _provider_request_url(self.config)

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        selected_model: str,
        temperature: float,
        schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": selected_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "top_p": self.config.top_p,
            self.config.max_tokens_field: self.config.output_tokens,
        }
        if self.config.response_format == "json-schema" and schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "mts_structured_response",
                    "strict": True,
                    "schema": dict(schema),
                },
            }
        elif self.config.response_format in {"json-schema", "json-object"}:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.config.extra_body)
        payload["model"] = selected_model
        payload["stream"] = False
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return payload

    def _completion_status(self, envelope: Mapping[str, Any]) -> None:
        reason = _optional_json_path(
            envelope,
            self.config.finish_reason_path,
        )
        if reason is None:
            return
        if not isinstance(reason, str):
            raise LocalLLMError("LLM provider returned an invalid finish reason")
        normalized = reason.strip().casefold()
        if normalized in {
            "length",
            "max_length",
            "max_tokens",
            "max_completion_tokens",
            "token_limit",
        }:
            raise LocalLLMError("LLM provider truncated the structured response")
        if normalized in {"content_filter", "error", "refusal"}:
            raise LocalLLMError("LLM provider declined the structured response")

    def _extract_content(self, envelope: Mapping[str, Any]) -> Any:
        return _extract_json_path(
            envelope,
            self.config.response_path,
            field_name="structured content",
        )

    def _record_usage(self, envelope: Mapping[str, Any]) -> None:
        usage = envelope.get("usage")
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 0
        if isinstance(usage, Mapping):
            values = {
                "prompt_tokens": "prompt",
                "completion_tokens": "output",
                "total_tokens": "total",
            }
            parsed: dict[str, int] = {}
            for source, target in values.items():
                value = usage.get(source)
                if (
                    not isinstance(value, bool)
                    and isinstance(value, int)
                    and value >= 0
                ):
                    parsed[target] = value
            prompt_tokens = parsed.get("prompt", 0)
            output_tokens = parsed.get("output", 0)
            total_tokens = parsed.get(
                "total",
                prompt_tokens + output_tokens,
            )
        with self._metrics_lock:
            self._generation_metrics["completedCalls"] += 1
            self._generation_metrics["promptTokens"] += prompt_tokens
            self._generation_metrics["outputTokens"] += output_tokens
            self._generation_metrics["totalTokens"] += total_tokens

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
        cancellation_check: Any = None,
    ) -> Mapping[str, Any]:
        _check_cancelled(cancellation_check)
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise LocalLLMError("LLM prompts must be text")
        selected_model = self._select_model(model)
        if temperature < 0 or not math.isfinite(temperature):
            raise LocalLLMError("temperature must be finite and non-negative")
        schema, validator, schema_text = _prepare_response_schema(response_schema)
        estimated_input_tokens = estimate_input_tokens(
            system_prompt,
            user_prompt,
            schema_text,
        )
        if estimated_input_tokens > (
            self.config.context_tokens - self.config.output_tokens
        ):
            raise LocalLLMContextWindowError(
                "LLM request exceeds the configured context window"
            )
        payload = self._build_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            selected_model=selected_model,
            temperature=temperature,
            schema=schema,
        )
        payload = parse_strict_json_object(payload)
        endpoint = self._request_url(selected_model)
        try:
            _assert_endpoint_for_network_policy(endpoint, self.network_policy)
        except ValueError as exc:
            raise LocalLLMError(
                "LLM request endpoint violates its network policy"
            ) from exc
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._request_headers(),
            method="POST",
        )
        try:
            with self._opener.open(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                _assert_provider_response_url(
                    response,
                    requested_url=endpoint,
                    network_policy=self.network_policy,
                )
                body = _read_bounded_response(
                    response,
                    limit=self.config.max_response_bytes,
                )
        except LocalLLMError:
            raise
        except urllib.error.HTTPError as exc:
            raise LocalLLMError(
                "LLM provider request was rejected",
                diagnostics=_http_error_diagnostics(exc),
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise LocalLLMError("LLM provider request failed") from exc
        _check_cancelled(cancellation_check)
        try:
            envelope = parse_strict_json_object(
                body.decode("utf-8", errors="strict")
            )
        except (UnicodeError, LocalLLMError) as exc:
            raise LocalLLMError("LLM provider returned malformed JSON") from exc
        if "error" in envelope:
            raise LocalLLMError(
                "LLM provider returned an error envelope",
                diagnostics=_provider_error_diagnostics(envelope),
            )
        self._completion_status(envelope)
        content = _coalesce_structured_content(self._extract_content(envelope))
        result = parse_strict_json_object(content)
        _validate_response_against_schema(result, validator=validator)
        self._record_usage(envelope)
        return result

    def release_resources(self) -> None:
        """Remote HTTP providers do not own a local model residency scope."""

        return None


class OpenAICompatibleProvider(HTTPJSONLLMProvider):
    """OpenAI chat-completions provider for official and compatible APIs."""

    provider_id = "openai-compatible"
    provider_version = "chat-completions-json-v1"

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        super().__init__(
            config or OpenAICompatibleConfig(),
            secret_resolver=secret_resolver,
        )


class HuggingFaceRouterProvider(OpenAICompatibleProvider):
    """Hugging Face Inference Router through its OpenAI-compatible API."""

    provider_id = "huggingface-router"
    provider_version = "openai-router-json-v1"

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        super().__init__(
            config or HuggingFaceRouterConfig(),
            secret_resolver=secret_resolver,
        )


class CustomJSONProvider(HTTPJSONLLMProvider):
    """Configurable JSON request/response adapter for private model gateways."""

    provider_id = "custom-http-json"
    provider_version = "template-json-v1"

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        super().__init__(
            config or CustomJSONProviderConfig(),
            secret_resolver=secret_resolver,
        )

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        selected_model: str,
        temperature: float,
        schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        template = self.config.request_template
        if template is None:
            return super()._build_payload(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                selected_model=selected_model,
                temperature=temperature,
                schema=schema,
            )
        rendered = _render_request_template(
            template,
            {
                "model": selected_model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": float(temperature),
                "top_p": self.config.top_p,
                "max_tokens": self.config.output_tokens,
                "response_schema": dict(schema) if schema is not None else None,
            },
        )
        return parse_strict_json_object(rendered)


class AnthropicProvider(HTTPJSONLLMProvider):
    """Native Anthropic Messages API provider with structured-output support."""

    provider_id = "anthropic"
    provider_version = "messages-json-v1"

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        super().__init__(
            config or AnthropicProviderConfig(),
            secret_resolver=secret_resolver,
        )

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        selected_model: str,
        temperature: float,
        schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **dict(self.config.extra_body),
            "model": selected_model,
            "max_tokens": self.config.output_tokens,
            "temperature": float(temperature),
            "top_p": self.config.top_p,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if schema is not None and self.config.response_format == "json-schema":
            payload["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": dict(schema),
                }
            }
        return payload

    def _completion_status(self, envelope: Mapping[str, Any]) -> None:
        reason = envelope.get("stop_reason")
        if reason is None:
            raise LocalLLMError("Anthropic response has no stop reason")
        if not isinstance(reason, str):
            raise LocalLLMError("Anthropic response has an invalid stop reason")
        normalized = reason.strip().casefold()
        if normalized == "max_tokens":
            raise LocalLLMError("Anthropic truncated the structured response")
        if normalized not in {"end_turn", "stop_sequence", "tool_use"}:
            raise LocalLLMError("Anthropic did not complete the structured response")

    def _record_usage(self, envelope: Mapping[str, Any]) -> None:
        usage = envelope.get("usage")
        prompt_tokens = 0
        output_tokens = 0
        if isinstance(usage, Mapping):
            raw_prompt = usage.get("input_tokens")
            raw_output = usage.get("output_tokens")
            if (
                not isinstance(raw_prompt, bool)
                and isinstance(raw_prompt, int)
                and raw_prompt >= 0
            ):
                prompt_tokens = raw_prompt
            if (
                not isinstance(raw_output, bool)
                and isinstance(raw_output, int)
                and raw_output >= 0
            ):
                output_tokens = raw_output
        with self._metrics_lock:
            self._generation_metrics["completedCalls"] += 1
            self._generation_metrics["promptTokens"] += prompt_tokens
            self._generation_metrics["outputTokens"] += output_tokens
            self._generation_metrics["totalTokens"] += prompt_tokens + output_tokens


class GoogleGeminiProvider(HTTPJSONLLMProvider):
    """Native Google Gemini generateContent provider."""

    provider_id = "google-gemini"
    provider_version = "generate-content-json-v1"

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        super().__init__(
            config or GoogleGeminiProviderConfig(),
            secret_resolver=secret_resolver,
        )

    def _request_url(self, selected_model: str) -> str:
        model_name = selected_model.removeprefix("models/").strip()
        if not model_name:
            raise LocalLLMError("Gemini model name must not be empty")
        encoded_model = urllib.parse.quote(model_name, safe="-._")
        return (
            self.config.endpoint.rstrip("/")
            + f"/models/{encoded_model}:generateContent"
        )

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        selected_model: str,
        temperature: float,
        schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        del selected_model
        generation_config: dict[str, Any] = {
            "temperature": float(temperature),
            "topP": self.config.top_p,
            "maxOutputTokens": self.config.output_tokens,
            "responseMimeType": "application/json",
        }
        if schema is not None and self.config.response_format == "json-schema":
            generation_config["responseJsonSchema"] = dict(schema)
        return {
            **dict(self.config.extra_body),
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": generation_config,
        }

    def _completion_status(self, envelope: Mapping[str, Any]) -> None:
        reason = _optional_json_path(
            envelope,
            self.config.finish_reason_path,
        )
        if not isinstance(reason, str):
            raise LocalLLMError("Gemini response has no valid finish reason")
        normalized = reason.strip().casefold().replace("_", "-")
        if normalized in {"max-tokens", "max-token"}:
            raise LocalLLMError("Gemini truncated the structured response")
        if normalized != "stop":
            raise LocalLLMError("Gemini did not complete the structured response")

    def _record_usage(self, envelope: Mapping[str, Any]) -> None:
        usage = envelope.get("usageMetadata")
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 0
        if isinstance(usage, Mapping):
            raw_prompt = usage.get("promptTokenCount")
            raw_output = usage.get("candidatesTokenCount")
            raw_total = usage.get("totalTokenCount")
            if (
                not isinstance(raw_prompt, bool)
                and isinstance(raw_prompt, int)
                and raw_prompt >= 0
            ):
                prompt_tokens = raw_prompt
            if (
                not isinstance(raw_output, bool)
                and isinstance(raw_output, int)
                and raw_output >= 0
            ):
                output_tokens = raw_output
            if (
                not isinstance(raw_total, bool)
                and isinstance(raw_total, int)
                and raw_total >= 0
            ):
                total_tokens = raw_total
        if total_tokens == 0:
            total_tokens = prompt_tokens + output_tokens
        with self._metrics_lock:
            self._generation_metrics["completedCalls"] += 1
            self._generation_metrics["promptTokens"] += prompt_tokens
            self._generation_metrics["outputTokens"] += output_tokens
            self._generation_metrics["totalTokens"] += total_tokens


OpenAICompatibleLLMProvider = OpenAICompatibleProvider
HuggingFaceProvider = HuggingFaceRouterProvider
AnthropicLLMProvider = AnthropicProvider
GeminiProvider = GoogleGeminiProvider


LLM_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    # Official OpenAI-compatible endpoints. A caller can override endpoint and
    # model for an enterprise deployment or an OpenAI-compatible relay.
    "openai": {
        "provider": "openai-compatible",
        "endpoint": "https://api.openai.com/v1",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
        "require_api_key": True,
    },
    "deepseek": {
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "require_api_key": True,
    },
    "openrouter": {
        "provider": "openrouter",
        "endpoint": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-5",
        "api_key_env": "OPENROUTER_API_KEY",
        "require_api_key": True,
    },
    "groq": {
        "provider": "groq",
        "endpoint": "https://api.groq.com/openai/v1",
        "model": "llama-4-scout-17b-16e-instruct",
        "api_key_env": "GROQ_API_KEY",
        "require_api_key": True,
    },
    "mistral": {
        "provider": "mistral",
        "endpoint": "https://api.mistral.ai/v1",
        "model": "mistral-large-latest",
        "api_key_env": "MISTRAL_API_KEY",
        "require_api_key": True,
    },
    "together": {
        "provider": "together",
        "endpoint": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "api_key_env": "TOGETHER_API_KEY",
        "require_api_key": True,
    },
    "siliconflow": {
        "provider": "siliconflow",
        "endpoint": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen3-32B",
        "api_key_env": "SILICONFLOW_API_KEY",
        "require_api_key": True,
    },
    "dashscope": {
        "provider": "dashscope",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.5-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "require_api_key": True,
    },
    "moonshot": {
        "provider": "moonshot",
        "endpoint": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.5",
        "api_key_env": "MOONSHOT_API_KEY",
        "require_api_key": True,
    },
    "zhipu": {
        "provider": "zhipu",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-5",
        "api_key_env": "ZHIPUAI_API_KEY",
        "require_api_key": True,
    },
    "xai": {
        "provider": "xai",
        "endpoint": "https://api.x.ai/v1",
        "model": "grok-4",
        "api_key_env": "XAI_API_KEY",
        "require_api_key": True,
    },
    "perplexity": {
        "provider": "perplexity",
        "endpoint": "https://api.perplexity.ai",
        "model": "sonar-pro",
        "api_key_env": "PERPLEXITY_API_KEY",
        "require_api_key": True,
    },
    "huggingface": {
        "provider": "huggingface-router",
        "endpoint": "https://router.huggingface.co/v1",
        "model": "Qwen/Qwen3-32B",
        "api_key_env": "HF_TOKEN",
        "require_api_key": True,
    },
    "huggingface-router": {
        "provider": "huggingface-router",
        "endpoint": "https://router.huggingface.co/v1",
        "model": "Qwen/Qwen3-32B",
        "api_key_env": "HF_TOKEN",
        "require_api_key": True,
    },
}

_PROVIDER_ALIASES = {
    "ollama": "ollama-loopback",
    "ollama-local": "ollama-loopback",
    "ollama-loopback": "ollama-loopback",
    "openai": "openai",
    "openai-compatible": "openai-compatible",
    "openai-compatible-http": "openai-compatible",
    "hf": "huggingface-router",
    "huggingface": "huggingface-router",
    "huggingface-router": "huggingface-router",
    "custom": "custom-http-json",
    "custom-json": "custom-http-json",
    "custom-http": "custom-http-json",
    "custom-http-json": "custom-http-json",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "google": "google-gemini",
    "gemini": "google-gemini",
    "google-gemini": "google-gemini",
}


def _normalize_provider_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("LLM provider must be text")
    normalized = value.strip().casefold()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _provider_config_from_input(
    config: LLMProviderConfig | Mapping[str, Any] | str,
) -> LLMProviderConfig | LocalLLMConfig:
    if isinstance(config, LocalLLMConfig):
        return config
    if isinstance(config, LLMProviderConfig):
        if _normalize_provider_id(config.provider) == "ollama-loopback":
            if config.network_policy != NETWORK_POLICY_LOOPBACK_ONLY:
                raise ValueError(
                    "Ollama provider network policy must be loopback-only"
                )
            if config.allow_model_override:
                raise ValueError(
                    "Ollama provider model override is not supported"
                )
            if (
                config.api_key_env is not None
                or config.header_env
                or config.proxy is not None
                or config.require_api_key
            ):
                raise ValueError(
                    "Ollama loopback provider must not configure remote authentication"
                )
            return LocalLLMConfig(
                model=config.model,
                endpoint=config.endpoint,
                timeout_seconds=config.timeout_seconds,
                temperature=config.temperature,
                top_p=config.top_p,
                context_tokens=config.context_tokens,
                output_tokens=config.output_tokens,
                expected_model_digest=config.expected_model_digest,
            )
        return config
    if isinstance(config, str):
        raw: Mapping[str, Any] = {"provider": config}
    elif isinstance(config, Mapping):
        raw = config
    else:
        raise TypeError(
            "LLM provider configuration must be LocalLLMConfig, LLMProviderConfig, mapping, or provider name"
        )
    raw_provider = raw.get("provider", raw.get("providerId", "openai-compatible"))
    provider_id = _normalize_provider_id(raw_provider)
    if provider_id == "ollama-loopback":
        allowed = {
            "model",
            "endpoint",
            "timeout_seconds",
            "timeoutSeconds",
            "temperature",
            "top_p",
            "topP",
            "network_policy",
            "networkPolicy",
            "context_tokens",
            "contextTokens",
            "output_tokens",
            "outputTokens",
            "keep_alive",
            "keepAlive",
            "release_on_close",
            "releaseOnClose",
            "offline_only",
            "offlineOnly",
            "expected_model_digest",
            "expectedDigest",
            "expectedModelDigest",
            "allow_model_override",
            "allowModelOverride",
            "api_key_env",
            "apiKeyEnv",
            "header_env",
            "headerEnv",
            "proxy",
            "proxyUrl",
            "require_api_key",
            "requireApiKey",
        }
        unknown = set(raw) - allowed - {"provider", "providerId"}
        if unknown:
            raise ValueError(
                "unsupported Ollama provider configuration field: "
                + sorted(str(item) for item in unknown)[0]
            )
        values: dict[str, Any] = {
            "model": raw.get("model", LocalLLMConfig.model),
            "endpoint": raw.get("endpoint", LocalLLMConfig.endpoint),
            "timeout_seconds": raw.get(
                "timeout_seconds",
                raw.get("timeoutSeconds", LocalLLMConfig.timeout_seconds),
            ),
            "temperature": raw.get("temperature", LocalLLMConfig.temperature),
            "top_p": raw.get("top_p", raw.get("topP", LocalLLMConfig.top_p)),
            "context_tokens": raw.get(
                "context_tokens",
                raw.get("contextTokens", LocalLLMConfig.context_tokens),
            ),
            "output_tokens": raw.get(
                "output_tokens",
                raw.get("outputTokens", LocalLLMConfig.output_tokens),
            ),
            "keep_alive": raw.get(
                "keep_alive",
                raw.get("keepAlive", LocalLLMConfig.keep_alive),
            ),
            "release_on_close": raw.get(
                "release_on_close",
                raw.get("releaseOnClose", LocalLLMConfig.release_on_close),
            ),
            "offline_only": raw.get(
                "offline_only",
                raw.get("offlineOnly", LocalLLMConfig.offline_only),
            ),
            "expected_model_digest": raw.get(
                "expected_model_digest",
                raw.get(
                    "expectedDigest",
                    raw.get("expectedModelDigest"),
                ),
            ),
        }
        network_policy = raw.get(
            "network_policy",
            raw.get("networkPolicy", "loopback-only"),
        )
        if network_policy != "loopback-only":
            raise ValueError("Ollama provider network policy must be loopback-only")
        allow_model_override = raw.get(
            "allow_model_override",
            raw.get("allowModelOverride", False),
        )
        if allow_model_override:
            raise ValueError("Ollama provider model override is not supported")
        api_key_env = raw.get("api_key_env", raw.get("apiKeyEnv"))
        header_env = raw.get("header_env", raw.get("headerEnv", {}))
        proxy = raw.get("proxy", raw.get("proxyUrl"))
        require_api_key = raw.get(
            "require_api_key",
            raw.get("requireApiKey", False),
        )
        if api_key_env is not None or header_env or proxy is not None or require_api_key:
            raise ValueError(
                "Ollama loopback provider must not configure remote authentication"
            )
        return LocalLLMConfig(**values)

    preset = LLM_PROVIDER_PRESETS.get(provider_id)
    if preset is None and provider_id in {
        "anthropic",
        "google-gemini",
        "custom-http-json",
        "huggingface-router",
    }:
        preset = {"provider": provider_id}
    merged: dict[str, Any] = dict(preset or {})
    merged.update(dict(raw))
    merged["provider"] = provider_id
    if provider_id == "anthropic":
        # Native providers have a distinct payload shape but share transport
        # controls with the generic config.
        return AnthropicProviderConfig.from_mapping(merged)
    if provider_id == "google-gemini":
        return GoogleGeminiProviderConfig.from_mapping(merged)
    if provider_id == "huggingface-router":
        return HuggingFaceRouterConfig.from_mapping(merged)
    if provider_id == "custom-http-json":
        return CustomJSONProviderConfig.from_mapping(merged)
    return LLMProviderConfig.from_mapping(merged)


def provider_config_from_mapping(
    config: Mapping[str, Any],
) -> LLMProviderConfig | LocalLLMConfig:
    """Public alias used by desktop/configuration layers."""

    return _provider_config_from_input(config)


def get_llm_provider_preset(
    provider: str,
    **overrides: Any,
) -> LLMProviderConfig | LocalLLMConfig:
    """Return a validated provider preset with caller-supplied overrides."""

    provider_id = _normalize_provider_id(provider)
    values: dict[str, Any] = {"provider": provider_id}
    values.update(overrides)
    return _provider_config_from_input(values)


def _coerce_provider_wire_defaults(
    config: LLMProviderConfig,
    provider_id: str,
) -> LLMProviderConfig:
    """Apply native wire defaults to a transport-neutral production config."""

    base_response_path = ("choices", 0, "message", "content")
    base_finish_path = ("choices", 0, "finish_reason")
    if provider_id == "anthropic" and not isinstance(
        config,
        AnthropicProviderConfig,
    ):
        headers = dict(config.headers)
        headers.setdefault("anthropic-version", "2023-06-01")
        return replace(
            config,
            endpoint=(
                "https://api.anthropic.com/v1"
                if config.endpoint == "https://api.openai.com/v1"
                else config.endpoint
            ),
            model=(
                "claude-sonnet-4-5"
                if config.model == "gpt-4o-mini"
                else config.model
            ),
            api_key_env=(
                "ANTHROPIC_API_KEY"
                if config.api_key_env in {None, "OPENAI_API_KEY"}
                else config.api_key_env
            ),
            require_api_key=True,
            auth_header=(
                "x-api-key"
                if config.auth_header == "Authorization"
                else config.auth_header
            ),
            auth_scheme=("" if config.auth_scheme == "Bearer" else config.auth_scheme),
            headers=headers,
            chat_path=(
                "/messages"
                if config.chat_path == "/chat/completions"
                else config.chat_path
            ),
            response_path=(
                ("content",)
                if tuple(config.response_path) == base_response_path
                else config.response_path
            ),
            finish_reason_path=(
                ("stop_reason",)
                if config.finish_reason_path is not None
                and tuple(config.finish_reason_path) == base_finish_path
                else config.finish_reason_path
            ),
        )
    if provider_id == "google-gemini" and not isinstance(
        config,
        GoogleGeminiProviderConfig,
    ):
        return replace(
            config,
            endpoint=(
                "https://generativelanguage.googleapis.com/v1beta"
                if config.endpoint == "https://api.openai.com/v1"
                else config.endpoint
            ),
            model=(
                "gemini-2.5-pro"
                if config.model == "gpt-4o-mini"
                else config.model
            ),
            api_key_env=(
                "GEMINI_API_KEY"
                if config.api_key_env in {None, "OPENAI_API_KEY"}
                else config.api_key_env
            ),
            require_api_key=True,
            auth_header=(
                "x-goog-api-key"
                if config.auth_header == "Authorization"
                else config.auth_header
            ),
            auth_scheme=("" if config.auth_scheme == "Bearer" else config.auth_scheme),
            chat_path=("" if config.chat_path == "/chat/completions" else config.chat_path),
            response_path=(
                ("candidates", 0, "content", "parts")
                if tuple(config.response_path) == base_response_path
                else config.response_path
            ),
            finish_reason_path=(
                ("candidates", 0, "finishReason")
                if config.finish_reason_path is not None
                and tuple(config.finish_reason_path) == base_finish_path
                else config.finish_reason_path
            ),
        )
    if provider_id == "huggingface-router" and not isinstance(
        config,
        HuggingFaceRouterConfig,
    ):
        return replace(
            config,
            endpoint=(
                "https://router.huggingface.co/v1"
                if config.endpoint == "https://api.openai.com/v1"
                else config.endpoint
            ),
            api_key_env=(
                "HF_TOKEN"
                if config.api_key_env in {None, "OPENAI_API_KEY"}
                else config.api_key_env
            ),
            require_api_key=True,
        )
    return config


def create_llm_provider(
    config: LLMProviderConfig | LocalLLMConfig | Mapping[str, Any] | str,
    *,
    secret_resolver: SecretResolver | None = None,
) -> LocalLLMProvider:
    """Create an LLM provider while preserving the legacy Ollama config path."""

    normalized = _provider_config_from_input(config)
    if isinstance(normalized, LocalLLMConfig):
        return OllamaLocalProvider(normalized)
    provider_id = _normalize_provider_id(normalized.provider)
    normalized = _coerce_provider_wire_defaults(normalized, provider_id)
    if provider_id == "openai-compatible":
        return OpenAICompatibleProvider(
            normalized,
            secret_resolver=secret_resolver,
        )
    if provider_id == "huggingface-router":
        return HuggingFaceRouterProvider(
            normalized,
            secret_resolver=secret_resolver,
        )
    if provider_id == "custom-http-json":
        return CustomJSONProvider(
            normalized,
            secret_resolver=secret_resolver,
        )
    if provider_id == "anthropic":
        return AnthropicProvider(
            normalized,
            secret_resolver=secret_resolver,
        )
    if provider_id == "google-gemini":
        return GoogleGeminiProvider(
            normalized,
            secret_resolver=secret_resolver,
        )
    # Any other stable provider identifier is an OpenAI-compatible deployment
    # name. This covers Azure deployments and user-defined relays without
    # requiring a hard-coded vendor allowlist.
    return OpenAICompatibleProvider(
        normalized,
        secret_resolver=secret_resolver,
    )


build_llm_provider = create_llm_provider


class LLMProviderFactory:
    """Small dependency-free factory facade for desktop and service wiring."""

    @staticmethod
    def create(
        config: LLMProviderConfig | LocalLLMConfig | Mapping[str, Any] | str,
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> LocalLLMProvider:
        return create_llm_provider(config, secret_resolver=secret_resolver)


class MappingLocalLLMProvider:
    """Small deterministic provider useful for tests and offline fixtures."""

    provider_id = "mapping-fixture"
    provider_version = "1"
    network_policy = "loopback-only"
    business_batch_size = 1
    business_translation_segment_attempts = 1
    business_generation_attempts = 1

    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self._responses = [dict(item) for item in responses]

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
        cancellation_check: Any = None,
    ) -> Mapping[str, Any]:
        del system_prompt, user_prompt, model, temperature, response_schema
        _check_cancelled(cancellation_check)
        if not self._responses:
            raise LocalLLMError("mapping provider has no remaining responses")
        return parse_strict_json_object(self._responses.pop(0))

    def release_resources(self) -> None:
        return None


__all__ = [
    "AnthropicConfig",
    "AnthropicLLMProvider",
    "AnthropicProvider",
    "AnthropicProviderConfig",
    "CustomJSONConfig",
    "CustomJSONProvider",
    "CustomJSONProviderConfig",
    "GeminiProvider",
    "GeminiProviderConfig",
    "GoogleGeminiConfig",
    "GoogleGeminiProvider",
    "GoogleGeminiProviderConfig",
    "HTTPJSONLLMProvider",
    "HuggingFaceProvider",
    "HuggingFaceRouterConfig",
    "HuggingFaceRouterProvider",
    "LLM_PROVIDER_PRESETS",
    "LLMProviderConfig",
    "LLMProviderFactory",
    "LocalLLMConfig",
    "LocalLLMContextWindowError",
    "LocalLLMError",
    "LocalLLMProvider",
    "MappingLocalLLMProvider",
    "NETWORK_POLICY_HTTPS_OR_LOOPBACK",
    "NETWORK_POLICY_LOOPBACK_ONLY",
    "NETWORK_POLICY_REMOTE_EXPLICIT",
    "OllamaLocalProvider",
    "OpenAICompatibleConfig",
    "OpenAICompatibleLLMProvider",
    "OpenAICompatibleProvider",
    "PROVIDER_NETWORK_POLICIES",
    "SecretResolver",
    "assert_provider_network_policy",
    "assert_loopback_provider",
    "build_llm_provider",
    "create_llm_provider",
    "estimate_input_tokens",
    "get_llm_provider_preset",
    "parse_strict_json_object",
    "provider_config_from_mapping",
]
