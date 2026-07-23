"""Strict, offline-first local language-model boundary.

The production worker deliberately treats a local model as an untrusted
business-output provider.  It may translate, polish, or summarize text, but
it cannot mutate acoustic evidence, timestamps, speaker identities, review
locks, or the immutable raw transcript.

The default provider is Ollama-compatible and is restricted to loopback
addresses.  This module uses only the Python standard library so that the
worker remains installable in the ``media-asr`` environment without adding a
network client dependency.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .errors import JobCancelled


class LocalLLMError(RuntimeError):
    """Raised when a local provider cannot produce a valid JSON response."""


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


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configuration for an explicitly local provider."""

    model: str = "qwen3.5:4b"
    endpoint: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 180.0
    temperature: float = 0.0
    top_p: float = 0.1
    context_tokens: int = 4096
    output_tokens: int = 1024
    keep_alive: str = "10m"
    offline_only: bool = True

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
        if self.offline_only is not True:
            raise ValueError(
                "offline_only must remain enabled for the local LLM boundary"
            )
        _assert_loopback_endpoint(endpoint)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "keep_alive", keep_alive)


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


def _estimated_tokens(*values: str) -> int:
    """Return a conservative deterministic token estimate for preflight checks."""

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
    raise LocalLLMError(
        f"local LLM output failed response schema at {location}: {error.message}"
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
        estimated_input_tokens = _estimated_tokens(
            system_prompt,
            user_prompt,
            schema_text,
        )
        input_token_budget = self.config.context_tokens - self.config.output_tokens
        if estimated_input_tokens > input_token_budget:
            raise LocalLLMError(
                "local LLM request exceeds the configured context window"
            )
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
        return result


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


__all__ = [
    "LocalLLMConfig",
    "LocalLLMError",
    "LocalLLMProvider",
    "MappingLocalLLMProvider",
    "OllamaLocalProvider",
    "assert_loopback_provider",
    "parse_strict_json_object",
]
