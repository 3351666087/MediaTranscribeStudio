from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from backend.business_contracts import (
    BusinessOutputContractError,
    validate_business_output_contract,
)
from backend.errors import JobCancelled
from backend.local_llm import (
    LocalLLMConfig,
    LocalLLMError,
    MappingLocalLLMProvider,
    OllamaLocalProvider,
    parse_strict_json_object,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


def test_business_contract_schemas_are_valid_json_and_versioned() -> None:
    expected = {
        "business-processing-request.schema.json": (
            "MediaTranscribeStudio Business Processing Request",
            "1.2.0",
        ),
        "translation-output.schema.json": (
            "MediaTranscribeStudio Translation Variant",
            "1.1.0",
        ),
        "polish-output.schema.json": (
            "MediaTranscribeStudio Polished Transcript Variant",
            "1.1.0",
        ),
        "summary-output.schema.json": (
            "MediaTranscribeStudio Evidence-Grounded Summary Variant",
            "1.1.0",
        ),
    }

    for filename, (title, version) in expected.items():
        path = CONTRACTS / filename
        assert path.is_file(), filename
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == title
        assert schema["$id"].endswith(f"/{version}")
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False


def test_runtime_validator_enforces_the_public_summary_schema() -> None:
    evidence = {
        "text": "Evidence-grounded point.",
        "evidenceSegmentIds": ["segment-1"],
        "timeRange": {"startMs": 0, "endMs": 1_000},
    }
    artifact = {
        "schemaVersion": "1.1.0",
        "variant": "summary",
        "inputHash": "a" * 64,
        "model": "qwen3.5:4b",
        "promptVersion": "business-v1",
        "provider": {
            "id": "mapping-fixture",
            "version": "1",
            "networkPolicy": "loopback-only",
        },
        "temperature": 0,
        "applicationPolicy": "suggestion-only",
        "requiresHumanApproval": True,
        "status": "completed",
        "language": "en",
        "executiveSummary": "Summary.",
        "keyPoints": [copy.deepcopy(evidence)],
        "topics": [],
        "actionItems": [],
    }

    validate_business_output_contract(artifact, variant="summary")

    invalid_extra = copy.deepcopy(artifact)
    invalid_extra["keyPoints"][0]["unexpected"] = True
    with pytest.raises(BusinessOutputContractError):
        validate_business_output_contract(invalid_extra, variant="summary")

    invalid_empty = copy.deepcopy(artifact)
    invalid_empty["keyPoints"][0]["text"] = ""
    with pytest.raises(BusinessOutputContractError):
        validate_business_output_contract(invalid_empty, variant="summary")

    invalid_duplicate = copy.deepcopy(artifact)
    invalid_duplicate["keyPoints"][0]["evidenceSegmentIds"] = [
        "segment-1",
        "segment-1",
    ]
    with pytest.raises(BusinessOutputContractError):
        validate_business_output_contract(invalid_duplicate, variant="summary")


def test_strict_json_rejects_fences_duplicates_non_finite_and_non_objects() -> None:
    assert parse_strict_json_object('{"answer":"ok"}') == {"answer": "ok"}
    assert parse_strict_json_object({"answer": "ok"}) == {"answer": "ok"}

    with pytest.raises(LocalLLMError, match="fenced"):
        parse_strict_json_object('```json\n{"answer":"ok"}\n```')
    with pytest.raises(LocalLLMError, match="duplicate"):
        parse_strict_json_object('{"answer":1,"answer":2}')
    with pytest.raises(LocalLLMError, match="non-finite"):
        parse_strict_json_object('{"answer":NaN}')
    with pytest.raises(LocalLLMError, match="root must be an object"):
        parse_strict_json_object("[]")


def test_local_config_normalizes_values_and_rejects_non_loopback_endpoints() -> None:
    config = LocalLLMConfig(
        model="  qwen3.5:4b  ",
        endpoint="  http://127.0.0.1:11434  ",
    )
    assert config.model == "qwen3.5:4b"
    assert config.endpoint == "http://127.0.0.1:11434"

    with pytest.raises(ValueError, match="loopback"):
        LocalLLMConfig(endpoint="http://192.0.2.1:11434")
    with pytest.raises(ValueError, match="offline_only"):
        LocalLLMConfig(offline_only=False)


def test_ollama_provider_version_tracks_batched_business_contract() -> None:
    assert OllamaLocalProvider.provider_version == "native-json-v3"


def test_mapping_provider_is_deterministic_and_fails_closed_when_exhausted() -> None:
    provider = MappingLocalLLMProvider([{"ok": True}])
    assert (
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="fixture",
        )
        == {"ok": True}
    )
    with pytest.raises(LocalLLMError, match="no remaining"):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="fixture",
        )


def test_event_cancellation_preserves_worker_cancellation_semantics() -> None:
    event = threading.Event()
    event.set()
    provider = MappingLocalLLMProvider([{"never": "used"}])

    with pytest.raises(JobCancelled):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model="fixture",
            cancellation_check=event,
        )


def test_ollama_provider_sends_strict_json_chat_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"message":{"content":"{\\"answer\\":\\"ok\\"}"}}'

        def geturl(self) -> str:
            return "http://127.0.0.1:11434/api/chat"

    class FakeOpener:
        def open(self, request: Any, *, timeout: float) -> FakeResponse:
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: FakeOpener(),
    )

    provider = OllamaLocalProvider(
        LocalLLMConfig(model="qwen3.5:4b", timeout_seconds=12.5)
    )
    result = provider.generate_json(
        system_prompt="Return JSON.",
        user_prompt="Answer.",
        model="qwen3.5:4b",
    )

    assert result == {"answer": "ok"}
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 12.5
    assert captured["payload"]["model"] == "qwen3.5:4b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["keep_alive"] == "10m"
    assert captured["payload"]["options"]["temperature"] == 0.0
    assert captured["payload"]["options"]["top_p"] == 0.1
    assert captured["payload"]["options"]["num_ctx"] == 4096
    assert captured["payload"]["options"]["num_predict"] == 1024
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Answer."},
    ]


def test_ollama_provider_sends_a_strict_response_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"message":{"content":"{\\"answer\\":\\"ok\\"}"}}'

        def geturl(self) -> str:
            return "http://127.0.0.1:11434/api/chat"

    class FakeOpener:
        def open(self, request: Any, timeout: float) -> FakeResponse:
            del timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: FakeOpener(),
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }

    provider = OllamaLocalProvider()
    result = provider.generate_json(
        system_prompt="system",
        user_prompt="user",
        model=provider.config.model,
        response_schema=schema,
    )

    assert result == {"answer": "ok"}
    assert captured["payload"]["format"] == schema


def test_ollama_provider_supports_legacy_response_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"response":"{\\"value\\":42}"}'

        def geturl(self) -> str:
            return "http://127.0.0.1:11434/api/chat"

    class FakeOpener:
        def open(self, request: Any, timeout: float) -> FakeResponse:
            del request, timeout
            return FakeResponse()

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: FakeOpener(),
    )
    provider = OllamaLocalProvider()

    assert (
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=provider.config.model,
        )
        == {"value": 42}
    )


def test_local_llm_config_rejects_unbounded_or_invalid_generation_options() -> None:
    with pytest.raises(ValueError, match="top_p"):
        LocalLLMConfig(top_p=0)
    with pytest.raises(ValueError, match="context_tokens"):
        LocalLLMConfig(context_tokens=512)
    with pytest.raises(ValueError, match="output_tokens"):
        LocalLLMConfig(context_tokens=1024, output_tokens=2048)
    with pytest.raises(ValueError, match="keep_alive"):
        LocalLLMConfig(keep_alive="forever")


def test_ollama_provider_rejects_empty_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"message":{"content":"","thinking":"hidden"}}'

        def geturl(self) -> str:
            return "http://127.0.0.1:11434/api/chat"

    class FakeOpener:
        def open(self, request: Any, timeout: float) -> FakeResponse:
            del request, timeout
            return FakeResponse()

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: FakeOpener(),
    )

    provider = OllamaLocalProvider()
    with pytest.raises(LocalLLMError, match="empty structured content"):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=provider.config.model,
        )


def test_ollama_provider_rejects_malformed_provider_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{not-json"

        def geturl(self) -> str:
            return "http://127.0.0.1:11434/api/chat"

    class FakeOpener:
        def open(self, request: Any, timeout: float) -> FakeResponse:
            del request, timeout
            return FakeResponse()

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: FakeOpener(),
    )
    provider = OllamaLocalProvider()

    with pytest.raises(LocalLLMError, match="malformed JSON"):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=provider.config.model,
        )


def test_ollama_provider_rejects_response_url_outside_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://example.com/api/chat"

        def read(self) -> bytes:
            return b'{"message":{"content":"{\\"answer\\":\\"unsafe\\"}"}}'

    class FakeOpener:
        def open(self, request: Any, timeout: float) -> FakeResponse:
            del request, timeout
            return FakeResponse()

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        lambda *handlers: FakeOpener(),
    )

    provider = OllamaLocalProvider()
    with pytest.raises(LocalLLMError, match="escaped the loopback"):
        provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            model=provider.config.model,
        )


def test_ollama_provider_installs_fail_closed_redirect_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeOpener:
        def open(self, request: Any, timeout: float) -> Any:
            del request, timeout
            raise AssertionError("network transport must not run in this test")

    def fake_build_opener(*handlers: Any) -> FakeOpener:
        captured["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setattr(
        "backend.local_llm.urllib.request.build_opener",
        fake_build_opener,
    )
    OllamaLocalProvider()

    handlers = captured["handlers"]
    assert len(handlers) == 1
    with pytest.raises(LocalLLMError, match="redirects are forbidden"):
        handlers[0].redirect_request(
            None,
            None,
            302,
            "Found",
            {},
            "https://example.com/api/chat",
        )
