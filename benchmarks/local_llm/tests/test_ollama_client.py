from __future__ import annotations

import pytest

from local_llm_bench.ollama_client import OllamaClient


SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _chat(client: OllamaClient):
    return client.chat(
        model="qwen3.5:4b",
        system_prompt="只返回 JSON。",
        input_payload={"request": "ok"},
        output_schema=SCHEMA,
        seed=20260721,
        retry_note=False,
    )


def test_chat_explicitly_disables_thinking(monkeypatch):
    client = OllamaClient("http://127.0.0.1:11434", 1)
    captured = {}

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"role": "assistant", "content": '{"ok":true}'}}

    monkeypatch.setattr(client, "_post_json", fake_post)

    response = _chat(client)

    assert response.content == '{"ok":true}'
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["think"] is False


def test_chat_fails_closed_for_thinking_only_response(monkeypatch):
    client = OllamaClient("http://127.0.0.1:11434", 1)

    def fake_post(_path, _payload):
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "private unconstrained reasoning",
            }
        }

    monkeypatch.setattr(client, "_post_json", fake_post)

    with pytest.raises(RuntimeError, match="ollama_empty_content_thinking_only"):
        _chat(client)


def test_chat_fails_closed_for_empty_content_without_thinking(monkeypatch):
    client = OllamaClient("http://127.0.0.1:11434", 1)
    monkeypatch.setattr(
        client,
        "_post_json",
        lambda _path, _payload: {
            "message": {"role": "assistant", "content": "   "}
        },
    )

    with pytest.raises(RuntimeError, match="ollama_empty_content$"):
        _chat(client)
