from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    wall_ms: float
    prompt_eval_count: int
    prompt_eval_duration_ns: int
    eval_count: int
    eval_duration_ns: int
    total_duration_ns: int
    load_duration_ns: int


class OllamaClient:
    def __init__(self, host: str, timeout_seconds: float) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> dict[str, Any]:
        return self._get_json("/api/tags")

    def show_model(self, model: str) -> dict[str, Any]:
        return self._post_json("/api/show", {"model": model})

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        input_payload: dict[str, Any],
        output_schema: dict[str, Any],
        seed: int,
        retry_note: bool,
    ) -> OllamaResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    input_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        if retry_note:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "上一次响应未通过严格契约。重新独立完成任务，只返回一个符合 Schema 的 JSON；"
                        "不要解释，不要复述上下文，不要增加字段。"
                    ),
                }
            )
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": output_schema,
            # Qwen3/Qwen3.5-class Ollama models may spend the entire output
            # budget in message.thinking and return an empty structured
            # message.content.  The benchmark evaluates the deployable,
            # schema-constrained response only, so hidden reasoning is
            # explicitly disabled for every model.
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "seed": seed,
                "top_p": 0.1,
                "num_ctx": 4096,
                "num_predict": 512,
            },
        }
        started = time.perf_counter()
        response = self._post_json("/api/chat", payload)
        wall_ms = (time.perf_counter() - started) * 1000
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("ollama_response_contract")
        content = message["content"]
        if not content.strip():
            # Never interpret or persist message.thinking as the model's
            # answer.  It is not constrained by the requested JSON schema and
            # is not a safe production output surface.
            if isinstance(message.get("thinking"), str) and message["thinking"].strip():
                raise RuntimeError("ollama_empty_content_thinking_only")
            raise RuntimeError("ollama_empty_content")
        return OllamaResponse(
            content=content,
            wall_ms=wall_ms,
            prompt_eval_count=int(response.get("prompt_eval_count") or 0),
            prompt_eval_duration_ns=int(response.get("prompt_eval_duration") or 0),
            eval_count=int(response.get("eval_count") or 0),
            eval_duration_ns=int(response.get("eval_duration") or 0),
            total_duration_ns=int(response.get("total_duration") or 0),
            load_duration_ns=int(response.get("load_duration") or 0),
        )

    def _get_json(self, path: str) -> dict[str, Any]:
        req = request.Request(
            self.host + path,
            headers={"Accept": "application/json"},
            method="GET",
        )
        return self._open(req)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.host + path,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self._open(req)

    def _open(self, req: request.Request) -> dict[str, Any]:
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except error.HTTPError as exc:
            raise RuntimeError(f"ollama_http_error:{exc.code}") from None
        except error.URLError as exc:
            reason = type(exc.reason).__name__
            raise RuntimeError(f"ollama_connection_error:{reason}") from None
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("ollama_invalid_json") from None
        if not isinstance(parsed, dict):
            raise RuntimeError("ollama_root_not_object")
        return parsed
