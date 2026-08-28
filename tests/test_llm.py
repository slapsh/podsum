"""Chat clients and the shared JSON extraction helpers."""

from __future__ import annotations

import json

import httpx
import pytest

from podsum.errors import BackendError
from podsum.llm.base import parse_json_object, strip_reasoning
from podsum.llm.ollama import OllamaChat
from podsum.llm.openrouter import OpenRouterChat


def mock_client(handler, base_url="https://openrouter.ai/api/v1"):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)


def chat_response(content: str, **extra):
    payload = {
        "id": "gen-1",
        "model": "google/gemini-3.7-flash",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "cost": 0.00042},
    }
    payload.update(extra)
    return payload


class TestJsonExtraction:
    def test_plain_json(self):
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_a_chatty_preamble(self):
        assert parse_json_object('Конечно! Вот результат:\n{"a": 1}') == {"a": 1}

    def test_reasoning_block_is_dropped(self):
        text = '<think>надо подумать</think>\n{"a": 1}'
        assert strip_reasoning(text) == '{"a": 1}'
        assert parse_json_object(text) == {"a": 1}

    def test_unparseable_output_raises_with_a_sample(self):
        with pytest.raises(BackendError) as excinfo:
            parse_json_object("совсем не json")
        assert "не json" in (excinfo.value.hint or "")


class TestOpenRouterChat:
    def test_sends_messages_and_returns_usage(self, cloud_config):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=chat_response('{"tldr": ["ок"]}'))

        client = OpenRouterChat(cloud_config, client=mock_client(handler))
        result = client.complete("system", "user", json_mode=True)

        assert captured["auth"] is None or "Bearer" in captured["auth"]
        payload = captured["payload"]
        assert payload["model"] == cloud_config.llm_model
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["usage"] == {"include": True}
        assert result.usage["cost"] == pytest.approx(0.00042)
        assert json.loads(result.text) == {"tldr": ["ок"]}

    def test_empty_content_names_the_likely_cause(self, cloud_config):
        handler = lambda r: httpx.Response(
            200,
            json={"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
        )
        client = OpenRouterChat(cloud_config, client=mock_client(handler))
        with pytest.raises(BackendError) as excinfo:
            client.complete("s", "u")
        assert "length" in (excinfo.value.hint or "")

    def test_context_limit_comes_from_the_live_catalog(self, cloud_config):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={"data": [{"id": cloud_config.llm_model, "context_length": 1048576}]},
                )
            return httpx.Response(200, json=chat_response("{}"))

        client = OpenRouterChat(cloud_config, client=mock_client(handler))
        assert client.context_limit() == 1_048_576

    def test_context_limit_falls_back_when_the_catalog_is_unreachable(self, cloud_config):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no network")

        client = OpenRouterChat(cloud_config, client=mock_client(handler))
        assert client.context_limit() == cloud_config.context_budget("openrouter")


class TestOllamaChat:
    def test_always_sends_an_explicit_num_ctx(self, local_config):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "qwen3.8:27b",
                    "message": {"role": "assistant", "content": '{"tldr": []}'},
                    "prompt_eval_count": 9000,
                    "eval_count": 700,
                    "done": True,
                },
            )

        local_config.num_ctx = 65536
        client = OllamaChat(local_config, client=mock_client(handler, "http://127.0.0.1:11434"))
        result = client.complete("system", "user", json_mode=True)

        payload = captured["payload"]
        assert payload["options"]["num_ctx"] == 65536
        assert payload["stream"] is False
        assert payload["format"] == "json"
        assert result.usage["prompt_tokens"] == 9000
        assert client.context_limit() == 65536

    def test_thinking_models_do_not_leak_their_reasoning(self, local_config):
        handler = lambda r: httpx.Response(
            200,
            json={"message": {"content": "<think>ммм</think>Итог: всё хорошо."}},
        )
        client = OllamaChat(local_config, client=mock_client(handler, "http://127.0.0.1:11434"))
        assert client.complete("s", "u").text == "Итог: всё хорошо."

    def test_unreachable_server_tells_you_what_to_start(self, local_config):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = OllamaChat(local_config, client=mock_client(handler, "http://127.0.0.1:11434"))
        with pytest.raises(BackendError) as excinfo:
            client.complete("s", "u")
        hint = excinfo.value.hint or ""
        assert "ollama serve" in hint and "ollama pull" in hint

    def test_availability_check_reports_a_missing_model(self, local_config):
        handler = lambda r: httpx.Response(200, json={"models": [{"name": "gemma4:12b"}]})
        client = OllamaChat(local_config, client=mock_client(handler, "http://127.0.0.1:11434"))
        available, message = client.is_available()
        assert available is False
        assert "ollama pull qwen3.8:27b" in message
