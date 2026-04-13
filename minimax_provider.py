#!/usr/bin/env python3
"""
MiniMax Custom LiteLLM Provider
Handles MiniMax authentication (user/password → x-auth-token) and delegates
actual API calls to LiteLLM's built-in OpenAI-compatible provider.
"""

from __future__ import annotations

import json
import os
import threading

import requests as http_requests

import litellm
from litellm import CustomLLM
from litellm.types.utils import ModelResponse, Usage

# ──────────────────────────────────────────────
# TokenManager
# ──────────────────────────────────────────────


class TokenManager:
    def __init__(
        self,
        base_api_1: str,
        user: str,
        password: str,
        refresh_interval: int = 3600,
    ):
        self.base_api_1 = base_api_1
        self.user = user
        self.password = password
        self.refresh_interval = refresh_interval
        self._token: str = ""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def get_token(self) -> str:
        with self._lock:
            return self._token

    def _fetch_token(self) -> str:
        print("[Token] 正在获取认证令牌...")
        resp = http_requests.post(
            f"{self.base_api_1}/login/v4/secureLogin",
            json={"user": self.user, "password": self.password},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise ValueError(f"登录失败: {data}")
        token = data["cloudDragonTokens"]["authToken"]
        print("[Token] 令牌获取成功")
        return token

    def init_token(self):
        with self._lock:
            self._token = self._fetch_token()

    def start_refresh_loop(self):
        def loop():
            while not self._stop_event.wait(self.refresh_interval):
                try:
                    with self._lock:
                        self._token = self._fetch_token()
                    print(
                        f"[Token] 令牌已刷新，{self.refresh_interval}秒后再次刷新"
                    )
                except Exception as e:
                    print(f"[Token] 刷新失败: {e}")

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def stop(self):
        self._stop_event.set()


# ──────────────────────────────────────────────
# Module-level instances (initialized on import)
# ──────────────────────────────────────────────

TARGET_MODEL = os.environ.get("TARGET_MODEL", "MiniMax-M2.5")
BASE_API_1 = os.environ.get("MINIMAX_BASE_API_1", "")
BASE_API_2 = os.environ.get("MINIMAX_BASE_API_2", "")
MINIMAX_USER = os.environ.get("MINIMAX_USER", "")
MINIMAX_PASSWORD = os.environ.get("MINIMAX_PASSWORD", "")

if not all([BASE_API_1, BASE_API_2, MINIMAX_USER, MINIMAX_PASSWORD]):
    raise RuntimeError(
        "请设置环境变量: MINIMAX_BASE_API_1, MINIMAX_BASE_API_2, "
        "MINIMAX_USER, MINIMAX_PASSWORD"
    )

token_manager = TokenManager(BASE_API_1, MINIMAX_USER, MINIMAX_PASSWORD)


def _safe_init():
    try:
        token_manager.init_token()
        token_manager.start_refresh_loop()
        print(f"[MiniMax Provider] 初始化完成, 目标模型: {TARGET_MODEL}")
    except Exception as e:
        print(f"[MiniMax Provider] Token 初始化失败: {e}")
        print("[MiniMax Provider] 请检查环境变量是否正确配置")


_safe_init()


# ──────────────────────────────────────────────
# Anthropic → OpenAI conversion helpers
# ──────────────────────────────────────────────


def _convert_messages(messages: list) -> list:
    """Convert Anthropic format messages to OpenAI format."""
    converted = []
    system_content = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_content.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_content.append(block["text"])
            continue
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        text_parts.append("[image]")
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        converted.append({"role": role, "content": content})
    if system_content:
        converted.insert(0, {"role": "system", "content": "\n".join(system_content)})
    return converted


def _convert_tools(tools: list) -> list | None:
    """Convert Anthropic format tools to OpenAI format."""
    if not tools:
        return None
    openai_tools = []
    for tool in tools:
        tool_type = tool.get("type", "")
        if tool_type == "function":
            openai_tools.append(tool)
        elif tool_type in ("computer_20250124", "web_search_20250305"):
            continue
        else:
            name = tool.get("name", "")
            if not name:
                continue
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
    return openai_tools if openai_tools else None


def _convert_tool_choice(tool_choice) -> str | None:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "")
        if tc_type == "auto":
            return "auto"
        if tc_type == "any":
            return "required"
        if tc_type == "tool":
            name = tool_choice.get("tool", {}).get("name", "")
            if name:
                return {"type": "function", "function": {"name": name}}
    return "auto"


def _convert_tool_calls_response(tool_calls) -> list:
    """Convert OpenAI tool_calls to Anthropic tool_use content blocks."""
    if not tool_calls:
        return []
    blocks = []
    for tc in tool_calls:
        tc_id = getattr(tc, "id", f"call_{id(tc)}")
        func = getattr(tc, "function", None)
        if func:
            name = getattr(func, "name", "")
            args = getattr(func, "arguments", "{}")
            try:
                args_dict = json.loads(args) if isinstance(args, str) else args
            except (json.JSONDecodeError, TypeError):
                args_dict = {}
            blocks.append({"type": "tool_use", "id": tc_id, "name": name, "input": args_dict})
    return blocks


# ──────────────────────────────────────────────
# MiniMax CustomLLM Provider
# ──────────────────────────────────────────────


class MiniMaxCustomAuth(CustomLLM):
    """Custom LiteLLM provider: injects x-auth-token, delegates to OpenAI provider."""

    def __init__(self) -> None:
        super().__init__()

    def completion(self, *args, **kwargs):
        params = self._build_params(kwargs, stream=False)
        response = litellm.completion(**params)
        return self._convert_response(response)

    async def acompletion(self, *args, **kwargs):
        params = self._build_params(kwargs, stream=False)
        response = await litellm.acompletion(**params)
        return self._convert_response(response)

    def streaming(self, *args, **kwargs):
        """Delegate to litellm.completion(openai/...), convert chunks to GenericStreamingChunk."""
        params = self._build_params(kwargs, stream=True)
        response = litellm.completion(**params)
        for chunk in response:
            for gc in _to_generic_chunk(chunk):
                yield gc

    async def astreaming(self, *args, **kwargs):
        """Delegate to litellm.acompletion(openai/...), convert chunks to GenericStreamingChunk."""
        params = self._build_params(kwargs, stream=True)
        response = await litellm.acompletion(**params)
        async for chunk in response:
            for gc in _to_generic_chunk(chunk):
                yield gc

    def _build_params(self, kwargs: dict, stream: bool) -> dict:
        token = token_manager.get_token()
        raw_messages = kwargs.get("messages", [])
        messages = _convert_messages(raw_messages)

        tools = kwargs.get("tools")
        if not tools:
            opt_params = kwargs.get("optional_params", {})
            if isinstance(opt_params, dict):
                tools = opt_params.get("tools")
        openai_tools = _convert_tools(tools)

        tool_choice = kwargs.get("tool_choice")
        if not tool_choice:
            opt_params = kwargs.get("optional_params", {})
            if isinstance(opt_params, dict):
                tool_choice = opt_params.get("tool_choice")
        openai_tool_choice = _convert_tool_choice(tool_choice)

        print(
            f"[MiniMax] -> model={TARGET_MODEL}, messages={len(messages)}, "
            f"stream={stream}, tools={len(openai_tools) if openai_tools else 0}"
        )

        params = {
            "model": f"openai/{TARGET_MODEL}",
            "messages": messages,
            "api_base": f"{BASE_API_2}/api/v2",
            "api_key": "not-needed",
            "extra_headers": {"x-auth-token": token},
            "stream": stream,
            "max_tokens": kwargs.get("max_tokens") or kwargs.get("optional_params", {}).get("max_tokens", 4096),
        }

        if openai_tools:
            params["tools"] = openai_tools
        if openai_tool_choice:
            params["tool_choice"] = openai_tool_choice
        for key in ("temperature", "top_p", "stop"):
            val = kwargs.get(key) or kwargs.get("optional_params", {}).get(key)
            if val is not None:
                params[key] = val

        return params

    def _convert_response(self, response: ModelResponse) -> ModelResponse:
        """Convert tool_calls to Anthropic tool_use in content."""
        if not response.choices:
            return response
        for choice in response.choices:
            message = getattr(choice, "message", None)
            if not message:
                continue
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls and len(tool_calls) > 0:
                tool_use_blocks = _convert_tool_calls_response(tool_calls)
                message.content = json.dumps(tool_use_blocks, ensure_ascii=False)
                choice.finish_reason = "tool_use"
                message.tool_calls = None
        return response


minimax_custom_auth = MiniMaxCustomAuth()


def _to_generic_chunk(chunk):
    """Convert ModelResponseStream to GenericStreamingChunk dicts.

    LiteLLM's CustomStreamWrapper requires GenericStreamingChunk dicts for
    custom providers - no built-in converter exists, so we write this once.
    Each tool_call in delta yields a separate chunk.
    """
    if not hasattr(chunk, "choices") or not chunk.choices:
        return

    choice = chunk.choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None:
        return

    content = getattr(delta, "content", None) or ""
    finish_reason = getattr(choice, "finish_reason", None) or ""
    is_finished = finish_reason != ""

    # Handle tool_calls: yield one chunk per tool_call
    tool_calls = getattr(delta, "tool_calls", None) or []
    for tc in tool_calls:
        tc_index = getattr(tc, "index", 0)
        func = getattr(tc, "function", None) or {}
        name = getattr(func, "name", "")
        args = getattr(func, "arguments", "")
        tool_use = {"type": "function", "index": tc_index}
        tc_id = getattr(tc, "id", None)
        if tc_id:
            tool_use["id"] = tc_id
        if name or args:
            tool_use["function"] = {}
            if name:
                tool_use["function"]["name"] = name
            if args:
                tool_use["function"]["arguments"] = args
        yield {
            "text": "",
            "is_finished": is_finished,
            "finish_reason": finish_reason,
            "index": tc_index,
            "tool_use": tool_use,
            "usage": None,
        }

    # Handle regular content
    if content or is_finished:
        yield {
            "text": content,
            "is_finished": is_finished,
            "finish_reason": finish_reason,
            "index": 0,
            "tool_use": None,
            "usage": None,
        }
