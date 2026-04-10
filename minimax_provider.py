#!/usr/bin/env python3
"""
MiniMax Custom LiteLLM Provider
Handles MiniMax authentication (user/password → x-auth-token) and provides
a CustomLLM implementation for LiteLLM proxy.
"""

from __future__ import annotations

import json
import os
import threading
import time

import httpx

from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse, Usage

# ──────────────────────────────────────────────
# TokenManager (migrated from proxy.py)
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
        resp = httpx.post(
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
    """Safely initialize token manager, logging errors but not crashing on bad URLs."""
    try:
        token_manager.init_token()
        token_manager.start_refresh_loop()
        print(f"[MiniMax Provider] 初始化完成, 目标模型: {TARGET_MODEL}")
    except Exception as e:
        print(f"[MiniMax Provider] Token 初始化失败: {e}")
        print("[MiniMax Provider] 请检查环境变量是否正确配置")


_safe_init()


def ensure_token_ready() -> None:
    """Ensure the token manager has been initialized (call after import)."""
    # Token is already initialized on module import
    pass


# ──────────────────────────────────────────────
# MiniMax CustomLLM Provider
# ──────────────────────────────────────────────


class MiniMaxCustomAuth(CustomLLM):
    """Custom LiteLLM provider for MiniMax with token-based authentication."""

    def __init__(self) -> None:
        super().__init__()

    def completion(self, *args, **kwargs) -> ModelResponse:
        """Sync completion — delegates to async implementation via sync wrapper."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside an event loop, use sync HTTP
            return self._call_minimax_sync(
                messages=kwargs.get("messages", []),
                model=kwargs.get("model", TARGET_MODEL),
                stream=False,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens", 4096),
            )
        else:
            return asyncio.run(
                self.acompletion(*args, **kwargs)
            )

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """Async completion."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", TARGET_MODEL)
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens", 4096)

        return self._call_minimax_sync(
            messages=messages,
            model=model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _call_minimax_sync(
        self,
        messages: list[dict],
        model: str,
        stream: bool,
        temperature: float | None,
        max_tokens: int,
    ) -> ModelResponse:
        """Call MiniMax API and return a ModelResponse."""
        token = token_manager.get_token()
        headers = {
            "x-auth-token": token,
            "Content-Type": "application/json",
        }
        payload = {
            "model": TARGET_MODEL,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        url = f"{BASE_API_2}/api/v2/chat/completions"
        print(
            f"[MiniMax] -> model={TARGET_MODEL}, messages={len(messages)}, stream={stream}"
        )

        resp = httpx.post(url, json=payload, headers=headers, timeout=180)
        if not resp.is_success:
            raise RuntimeError(
                f"MiniMax API error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        return self._parse_response(data, model)

    def _parse_response(self, data: dict, model: str) -> ModelResponse:
        """Parse MiniMax response into a LiteLLM ModelResponse."""
        choices = data.get("choices", [])
        parsed_choices = []
        for choice in choices:
            message = choice.get("message", {})
            parsed_choices.append(
                {
                    "finish_reason": choice.get("finish_reason", "stop"),
                    "index": choice.get("index", 0),
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                }
            )

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        response = ModelResponse(
            id=data.get("id", ""),
            created=int(time.time()),
            model=model,
            object="chat.completion",
            choices=parsed_choices,
            usage=usage,
        )
        return response

    def streaming(self, *args, **kwargs):
        """Sync streaming — yields GenericStreamingChunk objects."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", TARGET_MODEL)
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens", 4096)

        token = token_manager.get_token()
        headers = {
            "x-auth-token": token,
            "Content-Type": "application/json",
        }
        payload = {
            "model": TARGET_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        url = f"{BASE_API_2}/api/v2/chat/completions"
        print(
            f"[MiniMax Stream] -> model={TARGET_MODEL}, messages={len(messages)}"
        )

        with httpx.stream(
            "POST", url, json=payload, headers=headers, timeout=180
        ) as resp:
            if not resp.is_success:
                raise RuntimeError(
                    f"MiniMax API error {resp.status_code}: {resp.text[:500]}"
                )

            for line in resp.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    yield GenericStreamingChunk(
                        text="",
                        is_finished=True,
                        finish_reason="stop",
                        index=0,
                        tool_use=None,
                        usage=None,
                    )
                    break

                try:
                    chunk_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk_data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                usage_data = chunk_data.get("usage")
                usage = None
                if usage_data:
                    usage = {
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0),
                    }

                yield GenericStreamingChunk(
                    text=delta.get("content", "") or "",
                    is_finished=finish_reason is not None,
                    finish_reason=finish_reason or "",
                    index=choices[0].get("index", 0),
                    tool_use=delta.get("tool_calls"),
                    usage=usage,
                )

    async def astreaming(self, *args, **kwargs):
        """Async streaming — yields GenericStreamingChunk objects."""
        # For now, use sync streaming wrapped in async
        # A full async implementation would use httpx.AsyncClient.stream()
        for chunk in self.streaming(*args, **kwargs):
            yield chunk


# ──────────────────────────────────────────────
# Module-level handler instance
# ──────────────────────────────────────────────

minimax_custom_auth = MiniMaxCustomAuth()
