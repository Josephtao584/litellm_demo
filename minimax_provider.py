#!/usr/bin/env python3
"""
MiniMax Custom LiteLLM Provider
Handles MiniMax authentication (user/password → x-auth-token) and delegates
actual API calls to LiteLLM's built-in OpenAI-compatible provider.
"""

from __future__ import annotations

import os
import threading

import requests as http_requests

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk

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
    """Safely initialize token manager, logging errors but not crashing on bad URLs."""
    try:
        token_manager.init_token()
        token_manager.start_refresh_loop()
        print(f"[MiniMax Provider] 初始化完成, 目标模型: {TARGET_MODEL}")
    except Exception as e:
        print(f"[MiniMax Provider] Token 初始化失败: {e}")
        print("[MiniMax Provider] 请检查环境变量是否正确配置")


_safe_init()


# ──────────────────────────────────────────────
# MiniMax CustomLLM Provider
# Delegates to litellm's built-in OpenAI-compatible provider
# ──────────────────────────────────────────────


class MiniMaxCustomAuth(CustomLLM):
    """Custom LiteLLM provider that injects token auth and delegates to OpenAI provider."""

    def __init__(self) -> None:
        super().__init__()

    def completion(self, *args, **kwargs):
        """Sync completion — delegates to litellm's OpenAI provider."""
        return litellm.completion(**self._build_params(kwargs))

    async def acompletion(self, *args, **kwargs):
        """Async completion — delegates to litellm's OpenAI provider."""
        return await litellm.acompletion(**self._build_params(kwargs))

    def streaming(self, *args, **kwargs):
        """Sync streaming — delegates to litellm's OpenAI provider."""
        response = litellm.completion(**self._build_params(kwargs, stream=True))
        for chunk in response:
            yield self._to_generic_chunk(chunk)

    async def astreaming(self, *args, **kwargs):
        """Async streaming — delegates to litellm's OpenAI provider."""
        response = await litellm.acompletion(**self._build_params(kwargs, stream=True))
        async for chunk in response:
            yield self._to_generic_chunk(chunk)

    @staticmethod
    def _to_generic_chunk(chunk) -> GenericStreamingChunk:
        """Convert litellm streaming chunk to GenericStreamingChunk."""
        text = ""
        finish_reason = ""
        is_finished = False
        if chunk.choices:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None) or ""
            finish_reason = getattr(choice, "finish_reason", None) or ""
            is_finished = finish_reason != ""
        return GenericStreamingChunk(
            text=text,
            is_finished=is_finished,
            finish_reason=finish_reason,
            index=0,
            tool_use=None,
            usage=None,
        )

    def _build_params(self, kwargs: dict, stream: bool = False) -> dict:
        """Build litellm.completion params with token auth and OpenAI-compatible config."""
        token = token_manager.get_token()
        messages = kwargs.get("messages", [])

        # Debug: log all keys to find where tools are
        import pprint
        debug_keys = []
        for k, v in kwargs.items():
            if k == "optional_params":
                debug_keys.append(("optional_params", v))
            elif k == "litellm_params":
                # Just summarize
                debug_keys.append(("litellm_params_keys", list(v.keys()) if isinstance(v, dict) else type(v).__name__))
            elif k == "logging_obj":
                debug_keys.append(("logging_obj", type(v).__name__))
            else:
                debug_keys.append((k, v))
        for k, v in debug_keys:
            print(f"[MiniMax DEBUG] {k}={pprint.pformat(v)[:300]}")
        if "tools" not in kwargs:
            # Check nested locations
            for loc in ["optional_params", "litellm_params"]:
                if loc in kwargs and isinstance(kwargs[loc], dict) and "tools" in kwargs[loc]:
                    print(f"[MiniMax DEBUG] tools found in {loc}")
            print("[MiniMax DEBUG] NO tools anywhere in kwargs")

        print(
            f"[MiniMax] -> model={TARGET_MODEL}, messages={len(messages)}, stream={stream}"
        )

        params = {
            "model": f"openai/{TARGET_MODEL}",
            "messages": messages,
            "api_base": f"{BASE_API_2}/api/v2",
            "api_key": "not-needed",  # OpenAI provider requires an api_key, but we use x-auth-token
            "extra_headers": {
                "x-auth-token": token,
            },
            "stream": stream,
        }

        # Pass through optional params
        for key in ("max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice"):
            if key in kwargs:
                params[key] = kwargs[key]

        return params


# ──────────────────────────────────────────────
# Module-level handler instance
# ──────────────────────────────────────────────

minimax_custom_auth = MiniMaxCustomAuth()
