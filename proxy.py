#!/usr/bin/env python3
"""
MiniMax M2.5 Proxy
支持 OpenAI 和 Anthropic 两种格式请求，自动管理 token
兼容 Claude Code 客户端
"""

import os
import json
import time
import threading
import requests as http_requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

TARGET_MODEL = os.environ.get("TARGET_MODEL", "MiniMax-M2.5")


class TokenManager:
    def __init__(
        self, base_api_1: str, user: str, password: str, refresh_interval: int = 3600
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
        token = data["cloudDragonTokens"]["x-auth-token"]
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
                    print(f"[Token] 令牌已刷新，{self.refresh_interval}秒后再次刷新")
                except Exception as e:
                    print(f"[Token] 刷新失败: {e}")

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def stop(self):
        self._stop_event.set()


token_mgr: TokenManager | None = None
BASE_API_2 = ""


@app.on_event("startup")
def startup():
    global token_mgr, BASE_API_2
    base_api_1 = os.environ.get("MINIMAX_BASE_API_1")
    base_api_2 = os.environ.get("MINIMAX_BASE_API_2")
    user = os.environ.get("MINIMAX_USER")
    password = os.environ.get("MINIMAX_PASSWORD")

    if not all([base_api_1, base_api_2, user, password]):
        raise RuntimeError(
            "请设置环境变量: MINIMAX_BASE_API_1, MINIMAX_BASE_API_2, MINIMAX_USER, MINIMAX_PASSWORD"
        )

    BASE_API_2 = base_api_2
    token_mgr = TokenManager(base_api_1, user, password)
    token_mgr.init_token()
    token_mgr.start_refresh_loop()
    print(f"[Proxy] 启动完成, 目标模型: {TARGET_MODEL}")


def _extract_messages(body: dict) -> list[dict]:
    messages = []
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
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
        messages.append({"role": role, "content": content})

    if "system" in body and body["system"]:
        system_text = body["system"]
        if isinstance(system_text, list):
            system_text = "\n".join(
                b.get("text", "") for b in system_text if b.get("type") == "text"
            )
        messages.insert(0, {"role": "system", "content": system_text})

    return messages


def _call_minimax(
    messages: list[dict], max_tokens: int, stream: bool, temperature: float | None
):
    token = token_mgr.get_token()
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
        f"[Proxy] -> MiniMax model={TARGET_MODEL}, messages={len(messages)}, stream={stream}"
    )

    if stream:
        resp = http_requests.post(
            url, json=payload, headers=headers, stream=True, timeout=180
        )
        if not resp.ok:
            print(f"[Error] MiniMax API {resp.status_code}: {resp.text[:500]}")
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return StreamingResponse(
            resp.iter_content(chunk_size=None), media_type="text/event-stream"
        )
    else:
        resp = http_requests.post(url, json=payload, headers=headers, timeout=180)
        if not resp.ok:
            print(f"[Error] MiniMax API {resp.status_code}: {resp.text[:500]}")
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


def _check_auth(authorization: str | None = None, x_api_key: str | None = None):
    key = os.environ.get("PROXY_KEY", "sk-test-key")
    if authorization:
        if authorization.replace("Bearer ", "") != key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    if x_api_key:
        if x_api_key != key:
            raise HTTPException(status_code=401, detail="Invalid API key")


# ==================== Anthropic Format (/v1/messages) ====================


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    _check_auth(
        authorization=request.headers.get("authorization"),
        x_api_key=request.headers.get("x-api-key"),
    )

    messages = _extract_messages(body)
    max_tokens = body.get("max_tokens", 4096)
    stream = body.get("stream", False)
    temperature = body.get("temperature")

    result = _call_minimax(messages, max_tokens, stream, temperature)

    if stream:
        return result

    content_text = ""
    if isinstance(result, dict):
        choices = result.get("choices", [])
        if choices:
            content_text = choices[0].get("message", {}).get("content", "")

    usage = {"input_tokens": 0, "output_tokens": 0}
    if isinstance(result, dict) and "usage" in result:
        usage["input_tokens"] = result["usage"].get("prompt_tokens", 0)
        usage["output_tokens"] = result["usage"].get("completion_tokens", 0)

    anthropic_resp = {
        "id": result.get("id", "") if isinstance(result, dict) else "",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model": body.get("model", TARGET_MODEL),
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }

    return JSONResponse(content=anthropic_resp)


# ==================== Anthropic count_tokens ====================


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    _check_auth(
        authorization=request.headers.get("authorization"),
        x_api_key=request.headers.get("x-api-key"),
    )
    return JSONResponse(content={"input_tokens": 0})


# ==================== OpenAI Format (/v1/chat/completions) ====================


@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    body = await request.json()
    _check_auth(authorization=request.headers.get("authorization"))

    messages = _extract_messages(body)
    max_tokens = body.get("max_tokens", 4096)
    stream = body.get("stream", False)
    temperature = body.get("temperature")

    result = _call_minimax(messages, max_tokens, stream, temperature)
    if stream:
        return result
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("proxy:app", host="0.0.0.0", port=4000, reload=False)
