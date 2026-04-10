# MiniMax M2.5 Proxy

基于 LiteLLM 的代理服务，将 MiniMax M2.5 模型 API 转换为 OpenAI 和 Anthropic 兼容格式，可直接对接 Claude Code 等客户端。

## 功能

- 自动获取并定时刷新认证 Token（每小时刷新一次）
- 支持 OpenAI 格式 `/v1/chat/completions`
- 支持 Anthropic 格式 `/v1/messages`（兼容 Claude Code）
- 所有请求统一转发至 MiniMax `/api/v2/chat/completions`
- 支持 stream 流式输出
- 基于 LiteLLM 代理服务器，内置日志、重试、限流等生产特性

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
MINIMAX_BASE_API_1=http://your-base-api-1
MINIMAX_BASE_API_2=http://your-base-api-2
MINIMAX_USER=your-username
MINIMAX_PASSWORD=your-password
PROXY_KEY=sk-test-key
```

| 变量 | 说明 |
|------|------|
| `MINIMAX_BASE_API_1` | 登录 API 地址 |
| `MINIMAX_BASE_API_2` | 模型 API 地址 |
| `MINIMAX_USER` | 用户名 |
| `MINIMAX_PASSWORD` | 密码 |
| `PROXY_KEY` | 代理 API Key（默认 `sk-test-key`） |
| `TARGET_MODEL` | 目标模型名（默认 `MiniMax-M2.5`） |
| `PORT` | 代理端口（默认 `4000`） |

### 3. 启动 Proxy

```bash
python entrypoint.py
```

启动后监听 `http://0.0.0.0:4000`。

### 4. 调用 API

**OpenAI 格式**

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 4096
  }'
```

**Anthropic 格式**

```bash
curl http://localhost:4000/v1/messages \
  -H "Authorization: Bearer sk-test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 4096
  }'
```

> 所有请求的模型名都会被映射为实际的 `MiniMax-M2.5` 模型，客户端传入的模型名仅作标识。

**对接 Claude Code**

将 API Base URL 配置为 `http://localhost:4000`，API Key 配置为 `sk-test-key` 即可。

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容格式 |
| `/v1/messages` | POST | Anthropic 格式 |

## 项目结构

```
.
├── entrypoint.py         # 启动入口
├── minimax_provider.py   # MiniMax 自定义 LiteLLM Provider
├── litellm_config.yaml   # LiteLLM 代理配置
├── .env.example          # 环境变量模板
├── requirements.txt      # 依赖
└── README.md
```

## 停止服务

按 `Ctrl+C` 停止。
