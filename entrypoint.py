#!/usr/bin/env python3
"""
LiteLLM-based MiniMax Proxy Entrypoint
Pre-initializes the MiniMax token manager, then embeds LiteLLM proxy.
"""

import os
import asyncio
import signal

import litellm
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Pre-initialize MiniMax provider (triggers TokenManager setup)
import minimax_provider  # noqa: F401

# Register custom provider programmatically (more reliable than config file)
litellm.custom_provider_map = [
    {
        "provider": "custom_minimax",
        "custom_handler": minimax_provider.minimax_custom_auth,
    }
]

# Register the custom provider in LiteLLM's provider list
from litellm.utils import custom_llm_setup

custom_llm_setup()

PORT = int(os.environ.get("PORT", "4000"))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litellm_config.yaml")


def main():
    from litellm.proxy.proxy_server import initialize, app

    print(f"[Entrypoint] Loading config: {CONFIG_PATH}")
    print(f"[Entrypoint] Starting LiteLLM proxy on port {PORT}")

    # Initialize proxy with config
    asyncio.get_event_loop().run_until_complete(initialize(config=CONFIG_PATH))

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Handle graceful shutdown
    def handle_signal(signum, frame):
        print(f"\n[Entrypoint] Received signal {signum}, shutting down...")
        minimax_provider.token_manager.stop()
        server.should_exit = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    asyncio.get_event_loop().run_until_complete(server.serve())
    print("[Entrypoint] Proxy stopped.")


if __name__ == "__main__":
    main()
