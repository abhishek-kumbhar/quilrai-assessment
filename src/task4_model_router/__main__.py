import logging
import os
import sys

import httpx
import uvicorn

from task4_model_router.app import create_app
from task4_model_router.rate_limiter import DEFAULT_TOKENS_PER_MINUTE, TokenRateLimiter
from task4_model_router.router import ModelRouter, Provider

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

env = os.environ.get
primary = Provider("primary", env("PRIMARY_URL", "http://127.0.0.1:9004/v1/chat/completions"), env("PRIMARY_MODEL", "primary-model"), env("PRIMARY_API_KEY", ""))
secondary = Provider("secondary", env("SECONDARY_URL", "http://127.0.0.1:9005/v1/chat/completions"), env("SECONDARY_MODEL", "backup-model"), env("SECONDARY_API_KEY", ""))

limiter = TokenRateLimiter(
    env("RATE_LIMIT_DB", "rate_limits.db"),
    limit_tokens=int(env("TOKENS_PER_MINUTE", str(DEFAULT_TOKENS_PER_MINUTE))),
)
router = ModelRouter(primary, secondary, httpx.AsyncClient(timeout=30.0))
tenant_keys = {key.strip() for key in env("TENANT_API_KEYS", "tenant-a-key,tenant-b-key").split(",") if key.strip()}

uvicorn.run(create_app(limiter, router, tenant_keys), host="127.0.0.1", port=int(env("GATEWAY_PORT", "8004")))
