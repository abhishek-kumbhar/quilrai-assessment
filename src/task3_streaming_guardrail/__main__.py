import logging
import os
import sys

import uvicorn

from task3_streaming_guardrail.gateway import create_app

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

provider_url = os.environ.get("PROVIDER_URL", "http://127.0.0.1:9003/v1/chat/completions")
uvicorn.run(create_app(provider_url), host="127.0.0.1", port=int(os.environ.get("GATEWAY_PORT", "8003")))
