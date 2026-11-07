import logging
import os
import sys

import uvicorn

from task2_security_gateway.auth import DEV_SECRET
from task2_security_gateway.gateway import create_app

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

secret = os.environ.get("GATEWAY_JWT_SECRET")
if not secret:
    logging.warning("GATEWAY_JWT_SECRET not set, using the built-in development secret")
    secret = DEV_SECRET

downstream_url = os.environ.get("DOWNSTREAM_URL", "http://127.0.0.1:9002/mcp")
uvicorn.run(create_app(downstream_url, secret), host="127.0.0.1", port=int(os.environ.get("GATEWAY_PORT", "8002")))
