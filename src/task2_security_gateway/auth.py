"""Bearer token handling: HS256 JWTs carrying a `role` claim.

Mint a demo token:  uv run python -m task2_security_gateway.auth admin --subject alice
"""

import argparse
import os
import time
from dataclasses import dataclass

import jwt

ROLES = {"admin", "viewer"}
DEV_SECRET = "dev-only-secret-change-me-0123456789abcdef"


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Caller:
    subject: str
    role: str


def issue_token(subject: str, role: str, secret: str, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    return jwt.encode({"sub": subject, "role": role, "iat": now, "exp": now + ttl_seconds}, secret, algorithm="HS256")


def caller_from_authorization(header: str | None, secret: str) -> Caller:
    if not header:
        raise AuthError("Missing bearer token")

    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise AuthError("Malformed Authorization header")

    try:
        # Pinning the algorithm list blocks "alg: none" and algorithm-confusion tokens.
        claims = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub", "role"]})
    except jwt.PyJWTError:
        raise AuthError("Invalid or expired token") from None

    if claims["role"] not in ROLES:
        raise AuthError("Invalid or expired token")
    return Caller(subject=str(claims["sub"]), role=claims["role"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a gateway bearer token for local testing")
    parser.add_argument("role", choices=sorted(ROLES))
    parser.add_argument("--subject", default="demo-user")
    parser.add_argument("--ttl", type=int, default=3600)
    args = parser.parse_args()
    print(issue_token(args.subject, args.role, os.environ.get("GATEWAY_JWT_SECRET", DEV_SECRET), args.ttl))


if __name__ == "__main__":
    main()
