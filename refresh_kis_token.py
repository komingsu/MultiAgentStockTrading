"""Simple CLI helper to fetch Korea Investment & Securities access tokens.

The helper reuses the cached token when it was issued today, otherwise it
requests a fresh one and persists it via `kis_auth.get_or_load_access_token`.

Usage:
    python refresh_kis_token.py --env real        # default
    python refresh_kis_token.py --env mock
    python refresh_kis_token.py --env real --force
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

from kis_auth import get_or_load_access_token

EnvLiteral = Literal["real", "mock"]


def issue_token(env: EnvLiteral = "real", force_refresh: bool = False) -> str:
    """Return today's token for the requested environment."""
    return get_or_load_access_token(env=env, force_refresh=force_refresh)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Issue or reuse a cached KIS access token."
    )
    parser.add_argument(
        "--env",
        choices=("real", "mock"),
        default="real",
        help="Target environment",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cache and force a fresh token request",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    token = issue_token(env=args.env, force_refresh=args.force)
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

