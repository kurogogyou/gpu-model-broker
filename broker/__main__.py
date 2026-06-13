"""Entrypoint: `python -m broker`.

Reads broker host/port from BROKER_HOST / BROKER_PORT env vars.
Defaults: 127.0.0.1:8090 (localhost-only; the broker is a host-local service).
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("BROKER_HOST", "127.0.0.1")
    port = int(os.environ.get("BROKER_PORT", "8090"))
    uvicorn.run(
        "broker.app:app",
        host=host,
        port=port,
        log_level=os.environ.get("BROKER_LOG_LEVEL", "info"),
        access_log=False,  # too chatty; broker only logs lifecycle events
    )


if __name__ == "__main__":
    main()
