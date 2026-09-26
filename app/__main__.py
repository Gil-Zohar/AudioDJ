"""Run the server using HOST/PORT from .env: `python -m app`.

Equivalent to calling uvicorn directly; this just saves repeating the flags.
Set HOST=0.0.0.0 in .env to reach it from a phone on the same network.
"""
from __future__ import annotations

import socket

import uvicorn

from app.config import get_settings


def local_ip() -> str:
    """Best guess at this machine's LAN address, for the 'open on your phone' hint."""
    try:
        # No packets are sent; this just asks the OS which interface would be used.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 1))   # TEST-NET-1, guaranteed unroutable
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def main() -> None:
    settings = get_settings()
    print(f"  AutoDJ  ->  http://127.0.0.1:{settings.port}")
    if settings.host in ("0.0.0.0", "::"):
        print(f"  on your phone (same Wi-Fi)  ->  http://{local_ip()}:{settings.port}")
        print("  NOTE: no password. Anyone on this network can browse your library.")
    else:
        print("  (set HOST=0.0.0.0 in .env to reach it from your phone)")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
