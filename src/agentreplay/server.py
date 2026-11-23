"""Local web server (stdlib only) exposing runs, fork-and-replay, and the UI."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Optional, Tuple

from .core import Store, replay

MAX_BODY = 1_000_000


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"))

        def _route(self) -> Tuple[str, ...]:
            return tuple(p for p in self.path.split("?")[0].split("/") if p)

        def do_GET(self) -> None:  # noqa: N802
            parts = self._route()
            if not parts:
                html = resources.files("agentreplay").joinpath("static/index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if parts == ("api", "runs"):
                return self._json(200, [r.summary() for r in store.list()])
            if len(parts) == 3 and parts[:2] == ("api", "runs"):
                try:
                    return self._json(200, store.load(parts[2]).to_dict())
                except (KeyError, ValueError):
                    return self._json(404, {"error": "run not found"})
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parts = self._route()
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "fork":
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    return self._json(413, {"error": "body too large"})
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    run = replay(parts[2], int(body["fork_at"]), body.get("edit"), store)
                except (KeyError, ValueError, TypeError) as e:
                    msg = "run not found" if isinstance(e, KeyError) and e.args == (parts[2],) else f"bad request: {e}"
                    return self._json(404 if msg == "run not found" else 400, {"error": msg})
                except LookupError as e:
                    return self._json(409, {"error": str(e)})
                return self._json(200, run.to_dict())
            self._json(404, {"error": "not found"})

    return Handler


def serve(store: Store, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(store))
