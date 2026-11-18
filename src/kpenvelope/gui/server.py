"""A small local web server for the kpenvelope app (standard library only).

It listens on 127.0.0.1 only, serves one page, and answers POST
/api/<action> with the JSON of `kpenvelope.gui.api.handle`. Each run makes
a random token that the page sends back with every request, so other web
pages open in the same browser cannot use the server.
"""
from __future__ import annotations

import http.server
import json
import secrets
import threading
import webbrowser
from importlib import resources

from . import api

__all__ = ["make_server", "main"]

_MAX_BODY = 2_000_000


def _page(token):
    html = resources.files("kpenvelope.gui").joinpath("app.html").read_text(
        encoding="utf-8")
    return html.replace("__KPE_TOKEN__", token)


def make_server(port=0, token=None):
    """An HTTP server on 127.0.0.1:port (0 picks a free port); returns
    (server, token). Call server.serve_forever() to run it."""
    token = token or secrets.token_urlsafe(24)
    page = _page(token).encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "kpenvelope-app"

        def log_message(self, *args):        # keep the terminal quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/api/"):
                return self._send(404, b"not found", "text/plain")
            if self.headers.get("X-Token") != token:
                return self._send(403, b"forbidden", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if not (0 <= n <= _MAX_BODY):
                return self._send(413, b"too large", "text/plain")
            try:
                spec = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, b"bad json", "text/plain")
            out = api.handle(self.path[len("/api/"):], spec)
            body = json.dumps(out, allow_nan=False).encode("utf-8")
            self._send(200, body, "application/json")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv, token


def main(argv=None):
    """Start the app and open it in the default browser."""
    import argparse

    ap = argparse.ArgumentParser(prog="kpenvelope-app",
                                 description="Open the kpenvelope app in "
                                             "your web browser.")
    ap.add_argument("--port", type=int, default=0,
                    help="port to use (default: any free port)")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser; just print the address")
    a = ap.parse_args(argv)
    srv, _ = make_server(a.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"kpenvelope app running at {url}  (press Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
