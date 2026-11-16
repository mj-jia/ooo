"""带 COOP/COEP 头的本地服务器。

开这两个头之后浏览器才会启用 SharedArrayBuffer，onnxruntime-web 才能多线程，
WASM 推理速度能快 3~4 倍。GitHub Pages 设不了这两个头，所以线上比本地慢是正常的。

用法：
    python serve.py            # 默认 8123
    python serve.py 8000
"""

from __future__ import annotations

import http.server
import socketserver
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"http://127.0.0.1:{PORT}/  (多线程已启用)")
        httpd.serve_forever()
