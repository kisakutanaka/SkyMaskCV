"""bench.html 用の配信サーバ。WebAppUtils 配下を配信し、POST /result を保存する。

ヘッドレスのブラウザから結果を取り出す手段が必要なだけの、使い捨てに近いもの。
使い方: python3 tools/bench_server.py <ポート> <結果の保存先>
"""
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT, OUT = int(sys.argv[1]), sys.argv[2]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        open(OUT, 'wb').write(body)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
