#!/usr/bin/env python3
"""
简单 HTTP 抓包代理 — 明文存储请求/响应到本地文件。
监听 0.0.0.0:8000，转发到 UPSTREAM (默认 127.0.0.1:8001)。

用法:
  python sniff.py
  python sniff.py --upstream http://192.168.10.2:8001 --port 8000
"""

import argparse
import socket
import asyncio
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse

DUMP_DIR = Path("sniff_logs")
DUMP_DIR.mkdir(exist_ok=True)
UPSTREAM = "http://127.0.0.1:8001"
_cnt = 0
_sniff_client: httpx.AsyncClient | None = None


def _get_client():
    global _sniff_client
    if _sniff_client is None:
        _sniff_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    return _sniff_client


app = FastAPI()


def _next_id():
    global _cnt
    _cnt += 1
    return f"{time.strftime('%H%M%S')}_{_cnt:04d}"


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    rid = _next_id()
    body = await request.body()
    url_path = f"/{path}"

    # 保存请求明文
    (DUMP_DIR / f"{rid}_req.txt").write_bytes(body)

    # 转发
    upstream_url = f"{UPSTREAM}{url_path}"
    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "transfer-encoding")}
    client = _get_client()

    try:
        # 判断是否流式
        is_stream = b'"stream"' in body[:200] and b'true' in body[:200]

        if is_stream:
            resp_file = DUMP_DIR / f"{rid}_resp.txt"
            f = open(resp_file, "w")

            async def stream_and_log():
                try:
                    async with client.stream(request.method, upstream_url, headers=fwd_headers, content=body) as resp:
                        async for line in resp.aiter_lines():
                            yield line + "\n"
                            f.write(line + "\n")
                except Exception as e:
                    print(f"[{rid}] STREAM ERROR: {e}")
                finally:
                    f.close()
                    print(f"[{rid}] stream done -> {resp_file.name}")

            return StreamingResponse(stream_and_log(), media_type="text/event-stream")

        else:
            resp = await client.request(request.method, upstream_url, headers=fwd_headers, content=body)
            (DUMP_DIR / f"{rid}_resp.txt").write_bytes(resp.content)
            print(f"[{rid}] {resp.status_code} -> {rid}_resp.txt ({len(resp.content)}B)")
            return Response(content=resp.content, status_code=resp.status_code,
                            headers={k: v for k, v in resp.headers.items()
                                     if k.lower() not in ("content-encoding", "transfer-encoding")})
    except Exception as e:
        print(f"[{rid}] ERROR: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)


def main():
    global UPSTREAM
    parser = argparse.ArgumentParser(description="HTTP sniff proxy")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    UPSTREAM = args.upstream.rstrip("/")

    print(f"Sniff proxy 0.0.0.0:{args.port} -> {UPSTREAM}")
    print(f"Logs -> {DUMP_DIR.resolve()}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    sock.bind(("0.0.0.0", args.port))
    sock.listen(128)
    sock.setblocking(False)

    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


if __name__ == "__main__":
    main()
