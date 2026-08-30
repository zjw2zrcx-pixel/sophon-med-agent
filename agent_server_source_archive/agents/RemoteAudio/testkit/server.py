"""Agent-side protocol simulator for device developers."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import struct
import tempfile
import time
import wave
from pathlib import Path

from agents.RemoteAudio.operations import RemoteAudioOperations
from agents.RemoteAudio.protocol import ProtocolError, message, parse_json


def _tone_wav(text: str) -> bytes:
    """Last-resort audible fixture when no local TTS executable is installed."""
    sample_rate = 24000
    duration = max(0.5, min(3.0, len(text) * 0.08))
    frames = bytearray()
    for index in range(int(sample_rate * duration)):
        envelope = min(1.0, index / 300, (sample_rate * duration - index) / 300)
        value = int(9000 * envelope * math.sin(2 * math.pi * 523.25 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    output = tempfile.SpooledTemporaryFile()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(frames)
    output.seek(0)
    return output.read()


async def synthesize_local(text: str) -> bytes:
    executable = shutil.which("espeak-ng")
    if not executable:
        return _tone_wav(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as target:
        path = Path(target.name)
    try:
        argv = [executable, "-v", "cmn", "-w", str(path), text]
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        if process.returncode != 0 or path.stat().st_size < 44:
            raise RuntimeError(stderr.decode(errors="replace")[:300])
        return path.read_bytes()
    except Exception:
        return _tone_wav(text)
    finally:
        path.unlink(missing_ok=True)


class DeviceTestSession:
    def __init__(self, websocket, output_dir: Path, operation: str, text: str):
        self.websocket = websocket
        self.output_dir = output_dir
        self.operation = operation
        self.text = text
        self.events: list[dict] = []
        self.controller: RemoteAudioOperations | None = None
        self.done = asyncio.Event()

    async def send(self, payload: dict):
        self.events.append({"direction": "to_device", "time": time.time(), "message": self._report_message(payload)})
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _report_message(payload: dict) -> dict:
        value = json.loads(json.dumps(payload, ensure_ascii=False))
        audio = value.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("data"), str):
            raw = base64.b64decode(audio["data"])
            audio["data"] = f"<省略 {len(raw)} bytes>"
            audio["sha256"] = hashlib.sha256(raw).hexdigest()
        return value

    async def receive_loop(self):
        try:
            async for raw in self.websocket:
                value = parse_json(raw)
                self.events.append({"direction": "from_device", "time": time.time(), "message": self._report_message(value)})
                try:
                    await self.controller.handle_message(value)
                except ProtocolError as exc:
                    self.controller.fail_protocol(exc)
                    await self.send(message(
                        "protocol.error", operation_id=exc.operation_id,
                        error={"code": exc.code, "message": str(exc), "retryable": False},
                    ))
        finally:
            if self.controller:
                await self.controller.close()

    def save_recording(self, result, label: str):
        path = self.output_dir / f"{label}_{result.operation_id}.wav"
        path.write_bytes(result.to_wav())
        return path

    async def run(self):
        first = parse_json(await asyncio.wait_for(self.websocket.recv(), timeout=15))
        device_id = RemoteAudioOperations.validate_hello(first)
        self.events.append({"direction": "from_device", "time": time.time(), "message": self._report_message(first)})
        await self.send(RemoteAudioOperations.hello_ack(device_id))
        self.controller = RemoteAudioOperations(self.send, synthesize_local)
        receiver = asyncio.create_task(self.receive_loop())
        results: list[dict] = []
        status = "passed"
        error = ""
        try:
            operations = [self.operation] if self.operation != "all" else [
                "speak", "record", "speak_and_record"
            ]
            for operation in operations:
                started = time.time()
                if operation == "speak":
                    operation_id = await self.controller.speak(self.text)
                    results.append({"operation": operation, "operation_id": operation_id})
                elif operation == "record":
                    print("[record] 请对设备麦克风说话，静音 1.5 秒后自动结束。", flush=True)
                    result = await self.controller.record()
                    path = self.save_recording(result, "record")
                    results.append({
                        "operation": operation, "operation_id": result.operation_id,
                        "duration_seconds": result.duration_seconds, "output": str(path),
                    })
                else:
                    print("[speak_and_record] 播放结束后请立即说话。", flush=True)
                    result = await self.controller.speak_and_record(self.text)
                    path = self.save_recording(result, "speak_and_record")
                    results.append({
                        "operation": operation, "operation_id": result.operation_id,
                        "duration_seconds": result.duration_seconds, "output": str(path),
                    })
                results[-1]["elapsed_seconds"] = time.time() - started
                print(f"[{operation}] PASS {results[-1]}", flush=True)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            print(f"测试失败: {error}", flush=True)
        finally:
            report = {
                "protocol": "remote-audio.v1",
                "device_id": device_id,
                "status": status,
                "error": error,
                "results": results,
                "events": self.events,
            }
            report_path = self.output_dir / f"report_{int(time.time())}.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"报告: {report_path}", flush=True)
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            await self.websocket.close()
            self.done.set()


async def serve(args):
    try:
        from websockets.asyncio.server import serve as websocket_serve
    except ImportError as exc:
        raise RuntimeError("测试包需要 websockets（/data/env310 已安装）") from exc

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = asyncio.Event()
    active = False

    async def handler(websocket):
        nonlocal active
        if active:
            await websocket.close(code=1013, reason="已有设备正在测试")
            return
        active = True
        session = DeviceTestSession(websocket, output_dir, args.operation, args.text)
        try:
            await session.run()
        finally:
            completed.set()

    async with websocket_serve(handler, args.bind, args.port, max_size=16 * 1024 * 1024):
        print(f"等待设备连接: ws://{args.bind}:{args.port}/ws", flush=True)
        print("测试包模拟 Agent；请在待测设备上配置上述地址。", flush=True)
        await completed.wait()


def main():
    parser = argparse.ArgumentParser(description="remote-audio.v1 局域网设备适配测试包")
    parser.add_argument("--bind", default="0.0.0.0", help="监听 IP")
    parser.add_argument("--port", default=9876, type=int, help="监听端口")
    parser.add_argument(
        "--operation", default="all",
        choices=["speak", "record", "speak_and_record", "all"],
    )
    parser.add_argument("--text", default="远程音频协议测试。播放结束后，请说出测试成功。")
    parser.add_argument("--output-dir", default="./remote-audio-results")
    args = parser.parse_args()
    asyncio.run(serve(args))
