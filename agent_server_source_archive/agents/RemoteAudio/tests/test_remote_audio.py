from __future__ import annotations

import asyncio
import base64
import io
import math
import struct
import unittest
import wave

from agents.RemoteAudio.operations import RemoteAudioOperations
from agents.RemoteAudio.protocol import PROTOCOL_VERSION, ProtocolError, message
from agents.RemoteAudio.tts_stream import split_text
from agents.RemoteAudio.vad import WebRtcVadStopDetector


def wav_fixture() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(b"\0\0" * 240)
    return output.getvalue()


def pcm_frame(voiced: bool, frame_ms: int = 20) -> bytes:
    samples = []
    for index in range(16000 * frame_ms // 1000):
        value = int(14000 * math.sin(2 * math.pi * 220 * index / 16000)) if voiced else 0
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


class TextAndVadTests(unittest.TestCase):
    def test_split_text_retains_boundaries_and_limit(self):
        self.assertEqual(split_text("第一句。第二句！"), ["第一句。", "第二句！"])
        self.assertTrue(all(len(value) <= 80 for value in split_text("甲" * 170)))

    def test_vad_stops_after_speech_and_silence(self):
        detector = WebRtcVadStopDetector(min_seconds=0.1, silence_seconds=0.1)
        decision = None
        for _ in range(10):
            decision = detector.feed(pcm_frame(True))
        for _ in range(20):
            decision = detector.feed(pcm_frame(False))
            if decision.should_stop:
                break
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "silence")

    def test_hello_requires_all_audio_capabilities(self):
        device_id = RemoteAudioOperations.validate_hello({
            "protocol": PROTOCOL_VERSION,
            "type": "hello",
            "device_id": "test-device",
            "capabilities": {
                "operations": ["speak", "record", "speak_and_record"],
                "playback_formats": ["wav"],
                "record_format": {
                    "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                },
            },
        })
        self.assertEqual(device_id, "test-device")


class OperationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sent = []
        self.controller = None

        async def synthesize(_text):
            return wav_fixture()

        async def send(payload):
            self.sent.append(payload)
            operation_id = payload.get("operation_id", "")
            if payload["type"] == "operation.start":
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.create_task(self.controller.handle_message(message(
                        "operation.accepted", operation_id=operation_id
                    )))
                )
            elif payload["type"] == "playback.chunk" and payload["is_last"]:
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.create_task(self.controller.handle_message(message(
                        "playback.completed", operation_id=operation_id
                    )))
                )
            elif payload["type"] == "record.stop":
                async def finish():
                    active = self.controller._active
                    await self.controller.handle_message(message(
                        "record.chunk", operation_id=operation_id,
                        seq=active.expected_record_seq,
                        is_first=False, is_last=True,
                        audio={
                            "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                            "data": base64.b64encode(pcm_frame(False)).decode(),
                        },
                    ))
                    await self.controller.handle_message(message(
                        "record.completed", operation_id=operation_id
                    ))
                asyncio.get_running_loop().call_soon(lambda: asyncio.create_task(finish()))

        self.controller = RemoteAudioOperations(
            send, synthesize, accept_timeout=1, playback_timeout=1,
            record_start_timeout=1, record_max_seconds=2, record_drain_timeout=1,
        )

    async def test_speak_streams_chunks_and_waits_for_completion(self):
        operation_id = await self.controller.speak("第一句。第二句！")
        chunks = [value for value in self.sent if value["type"] == "playback.chunk"]
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0]["is_first"])
        self.assertTrue(chunks[-1]["is_last"])
        self.assertTrue(all(value["operation_id"] == operation_id for value in chunks))
        self.assertEqual(self.sent[-1]["type"], "operation.completed")

    async def test_record_assembles_pcm_and_server_requests_stop(self):
        task = asyncio.create_task(self.controller.record())
        while not self.controller.active_operation_id:
            await asyncio.sleep(0)
        operation_id = self.controller.active_operation_id
        await asyncio.sleep(0)
        await self.controller.handle_message(message("record.started", operation_id=operation_id))
        seq = 0
        for _ in range(60):
            await self.controller.handle_message(message(
                "record.chunk", operation_id=operation_id, seq=seq,
                is_first=seq == 0, is_last=False,
                audio={
                    "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                    "data": base64.b64encode(pcm_frame(True)).decode(),
                },
            ))
            seq += 1
        for _ in range(100):
            if any(value["type"] == "record.stop" for value in self.sent):
                break
            await self.controller.handle_message(message(
                "record.chunk", operation_id=operation_id, seq=seq,
                is_first=False, is_last=False,
                audio={
                    "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                    "data": base64.b64encode(pcm_frame(False)).decode(),
                },
            ))
            seq += 1
        result = await task
        self.assertGreater(len(result.pcm), 0)
        self.assertIn(result.stop_reason, {"silence", "max_duration"})
        self.assertTrue(any(value["type"] == "record.stop" for value in self.sent))
        with wave.open(io.BytesIO(result.to_wav()), "rb") as source:
            self.assertEqual(source.getframerate(), 16000)

    async def test_combined_rejects_record_before_playback_completed(self):
        active = self.controller._new_active("speak_and_record")
        self.controller._active = active
        with self.assertRaises(ProtocolError) as caught:
            await self.controller.handle_message(message(
                "record.started", operation_id=active.operation_id
            ))
        self.assertEqual(caught.exception.code, "RECORD_STARTED_EARLY")
        self.controller._active = None

    async def test_speak_and_record_runs_both_phases_with_one_id(self):
        task = asyncio.create_task(self.controller.speak_and_record("请在播放后说话。"))
        while not self.controller.active_operation_id:
            await asyncio.sleep(0)
        operation_id = self.controller.active_operation_id
        while not self.controller._active.playback_completed.done():
            await asyncio.sleep(0)
        await self.controller.handle_message(message("record.started", operation_id=operation_id))
        for seq in range(100):
            voiced = seq < 60
            await self.controller.handle_message(message(
                "record.chunk", operation_id=operation_id, seq=seq,
                is_first=seq == 0, is_last=False,
                audio={
                    "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                    "data": base64.b64encode(pcm_frame(voiced)).decode(),
                },
            ))
            if any(value["type"] == "record.stop" for value in self.sent):
                break
        result = await task
        operation_ids = {
            value["operation_id"] for value in self.sent if value.get("operation_id")
        }
        self.assertEqual(operation_ids, {operation_id})
        self.assertEqual(result.operation_id, operation_id)

    async def test_wrong_operation_id_is_rejected(self):
        active = self.controller._new_active("speak")
        self.controller._active = active
        with self.assertRaises(ProtocolError) as caught:
            await self.controller.handle_message({
                "protocol": PROTOCOL_VERSION,
                "type": "operation.accepted",
                "operation_id": "63551ed8-d339-44a1-9b90-778910944295",
            })
        self.assertEqual(caught.exception.code, "UNKNOWN_OPERATION")
        self.controller._active = None
