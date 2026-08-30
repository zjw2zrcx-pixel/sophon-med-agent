#!/usr/bin/env python3
"""Real-time detection of the Chinese keywords: 小麦 and 小麦小麦.

The keyword stream intentionally stays alive after the first 小麦.  Therefore
both a connected pronunciation ("小麦小麦") and a short pause between the two
occurrences ("小麦 … 小麦") can finish the same keyword sequence.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import sherpa_onnx

try:
    import sounddevice as sd
except ImportError as e:
    raise SystemExit("缺少 sounddevice，请在 jcs 环境执行：pip install sounddevice") from e


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"


def parse_args():
    parser = argparse.ArgumentParser(
        description="通过麦克风检测“小麦小麦”（连读或短暂停顿均可）。"
    )
    parser.add_argument("--list-devices", action="store_true", help="列出输入设备并退出")
    parser.add_argument(
        "--device", default=None, help="输入设备 ID 或名称；默认使用系统默认输入设备"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "sounddevice", "pulse"),
        default="auto",
        help="采集后端。WSLg/PulseAudio 下推荐 pulse",
    )
    parser.add_argument(
        "--pulse-source",
        default="@DEFAULT_SOURCE@",
        help="PulseAudio 输入源，例如 RDPSource",
    )
    parser.add_argument("--threshold", type=float, default=0.18, help="触发阈值")
    parser.add_argument("--score", type=float, default=2.0, help="关键词加分")
    parser.add_argument("--threads", type=int, default=2, help="推理线程数")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.012,
        help="能量 VAD 的 RMS 阈值；设为 0 可关闭",
    )
    parser.add_argument(
        "--vad-silence-frames",
        type=int,
        default=8,
        help="连续多少个 100ms 静音帧后结束一次语音段",
    )
    parser.add_argument(
        "--vad-debug",
        action="store_true",
        help="每秒打印一次当前 RMS；用于排查没有检测到语音",
    )
    return parser.parse_args()


def list_devices():
    try:
        devices = sd.query_devices()
    except Exception as e:
        raise SystemExit(f"无法读取音频设备：{e}") from e
    found = False
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            found = True
            print(f"[{index}] {device['name']} (输入声道: {device['max_input_channels']})")
    if not found:
        print("sounddevice/ALSA 未发现输入设备。")
    if shutil.which("pactl"):
        print("\nPulseAudio 输入源：")
        try:
            subprocess.run(["pactl", "list", "short", "sources"], check=False)
        except OSError as e:
            print(f"无法运行 pactl：{e}")


def require_files():
    files = {
        "tokens": MODEL / "tokens.txt",
        "encoder": MODEL / "encoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "decoder": MODEL / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "joiner": MODEL / "joiner-epoch-13-avg-2-chunk-16-left-64.onnx",
        "keywords": ROOT / "keywords.txt",
        "single_keywords": ROOT / "keywords_single.txt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("缺少所需文件：\n" + "\n".join(missing))
    return files


def create_spotter(files, keyword_key, args):
    return sherpa_onnx.KeywordSpotter(
        tokens=str(files["tokens"]),
        encoder=str(files["encoder"]),
        decoder=str(files["decoder"]),
        joiner=str(files["joiner"]),
        keywords_file=str(files[keyword_key]),
        keywords_score=args.score,
        keywords_threshold=args.threshold,
        num_threads=args.threads,
        max_active_paths=4,
        num_trailing_blanks=1,
        provider="cpu",
    )


def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return

    files = require_files()
    backend = args.backend
    if backend == "auto":
        try:
            backend = (
                "sounddevice"
                if any(d["max_input_channels"] > 0 for d in sd.query_devices())
                else "pulse"
            )
        except Exception:
            backend = "pulse"
    if backend == "sounddevice":
        try:
            if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
                raise SystemExit("sounddevice 未检测到麦克风；可改用 --backend pulse。")
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"无法读取 sounddevice 输入设备：{e}") from e
    elif shutil.which("parec") is None:
        raise SystemExit("未找到 parec，无法使用 PulseAudio 采集。")

    # Keep two separate streams. Resetting the short keyword must not erase
    # progress of the complete 小麦小麦 phrase.
    phrase_spotter = create_spotter(files, "keywords", args)
    single_spotter = create_spotter(files, "single_keywords", args)

    sample_rate = 16000
    samples_per_read = sample_rate // 10
    phrase_stream = phrase_spotter.create_stream()
    single_stream = single_spotter.create_stream()
    in_speech = False
    silence_frames = 0
    vad_frames = 0

    def process_samples(samples):
        nonlocal phrase_stream, single_stream, in_speech, silence_frames, vad_frames
        rms = float((samples.astype("float64") ** 2).mean() ** 0.5)
        vad_frames += 1
        if args.vad_debug and vad_frames % 10 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] VAD RMS={rms:.4f}, "
                f"阈值={args.vad_threshold:.4f}"
            )
        if args.vad_threshold > 0:
            if rms >= args.vad_threshold:
                if not in_speech:
                    print(f"[{time.strftime('%H:%M:%S')}] 检测到语音 (RMS={rms:.4f})")
                in_speech = True
                silence_frames = 0
            elif in_speech:
                silence_frames += 1
                if silence_frames >= args.vad_silence_frames:
                    print(f"[{time.strftime('%H:%M:%S')}] 语音结束")
                    in_speech = False
                    silence_frames = 0

        for spotter, stream_name, label in (
            (single_spotter, "single", "单关键词"),
            (phrase_spotter, "phrase", "完整关键词"),
        ):
            stream = single_stream if stream_name == "single" else phrase_stream
            stream.accept_waveform(sample_rate, samples)
            while spotter.is_ready(stream):
                spotter.decode_stream(stream)
                result = spotter.get_result(stream)
                if result:
                    print(f"[{time.strftime('%H:%M:%S')}] {label}检测到：{result}")
                    if stream_name == "single":
                        single_stream = spotter.create_stream()
                    else:
                        phrase_stream = spotter.create_stream()
    print(f"开始监听（{backend}）：请说“小麦小麦”，也可说“小麦 … 小麦”。按 Ctrl+C 停止。")
    if backend == "pulse":
        print(
            "Pulse 输入诊断："
            f"PULSE_SERVER={os.environ.get('PULSE_SERVER', '(未设置)')}，"
            f"source={args.pulse_source}"
        )
    try:
        if backend == "sounddevice":
            with sd.InputStream(
                device=args.device,
                channels=1,
                dtype="float32",
                samplerate=sample_rate,
            ) as mic:
                while True:
                    samples, _ = mic.read(samples_per_read)
                    process_samples(samples.reshape(-1))
        else:
            import numpy as np

            command = [
                "parec", "--raw", "--format=float32le", "--rate=16000", "--channels=1",
                f"--device={args.pulse_source}",
            ]
            with subprocess.Popen(command, stdout=subprocess.PIPE) as recorder:
                assert recorder.stdout is not None
                while True:
                    raw = recorder.stdout.read(samples_per_read * 4)
                    if len(raw) != samples_per_read * 4:
                        raise RuntimeError("PulseAudio 录音流意外结束")
                    process_samples(np.frombuffer(raw, dtype="<f4"))
    except KeyboardInterrupt:
        print("\n已停止监听。")


if __name__ == "__main__":
    main()
