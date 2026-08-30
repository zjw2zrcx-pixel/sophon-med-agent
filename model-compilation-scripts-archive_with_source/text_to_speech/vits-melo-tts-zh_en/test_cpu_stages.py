#!/usr/bin/env python3
"""CPU-only smoke test for the VITS duration predictor and controller."""
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import torch
ROOT = Path(__file__).parent
session = ort.InferenceSession(str(ROOT / "vits_dp_cpu_50.onnx"), providers=["CPUExecutionProvider"])
controller = torch.jit.load(str(ROOT / "cpu_dynamic_controller.pt"), map_location="cpu").eval()
hidden = np.random.randn(1, 192, 50).astype(np.float32)
speaker = np.random.randn(1, 256, 1).astype(np.float32)
x_mask = np.ones((1, 1, 50), dtype=np.float32)
start = time.perf_counter()
logw = session.run(None, {"/enc_p/encoder/Mul_3_output_0": hidden, "/Unsqueeze_6_output_0": speaker, "/enc_p/Cast_1_output_0": x_mask})[0]
with torch.inference_mode():
    z, lengths, _ = controller(torch.from_numpy(hidden), torch.zeros_like(torch.from_numpy(hidden)), torch.from_numpy(x_mask), torch.from_numpy(logw), torch.tensor([0.667]), torch.tensor([1.0]))
print({"cpu_total_ms": round((time.perf_counter() - start) * 1000, 3), "logw_shape": list(logw.shape), "z_shape": list(z.shape), "frames": int(lengths[0]), "within_decoder_bucket": bool(int(lengths[0]) <= 512)})
