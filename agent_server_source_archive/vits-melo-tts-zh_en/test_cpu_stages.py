#!/usr/bin/env python3
"""CPU-only smoke test for the VITS duration predictor and controller."""
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import torch
ROOT = Path(__file__).parent
session = ort.InferenceSession(str(ROOT / "vits_dp_cpu_50.onnx"), providers=["CPUExecutionProvider"])
controller = torch.jit.load(str(ROOT / "cpu_dynamic_controller_256f.pt"), map_location="cpu").eval()
# Use a deterministic 49-token-shaped probe.  Unbounded random hidden states
# are not a meaningful deployment test and can intentionally predict >256
# frames even when the real encoder/calibration path is within the bucket.
hidden = np.zeros((1, 192, 50), dtype=np.float32)
speaker = np.zeros((1, 256, 1), dtype=np.float32)
x_mask = np.zeros((1, 1, 50), dtype=np.float32)
x_mask[:, :, :49] = 1.0
start = time.perf_counter()
logw = session.run(None, {"/enc_p/encoder/Mul_3_output_0": hidden, "/Unsqueeze_6_output_0": speaker, "/enc_p/Cast_1_output_0": x_mask})[0]
with torch.inference_mode():
    latent_noise = torch.randn((1, 192, 256), dtype=torch.float32)
    z, y_mask, lengths, duration = controller(torch.from_numpy(hidden), torch.zeros_like(torch.from_numpy(hidden)), torch.from_numpy(x_mask), torch.from_numpy(logw), torch.tensor([0.667]), torch.tensor([1.0]), latent_noise)
print({"cpu_total_ms": round((time.perf_counter() - start) * 1000, 3), "logw_shape": list(logw.shape), "z_shape": list(z.shape), "y_mask_shape": list(y_mask.shape), "duration_shape": list(duration.shape), "frames": int(lengths[0]), "within_decoder_bucket": bool(int(lengths[0]) <= 256)})
