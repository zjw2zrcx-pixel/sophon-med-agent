#!/usr/bin/env python3
"""End-to-end hybrid golden comparison using SAIL; original ONNX is not needed."""
from pathlib import Path
import json
import argparse
import sys
import numpy as np
import onnxruntime as ort
import torch
from sophon import sail

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--golden", default="hello_static50", help="directory below golden_vectors")
args = parser.parse_args()
G = ROOT / "golden_vectors" / args.golden
meta = json.loads((G / "manifest.json").read_text())
encoder = sail.Engine(str(ROOT / "vits_encoder_50_bm1684x_f32.bmodel"), 0, sail.IOMode.SYSIO)
decoder = sail.Engine(str(ROOT / "vits_flow_decoder_256_bm1684x_f32.bmodel"), 0, sail.IOMode.SYSIO)
eg, dg = encoder.get_graph_names()[0], decoder.get_graph_names()[0]
inputs = {n: np.load(G / f"{n}.npy").astype(np.int32) for n in ("x", "x_lengths", "tones", "sid")}
e = encoder.process(eg, inputs)
hidden = e["/enc_p/encoder/Mul_3_output_0_Mul"]
m_p = e["/enc_p/Split_output_0_Split"]
logs_p = e["/enc_p/Split_output_1_Split"]
x_mask = e["/enc_p/Unsqueeze_2_output_0_Unsqueeze"]
speaker = e["/Unsqueeze_6_output_0_Unsqueeze"]
condition = e["/enc_p/encoder/Transpose_output_0_Transpose"]
dp = ort.InferenceSession(str(ROOT / "vits_dp_cpu_50.onnx"), providers=["CPUExecutionProvider"])
logw = dp.run(None, {"/enc_p/encoder/Mul_3_output_0": hidden, "/Unsqueeze_6_output_0": speaker, "/enc_p/Cast_1_output_0": x_mask})[0]
torch.manual_seed(meta.get("seed", meta.get("torch_seed", 20260727)))
controller = torch.jit.load(str(ROOT / "cpu_dynamic_controller_256f.pt"), map_location="cpu").eval()
latent_noise = torch.randn((1, 192, 256), dtype=torch.float32)
with torch.inference_mode():
    z, ymask, lengths, _ = controller(torch.from_numpy(m_p), torch.from_numpy(logs_p), torch.from_numpy(x_mask), torch.from_numpy(logw), torch.tensor([0.667]), torch.tensor([1.0]), latent_noise)
frames = int(lengths[0])
if frames > 256:
    raise ValueError(f"golden sample predicts {frames} frames, over 256-frame bucket")
zpad = z.numpy().astype(np.float32); ymask = ymask.numpy().astype(np.float32)
out = decoder.process(dg, {"/Add_2_output_0": zpad, "/Cast_4_output_0": ymask, "/Unsqueeze_10_output_0": ymask[..., None], "/enc_p/encoder/Transpose_output_0": condition, "/Unsqueeze_6_output_0": speaker})["y_Tanh"]
actual = out[0, 0, :frames * 512]
reference = np.load(G / "wav_reference.npy")
maximum = float(np.max(np.abs(actual - reference)))
ok = actual.shape == reference.shape and np.allclose(actual, reference, atol=1e-4, rtol=1e-4)
print({"text": meta["text"], "frames": frames, "samples": len(actual), "max_abs": maximum, "result": "OK" if ok else "FAIL"})
sys.exit(0 if ok else 1)
