#!/usr/bin/env python3
"""Numerically compare source ONNX encoder values with the TPU encoder bmodel."""
from pathlib import Path
import sys, tempfile
import numpy as np
import onnx, onnxruntime as ort
from onnx.utils import extract_model

ROOT = Path(__file__).resolve().parent
# Optional compiler-side source model; production deployment only needs the
# bmodels and does not ship the original full ONNX graph.
SOURCE = ROOT / 'model.onnx'
sys.path.insert(0, str(ROOT / 'runtime' / 'melotts' / 'melo'))
from text import chinese
sys.path.insert(0, str(ROOT))
from hybrid_vits_runtime import HybridVitsRuntime

# Original ONNX names are pre-TPU-MLIR; the bmodel appends op suffixes.
names = ['/enc_p/encoder/Mul_3_output_0', '/enc_p/Split_output_0', '/enc_p/Split_output_1', '/enc_p/Unsqueeze_2_output_0', '/Unsqueeze_6_output_0', '/enc_p/encoder/Transpose_output_0']
bmodel_names = ['/enc_p/encoder/Mul_3_output_0_Mul', '/enc_p/Split_output_0_Split', '/enc_p/Split_output_1_Split', '/enc_p/Unsqueeze_2_output_0_Unsqueeze', '/Unsqueeze_6_output_0_Unsqueeze', '/enc_p/encoder/Transpose_output_0_Transpose']
tmp = Path(tempfile.gettempdir()) / 'vits_encoder_probe.onnx'
# Extracting only the encoder avoids allocating the complete flow/decoder graph.
extract_model(str(SOURCE), str(tmp), ['x', 'x_lengths', 'tones', 'sid'], names)
phones, tones, _ = chinese.g2p(chinese.text_normalize('您好,我正在为您服务.'))
token = {line.rsplit(maxsplit=1)[0]: int(line.rsplit(maxsplit=1)[1]) for line in (ROOT / 'preprocess_assets' / 'tokens.txt').read_text().splitlines()}
x = np.array([token[p] for p in phones], np.int64); t = np.array(tones, np.int64); n = len(x)
xp = np.zeros((1, 50), np.int64); tp = np.zeros((1, 50), np.int64); xp[0, :n] = x; tp[0, :n] = t
ref = ort.InferenceSession(tmp, providers=['CPUExecutionProvider']).run(names, {'x': xp, 'x_lengths': np.array([n], np.int64), 'tones': tp, 'sid': np.array([0], np.int64)})
rt = HybridVitsRuntime(ROOT, 0)
out = rt._run(rt.encoder, rt.encoder_graph, {'x': xp.astype(np.int32), 'x_lengths': np.array([n], np.int32), 'tones': tp.astype(np.int32), 'sid': np.array([0], np.int32)})
for name, bname, value in zip(names, bmodel_names, ref):
    got = out[bname]; d = np.abs(value.astype(np.float32) - got)
    print(name, {'shape': list(value.shape), 'max_abs': float(d.max()), 'mean_abs': float(d.mean())})
