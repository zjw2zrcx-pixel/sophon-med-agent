#!/usr/bin/env python3
"""Compare encoder bmodel outputs with supplied ONNX golden vectors; no ORT needed."""
from pathlib import Path
import json
import sys
import numpy as np
from sophon import sail

ROOT = Path(__file__).parent
GOLDEN = ROOT / "golden_vectors" / "encoder_contract_50"
meta = json.loads((GOLDEN / "manifest.json").read_text())
expected = np.load(GOLDEN / "encoder_outputs.npz")
engine = sail.Engine(str(ROOT / "vits_encoder_50_bm1684x_f32.bmodel"), 0, sail.IOMode.SYSIO)
graph = engine.get_graph_names()[0]
inputs = {
    "x": np.load(GOLDEN / "x.npy").astype(np.int32),
    "x_lengths": np.load(GOLDEN / "x_lengths.npy").astype(np.int32),
    "tones": np.load(GOLDEN / "tones.npy").astype(np.int32),
    "sid": np.load(GOLDEN / "sid.npy").astype(np.int32),
}
actual = engine.process(graph, inputs)
atol, rtol = meta["tolerance"]["atol"], meta["tolerance"]["rtol"]
failed = False
for name in meta["bmodel_output_names"]:
    got, ref = actual[name], expected[name]
    maximum = float(np.max(np.abs(got - ref)))
    ok = np.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"{name}: shape={list(got.shape)} max_abs={maximum:.8g} {'OK' if ok else 'FAIL'}")
    failed |= not ok
sys.exit(1 if failed else 0)
