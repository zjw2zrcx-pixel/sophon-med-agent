#!/usr/bin/env python3
"""Print BM1684X bmodel graph input/output metadata."""
from pathlib import Path
from sophon import sail
for name in ("vits_encoder_50_bm1684x_f32.bmodel", "vits_flow_decoder_512_bm1684x_f32.bmodel"):
    engine = sail.Engine(str(Path(__file__).parent / name), 0, sail.IOMode.SYSIO)
    graph = engine.get_graph_names()[0]
    print(f"\n{name}: {graph}")
    for item in engine.get_input_names(graph): print(" input ", item, engine.get_input_shape(graph, item))
    for item in engine.get_output_names(graph): print(" output", item, engine.get_output_shape(graph, item))
