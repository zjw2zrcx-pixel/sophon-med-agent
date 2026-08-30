#!/usr/bin/env python3
"""Server-side Sophon-SAIL smoke check for the combined KWS bmodel."""
import sys
import sophon.sail as sail

path = sys.argv[1] if len(sys.argv) == 2 else "kws_transducer_chunk16_bm1684x_f32.bmodel"
engine = sail.Engine(path, 0, sail.IOMode.SYSIO)
for graph in engine.get_graph_names():
    print(graph)
    print("  inputs:", engine.get_input_names(graph))
    print("  outputs:", engine.get_output_names(graph))
