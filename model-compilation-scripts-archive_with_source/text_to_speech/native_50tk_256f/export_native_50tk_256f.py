#!/usr/bin/env python3
"""Create native 50-token / 256-frame VITS split assets from canonical model.onnx."""
from pathlib import Path
import argparse
import onnx
from onnx import utils
import torch
from typing import Tuple

FRAMES = 256
DECODER_INPUTS = [
    "/Add_2_output_0", "/Cast_4_output_0", "/Unsqueeze_10_output_0",
    "/enc_p/encoder/Transpose_output_0", "/Unsqueeze_6_output_0",
]
DECODER_SHAPES = [[1, 192, FRAMES], [1, 1, FRAMES], [1, 1, FRAMES, 1], [1, 1, 256], [1, 256, 1]]

class StaticVitsController256(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_frames = FRAMES
        self.register_buffer("positions", torch.arange(FRAMES, dtype=torch.int64).view(1, 1, FRAMES, 1))

    def forward(self, m_p: torch.Tensor, logs_p: torch.Tensor, x_mask: torch.Tensor,
                logw: torch.Tensor, noise_scale: torch.Tensor, length_scale: torch.Tensor,
                latent_noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        duration = torch.ceil(torch.exp(logw) * x_mask * length_scale).to(torch.int64)
        y_lengths = torch.clamp(duration.sum((1, 2)), min=1)
        ends = torch.cumsum(duration, 2).unsqueeze(2)
        starts = (torch.cumsum(duration, 2) - duration).unsqueeze(2)
        attn = ((self.positions >= starts) & (self.positions < ends)).to(m_p.dtype)
        m = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        frame_mask = (torch.arange(self.max_frames, device=m_p.device).view(1, 1, -1) < y_lengths.view(-1, 1, 1)).to(m_p.dtype)
        z = (m + latent_noise * torch.exp(logs) * noise_scale) * frame_mask
        return z, frame_mask, y_lengths, duration

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    decoder = args.output_dir / 'decoder_50tk_256f.onnx'
    # Intermediate tensors in the canonical graph do not all carry shape
    # metadata. Defer validation until the fixed decoder contract is applied.
    utils.extract_model(
        str(args.model), str(decoder), DECODER_INPUTS, ['y'], check_model=False
    )
    model = onnx.load(str(decoder))
    for value, shape in zip(model.graph.input, DECODER_SHAPES):
        dims = value.type.tensor_type.shape.dim
        del dims[:]
        for size in shape:
            dim = dims.add(); dim.dim_value = size
    onnx.checker.check_model(model)
    onnx.save(model, str(decoder))
    controller = torch.jit.script(StaticVitsController256().eval())
    controller.save(str(args.output_dir / 'cpu_dynamic_controller_256f.pt'))

if __name__ == '__main__': main()
