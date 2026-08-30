#!/usr/bin/env python3
"""Run a synthetic smoke test for the exported 256-frame TorchScript controller."""
import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="path to cpu_dynamic_controller_256f.pt")
    args = parser.parse_args()

    controller = torch.jit.load(str(args.model), map_location="cpu").eval()
    torch.manual_seed(0)
    m_p = torch.randn(1, 192, 50)
    logs_p = torch.zeros(1, 192, 50)
    x_mask = torch.ones(1, 1, 50)
    # A negative synthetic log-duration keeps the result inside the 256-frame bucket.
    logw = torch.full((1, 1, 50), -1.0)
    latent_noise = torch.randn(1, 192, 256)

    with torch.inference_mode():
        z, frame_mask, y_lengths, duration = controller(
            m_p,
            logs_p,
            x_mask,
            logw,
            torch.tensor([0.667], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            latent_noise,
        )

    print(
        {
            "z_shape": list(z.shape),
            "frame_mask_shape": list(frame_mask.shape),
            "frames": int(y_lengths[0]),
            "duration_shape": list(duration.shape),
        }
    )


if __name__ == "__main__":
    main()
