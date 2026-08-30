#!/usr/bin/env python3
"""Hybrid VITS inference: native 50-token / 256-frame FP32 deployment."""
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
import torch
from sophon import sail

TOKENS = 50
MAX_FRAMES = 256
SAMPLES_PER_FRAME = 512


class HybridVitsRuntime:
    def __init__(self, model_dir: str | Path, device_id: int = 0) -> None:
        root = Path(model_dir)
        self.encoder = sail.Engine(str(root / "vits_encoder_50_bm1684x_f32.bmodel"), device_id, sail.IOMode.SYSIO)
        self.decoder = sail.Engine(str(root / "vits_flow_decoder_256_bm1684x_f32.bmodel"), device_id, sail.IOMode.SYSIO)
        self.encoder_graph = self.encoder.get_graph_names()[0]
        self.decoder_graph = self.decoder.get_graph_names()[0]
        self.dp = ort.InferenceSession(str(root / "vits_dp_cpu_50.onnx"), providers=["CPUExecutionProvider"])
        self.controller = torch.jit.load(str(root / "cpu_dynamic_controller_256f.pt"), map_location="cpu").eval()

    @staticmethod
    def _run(engine: sail.Engine, graph: str, values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return engine.process(graph, values)

    @staticmethod
    def _pad_tokens(values: np.ndarray, length: int) -> np.ndarray:
        padded = np.zeros((1, TOKENS), dtype=np.int32)
        padded[0, :length] = values.astype(np.int32, copy=False)
        return padded

    def synthesize_text(self, text: str, assets_dir: str | Path | None = None, **kwargs) -> np.ndarray:
        """Run the packaged frontend and synthesize one text string."""
        from melo_zh_lexicon_frontend import MeloZhLexiconFrontend

        assets = Path(assets_dir) if assets_dir else Path(__file__).parent / "preprocess_assets"
        x, tones = MeloZhLexiconFrontend(assets).convert(text)
        return self.synthesize_tokens(x, tones, **kwargs)

    def synthesize_tokens(self, x: np.ndarray, tones: np.ndarray, sid: int = 1,
                          noise_scale: float = 0.667, length_scale: float = 1.0,
                          seed: int = 20260727,
                          ) -> np.ndarray:
        """Synthesize pre-tokenized input and return a 44.1 kHz float32 waveform."""
        x, tones = np.asarray(x).reshape(-1), np.asarray(tones).reshape(-1)
        if not 0 < len(x) <= TOKENS or len(x) != len(tones):
            raise ValueError("x and tones must have the same length in [1, 50]")
        length = len(x)
        enc = self._run(self.encoder, self.encoder_graph, {
            "x": self._pad_tokens(x, length), "x_lengths": np.asarray([length], dtype=np.int32),
            "tones": self._pad_tokens(tones, length), "sid": np.asarray([sid], dtype=np.int32),
        })
        hidden = enc["/enc_p/encoder/Mul_3_output_0_Mul"]
        m_p, logs_p = enc["/enc_p/Split_output_0_Split"], enc["/enc_p/Split_output_1_Split"]
        x_mask = enc["/enc_p/Unsqueeze_2_output_0_Unsqueeze"]
        speaker, condition = enc["/Unsqueeze_6_output_0_Unsqueeze"], enc["/enc_p/encoder/Transpose_output_0_Transpose"]
        logw = self.dp.run(None, {
            "/enc_p/encoder/Mul_3_output_0": hidden, "/Unsqueeze_6_output_0": speaker,
            "/enc_p/Cast_1_output_0": x_mask,
        })[0]
        # The native 256-frame controller takes explicit noise.  Keep the seed
        # and tensor shape fixed so controller and end-to-end runs are replayable.
        torch.manual_seed(seed)
        latent_noise = torch.randn((1, 192, MAX_FRAMES), dtype=torch.float32)
        with torch.inference_mode():
            z, y_mask, y_lengths, _ = self.controller(
                torch.from_numpy(m_p), torch.from_numpy(logs_p), torch.from_numpy(x_mask),
                torch.from_numpy(logw), torch.tensor([noise_scale]), torch.tensor([length_scale]), latent_noise,
            )
        frames = int(y_lengths[0])
        if frames > MAX_FRAMES:
            raise ValueError(f"predicted {frames} frames; limit is {MAX_FRAMES}; split the sentence")
        z_pad = z.numpy().astype(np.float32, copy=False)
        y_mask = y_mask.numpy().astype(np.float32, copy=False)
        output = self._run(self.decoder, self.decoder_graph, {
            "/Add_2_output_0": z_pad, "/Cast_4_output_0": y_mask, "/Unsqueeze_10_output_0": y_mask[..., None],
            "/enc_p/encoder/Transpose_output_0": condition, "/Unsqueeze_6_output_0": speaker,
        })["y_Tanh"]
        return output[0, 0, :frames * SAMPLES_PER_FRAME].copy()
