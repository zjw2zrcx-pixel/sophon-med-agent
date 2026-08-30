#!/usr/bin/env python3
"""Regenerate the first-chunk KWS golden vectors from test.wav.

Requires Python packages: numpy, soundfile, onnxruntime and kaldi-native-fbank.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf


def sherpa_linear_resample(x: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Bit-for-bit algorithm counterpart of sherpa-onnx LinearResample for one flush."""
    if in_rate == out_rate:
        return x.astype(np.float32, copy=False)
    cutoff = 0.99 * 0.5 * min(in_rate, out_rate)
    num_zeros = 6
    base = math.gcd(in_rate, out_rate)
    in_unit, out_unit = in_rate // base, out_rate // base
    width = num_zeros / (2.0 * cutoff)
    n_out = (len(x) * out_rate - 1) // in_rate + 1
    out = np.zeros(n_out, dtype=np.float32)
    for out_i in range(n_out):
        wrapped = out_i % out_unit
        unit = out_i // out_unit
        output_t = wrapped / out_rate
        first = math.ceil((output_t - width) * in_rate) + unit * in_unit
        last = math.floor((output_t + width) * in_rate) + unit * in_unit
        idx = np.arange(first, last + 1, dtype=np.int64)
        delta = idx.astype(np.float64) / in_rate - out_i / out_rate
        window = np.where(np.abs(delta) < width,
                          0.5 * (1.0 + np.cos(2.0 * np.pi * cutoff / num_zeros * delta)),
                          0.0)
        filt = np.full(delta.shape, 2.0 * cutoff, dtype=np.float64)
        nonzero = delta != 0.0
        filt[nonzero] = (np.sin(2.0 * np.pi * cutoff * delta[nonzero]) /
                         (np.pi * delta[nonzero]))
        valid = (idx >= 0) & (idx < len(x))
        out[out_i] = np.sum(x[idx[valid]] * (filt[valid] * window[valid] / in_rate))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--wav', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio, rate = sf.read(args.wav, dtype='float32', always_2d=True)
    # The supplied test WAV is stereo-identical; mean establishes a precise mono policy.
    mono = audio.mean(axis=1, dtype=np.float32)
    samples = sherpa_linear_resample(mono, rate, 16000)

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.window_type = 'povey'
    opts.frame_opts.round_to_power_of_two = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0
    opts.mel_opts.is_librosa = False
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, samples)
    fbank.input_finished()
    feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)], axis=0).astype(np.float32)
    x = feats[:45][None, ...]
    np.save(args.output_dir / 'fbank_45.npy', x)
    np.save(args.output_dir / 'resampled_16k_mono.npy', samples)

    enc_path = next(args.model_dir.glob('encoder-*-chunk-16-left-64.onnx'))
    dec_path = next(args.model_dir.glob('decoder-*-chunk-16-left-64.onnx'))
    join_path = next(args.model_dir.glob('joiner-*-chunk-16-left-64.onnx'))
    enc = ort.InferenceSession(str(enc_path), providers=['CPUExecutionProvider'])
    feeds = {}
    for item in enc.get_inputs():
        if item.name == 'x':
            feeds[item.name] = x
        elif item.name == 'processed_lens':
            feeds[item.name] = np.zeros((1,), dtype=np.int64)
        else:
            shape = [1 if isinstance(d, str) else d for d in item.shape]
            feeds[item.name] = np.zeros(shape, dtype=np.float32)
    enc_values = enc.run(None, feeds)
    enc_out = enc_values[0]
    dec = ort.InferenceSession(str(dec_path), providers=['CPUExecutionProvider'])
    decoder_y = np.zeros((1, 2), dtype=np.int64)  # blank (ID 0) history
    decoder_out = dec.run(None, {'y': decoder_y})[0]
    join = ort.InferenceSession(str(join_path), providers=['CPUExecutionProvider'])
    logits = join.run(None, {'encoder_out': enc_out[:, 0, :], 'decoder_out': decoder_out})[0]
    np.savez(args.output_dir / 'encoder_decoder_joiner_golden.npz',
             encoder_input_x=x, encoder_out=enc_out,
             decoder_y=decoder_y, decoder_out=decoder_out,
             joiner_encoder_out_t0=enc_out[:, 0, :], joiner_logits_t0=logits,
             **{f'encoder_input_{k}': v for k, v in feeds.items() if k != 'x'})
    (args.output_dir / 'feature_config.json').write_text(json.dumps({
        'producer': 'sherpa-onnx compatible preprocessing', 'sherpa_onnx_version': '1.13.4',
        'input_wav': args.wav.name, 'input_sample_rate': rate, 'input_channels': int(audio.shape[1]),
        'mono_policy': 'arithmetic mean; channels are bit-identical for this test.wav',
        'resampler': {'implementation': 'sherpa_onnx::LinearResample', 'output_sample_rate': 16000,
                      'filter_cutoff_hz': 7920.0, 'num_zeros': 6},
        'fbank': {'num_bins': 80, 'sample_rate': 16000, 'frame_length_ms': 25, 'frame_shift_ms': 10,
                  'low_freq_hz': 20, 'high_freq_hz': -400, 'dither': 0, 'snip_edges': False,
                  'remove_dc_offset': True, 'preemph_coeff': 0.97, 'window_type': 'povey',
                  'round_to_power_of_two': True, 'is_librosa': False, 'normalize_samples': True},
        'first_encoder_call': {'x_shape': [1, 45, 80], 'all_cache_states': 'float32 zero',
                               'processed_lens': [0]}
    }, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
