#!/usr/bin/env python3
"""
Validate C mel spectrogram against torchaudio.

Run from project root:
  uv run python audio/validate_mel.py

Requires: mel_from_raw built (make -C audio mel_from_raw)
"""

from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "audio"


def torchaudio_mel(audio: np.ndarray, sample_rate: int = 16000, center: bool = False) -> np.ndarray:
    """Compute mel spectrogram using torchaudio (matches training pipeline)."""
    config = {
        "sample_rate": sample_rate,
        "n_fft": 1024,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 80,
        "f_min": 50,
        "f_max": 8000,
        "power": 2.0,
        "center": center,  # False to match C (no padding)
    }
    mel_t = T.MelSpectrogram(**config)
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    mel = mel_t(waveform)
    mel_db = amp_to_db(mel)
    # (1, n_mels, time) -> (time, n_mels)
    return mel_db.squeeze(0).transpose(0, 1).numpy()


def c_mel(audio: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram using C implementation."""
    mel_from_raw = AUDIO_DIR / "mel_from_raw"
    if not mel_from_raw.exists():
        raise FileNotFoundError(
            f"Build mel_from_raw first: make -C {AUDIO_DIR} mel_from_raw"
        )

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as fin:
        fin.write(audio.astype(np.float32).tobytes())
        in_path = fin.name
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as fout:
        out_path = fout.name

    try:
        result = subprocess.run(
            [str(mel_from_raw), in_path, out_path],
            capture_output=True,
            text=True,
            cwd=str(AUDIO_DIR),
        )
        if result.returncode != 0:
            raise RuntimeError(f"mel_from_raw failed: {result.stderr}")

        data = np.fromfile(out_path, dtype=np.float32)
        n_mels = 80
        n_frames = len(data) // n_mels
        return data.reshape(n_frames, n_mels)
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def main() -> int:
    print("Generating 1 second of 440 Hz test tone at 16 kHz...")
    sample_rate = 16000
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("Computing mel with torchaudio (center=False to match C)...")
    mel_py = torchaudio_mel(audio, sample_rate, center=False)

    print("Computing mel with C implementation...")
    mel_c = c_mel(audio)

    print(f"  torchaudio: {mel_py.shape}")
    print(f"  C:          {mel_c.shape}")

    # Allow slight frame count difference at boundaries
    n = min(mel_py.shape[0], mel_c.shape[0])
    diff = np.abs(mel_py[:n] - mel_c[:n])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nComparison (first {n} frames):")
    print(f"  Max absolute difference:  {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Built-in radix-2 FFT may differ from torch's FFT; use custom FFT callback for exact match
    tol = 2.0  # dB tolerance for validation
    if max_diff < tol:
        print(f"\nPASS: Differences within tolerance ({tol})")
        return 0
    else:
        print(f"\nFAIL: Max diff {max_diff:.4f} exceeds tolerance {tol}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
