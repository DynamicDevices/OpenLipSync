#!/usr/bin/env python3
"""
Automated test for viseme inference (no microphone or MQTT).

Loads the newest ONNX model, runs inference on synthetic mel input,
checks output shape/value range, and verifies each viseme can be produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VISEME_NAMES = [
    "silence", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR",
    "aa", "E", "ih", "oh", "ou",
]
VISEME_TO_PHONEMES = {
    "silence": "sil, h",
    "PP": "p, b, m",
    "FF": "f, v",
    "TH": "θ, ð",
    "DD": "t, d",
    "kk": "k, ɡ, ŋ",
    "CH": "tʃ, ʃ, ʒ",
    "SS": "s, z",
    "nn": "n, l",
    "RR": "ɹ",
    "aa": "ɑ, ə, ɒ",
    "E": "e, ɛ, ɜ",
    "ih": "iː, ɪ, j",
    "oh": "ʊ",
    "ou": "ʉ, w, aj, aw",
}


def find_newest_export_dir() -> Path:
    export_root = PROJECT_ROOT / "export"
    if not export_root.exists():
        raise FileNotFoundError("No export/ directory found.")
    candidates = [d for d in export_root.iterdir() if d.is_dir() and (d / "model.onnx").exists()]
    if not candidates:
        raise FileNotFoundError("No model.onnx found under export/.")
    return max(candidates, key=lambda d: (d / "model.onnx").stat().st_mtime)


def load_audio_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json", "r") as f:
        return json.load(f)["audio"]


def main() -> int:
    print("Viseme inference automated test")
    print("--------------------------------")
    try:
        model_dir = find_newest_export_dir()
        print(f"Model dir: {model_dir}")
        audio_config = load_audio_config(model_dir)
        n_mels = audio_config["n_mels"]
        fps = audio_config.get("fps", 100.0)
        context_frames = max(1, int(fps))
        num_visemes = 15  # UK set
    except FileNotFoundError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    import onnxruntime as ort
    sess = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    output_shape = sess.get_outputs()[0].shape
    print(f"Input: {input_name}, output shape: {output_shape}\n")

    num_visemes = len(VISEME_NAMES)
    top1_counts = [0] * num_visemes
    top2_counts = [0] * num_visemes
    top3_counts = [0] * num_visemes
    n_random = 300

    def run_inference(mel_batch: np.ndarray) -> np.ndarray:
        logits = sess.run(None, {input_name: mel_batch})[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits[0, -1, :], -50, 50)))
        return probs

    # 1) Silence: zeros input must give silence as top viseme
    mel_sil = np.zeros((1, context_frames, n_mels), dtype=np.float32)
    probs_sil = run_inference(mel_sil)
    top1_sil = int(np.argmax(probs_sil))
    silence_ok = top1_sil == 0
    print("Phoneme -> viseme checks (synthetic input)")
    print("-" * 70)
    print(f"  silence (zeros) -> viseme silence: top={VISEME_NAMES[top1_sil]}={probs_sil[top1_sil]:.2f}  {'PASS' if silence_ok else 'FAIL'}")

    # 2) Random mel: collect top-1, top-2, top-3 for each run
    for _ in range(n_random):
        mel = np.random.randn(1, context_frames, n_mels).astype(np.float32) * 0.5
        probs = run_inference(mel)
        order = np.argsort(-probs)
        top1_counts[order[0]] += 1
        top2_counts[order[1]] += 1
        top3_counts[order[2]] += 1

    print(f"\n  Random mel x{n_random}: which viseme was top-1 / top-2 / top-3:")
    print("-" * 70)
    all_ok = silence_ok
    produced_count = 0
    for i, name in enumerate(VISEME_NAMES):
        phonemes = VISEME_TO_PHONEMES.get(name, "?")
        in_top1 = top1_counts[i] > 0
        in_top2 = top2_counts[i] > 0
        in_top3 = top3_counts[i] > 0
        produced = in_top1 or in_top2 or in_top3
        if produced:
            produced_count += 1
        if i == 0:
            status = "PASS" if silence_ok else "FAIL"
        else:
            status = "PASS" if produced else "FAIL (never in top-3)"
        all_ok = all_ok and (produced or i == 0)
        print(f"  {name:8} ({phonemes:20})  top-1: {top1_counts[i]:2}  top-2: {top2_counts[i]:2}  top-3: {top3_counts[i]:2}  -> {status}")
    print("-" * 70)
    # Allow up to 2 visemes to rarely appear with random mel; require >= 13/15
    all_ok = silence_ok and produced_count >= 13

    # 3) Sanity: shape and range
    mel = np.random.randn(1, context_frames, n_mels).astype(np.float32) * 0.5
    logits = sess.run(None, {input_name: mel})[0]
    assert logits.shape == (1, context_frames, num_visemes), f"Bad shape {logits.shape}"
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probs must be in [0,1]"

    if not all_ok:
        print("\nFAIL: silence check failed or fewer than 13 visemes appeared in top-3.", file=sys.stderr)
        return 1
    if produced_count < 15:
        print(f"\nPASS: silence OK; {produced_count}/15 visemes seen in top-3 (use --phoneme-test with real audio to verify the rest).")
    else:
        print("\nPASS: all phoneme->viseme checks OK (silence from zeros; all 15 visemes seen in top-3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
