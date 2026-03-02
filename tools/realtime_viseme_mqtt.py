#!/usr/bin/env python3
"""
Real-time microphone → visemes → MQTT test harness.

Captures live audio from the default microphone, runs the OpenLipSync ONNX model
to get viseme activations, and publishes them as JSON to an MQTT broker.

Usage:
  pip install -e ".[realtime]"
  python tools/realtime_viseme_mqtt.py
  python tools/realtime_viseme_mqtt.py --broker mqtt.dynamicdevices.co.uk --topic openlipsync/visemes

Normalization uses a short warmup period (default 1s) to compute mean/std over
ambient noise/speech, then uses those stats for the rest of the session.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CLIENT_ID_FILE = Path.home() / ".openlipsync-mqtt-client-id"


def _play_audio_file(path: Path) -> bool:
    """Play an audio file (e.g. MP3) using ffplay, afplay, or mpv. Returns True if played."""
    candidates = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        ["afplay", str(path)],  # macOS
        ["mpv", "--no-video", "--really-quiet", str(path)],
    ]
    for cmd in candidates:
        exe = shutil.which(cmd[0])
        if exe is None:
            continue
        try:
            subprocess.run(
                [exe] + cmd[1:],
                capture_output=True,
                timeout=30,
                cwd=path.parent,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return False


def _start_phoneme_playback(segment_index: int) -> None:
    """Start playing the phoneme clip for this segment in the background (mic will capture it)."""
    for ext in (".wav", ".mp3"):
        path = PHONEME_PROMPTS_DIR / f"segment_{segment_index:02d}{ext}"
        if not path.exists():
            continue
        exe = None
        cmd = None
        for player, args in (
            ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]),
            ("afplay", [str(path)]),
            ("mpv", ["--no-video", "--really-quiet", str(path)]),
        ):
            exe = shutil.which(player)
            if exe:
                cmd = [exe] + args
                break
        if cmd:
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=path.parent)
            except (FileNotFoundError, OSError):
                pass
        return


def _default_client_id() -> str:
    """Stable per-device ID across runs: prefer persisted file, else MAC-derived, else generate once and persist."""
    if _CLIENT_ID_FILE.exists():
        try:
            raw = _CLIENT_ID_FILE.read_text().strip()
            if raw and len(raw) <= 23:
                return raw
        except Exception:
            pass
    try:
        node = uuid.getnode()
        if node != 0:
            cid = f"olips-{node:012x}"
        else:
            cid = f"openlipsync-{uuid.uuid4().hex[:8]}"
    except Exception:
        cid = f"openlipsync-{uuid.uuid4().hex[:8]}"
    try:
        _CLIENT_ID_FILE.write_text(cid)
    except Exception:
        pass
    return cid


# Viseme names in model output order (from viseme_map_en_uk_mfa.json)
VISEME_NAMES = [
    "silence", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR",
    "aa", "E", "ih", "oh", "ou",
]

# Example phoneme(s) per viseme for display (UK set)
VISEME_TO_PHONEMES: dict[str, str] = {
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

# For --phoneme-test: (prompt, expected viseme, expected phoneme hint for display)
PHONEME_PROMPTS = [
    ("Stay silent", "silence", "sil"),
    ("Say 'eeee' (long e)", "ih", "iː"),
    ("Say 'ah' as in father", "aa", "ɑ"),
    ("Say 'eh' as in bed", "E", "ɛ"),
    ("Say 'oh' as in lot", "oh", "ʊ"),
    ("Say 'oo' as in moon", "ou", "ʉ / w"),
    ("Say 'p' or 'b' or 'm'", "PP", "p, b, m"),
    ("Say 'f' or 'v'", "FF", "f, v"),
    ("Say 'th' (thin)", "TH", "θ"),
    ("Say 't' or 'd'", "DD", "t, d"),
    ("Say 'k' or 'g'", "kk", "k, ɡ"),
    ("Say 'sh' or 'ch'", "CH", "ʃ, tʃ"),
    ("Say 's' or 'z'", "SS", "s, z"),
    ("Say 'n' or 'l'", "nn", "n, l"),
    ("Say 'r'", "RR", "ɹ"),
]

# Pre-generated phoneme test clips: segment_01.wav (silence), segment_02..15.mp3 (sounds e, ah, eh, ...).
# With --speak, these are played in the background while the mic captures; harness checks expected viseme.
# Generate with: uv run --extra realtime python tools/generate_phoneme_prompts.py
PHONEME_PROMPTS_DIR = PROJECT_ROOT / "data" / "phoneme_prompts"


def find_newest_export_dir() -> Path:
    """Return the export directory containing the newest model.onnx."""
    export_root = PROJECT_ROOT / "export"
    if not export_root.exists():
        raise FileNotFoundError("No export/ directory found. Export a model first.")
    candidates = [d for d in export_root.iterdir() if d.is_dir() and (d / "model.onnx").exists()]
    if not candidates:
        raise FileNotFoundError("No model.onnx found under export/. Export a model first.")
    return max(candidates, key=lambda d: (d / "model.onnx").stat().st_mtime)


def load_audio_config(model_dir: Path) -> dict:
    """Load audio config from export dir config.json."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["audio"]


def build_mel_transform(audio_config: dict) -> T.MelSpectrogram:
    """Build torchaudio mel spectrogram transform from config."""
    return T.MelSpectrogram(
        sample_rate=audio_config["sample_rate"],
        n_fft=audio_config["n_fft"],
        win_length=audio_config["window_length_samples"],
        hop_length=audio_config["hop_length_samples"],
        n_mels=audio_config["n_mels"],
        f_min=audio_config["fmin"],
        f_max=audio_config["fmax"],
        power=2.0,
        normalized=False,
    )


def amplitude_to_db() -> T.AmplitudeToDB:
    return T.AmplitudeToDB(stype="power", top_db=80)


def waveform_to_mel_frame(
    waveform_chunk: np.ndarray,
    mel_transform: T.MelSpectrogram,
    amp_to_db: T.AmplitudeToDB,
    device: torch.device,
) -> np.ndarray:
    """Compute mel frame(s) from waveform; returns the last (most recent) frame only, shape (n_mels,)."""
    t = torch.from_numpy(waveform_chunk.astype(np.float32)).unsqueeze(0).to(device)
    mel = mel_transform(t)
    mel_db = amp_to_db(mel)
    # (1, n_mels, time) -> take last time step -> (n_mels,)
    out = mel_db.squeeze(0).transpose(0, 1)
    return out[-1].cpu().numpy()


def run_realtime(
    model_dir: Path,
    broker: str,
    port: int,
    topic_base: str,
    client_id: str | None,
    warmup_seconds: float,
    publish_interval_frames: int,
    device_name: str | None,
    phoneme_test: bool = False,
    segment_duration: float = 5.0,
    speak_prompts: bool = False,
) -> None:
    import sounddevice as sd
    import onnxruntime as ort
    import paho.mqtt.client as mqtt

    audio_config = load_audio_config(model_dir)
    sample_rate = audio_config["sample_rate"]
    hop = audio_config["hop_length_samples"]
    n_fft = audio_config["n_fft"]
    n_mels = audio_config["n_mels"]
    fps = audio_config.get("fps", 100.0)
    context_frames = max(1, int(fps))  # 1 second of context (TCN needs temporal context)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_transform = build_mel_transform(audio_config).to(device)
    amp_to_db = amplitude_to_db()

    # ONNX session
    onnx_path = model_dir / "model.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Viseme names: ensure we have as many as model outputs
    num_visemes = sess.get_outputs()[0].shape[-1]
    viseme_names = VISEME_NAMES[:num_visemes]
    if len(viseme_names) < num_visemes:
        viseme_names.extend([f"v{i}" for i in range(len(viseme_names), num_visemes)])

    # MQTT client (skip when phoneme-test)
    if client_id is None or client_id == "":
        client_id = _default_client_id()
    topic = f"{topic_base.rstrip('/')}/{client_id}"
    client = None
    if not phoneme_test:
        client = mqtt.Client(client_id=client_id)
        client.connect(broker, port, 60)
        client.loop_start()

    # Phoneme-test: per-segment collection and prompts
    segment_state: dict | None = None
    if phoneme_test:
        segment_state = {
            "segment_index": 0,
            "segment_start": time.time(),
            "segment_values": [[] for _ in PHONEME_PROMPTS],
            "done": False,
        }
        print("\n--- Phoneme check: playing each test sound, capturing from mic, checking viseme ---")
        print(f"Segment 1/{len(PHONEME_PROMPTS)}: silence (expect viseme: {PHONEME_PROMPTS[0][1]}, phoneme: {PHONEME_PROMPTS[0][2]})\n")
        if speak_prompts:
            _start_phoneme_playback(1)

    # Ring buffer: mel transform needs n_fft samples per frame; advance by hop each time
    # Mel context: TCN expects a sequence (e.g. 100 frames); we feed last context_frames and use last output
    buffer: list[float] = []
    mel_context: deque[np.ndarray] = deque(maxlen=context_frames)
    frame_count = 0
    warmup_frames = max(1, int(warmup_seconds * sample_rate / hop))
    warmup_mels: list[np.ndarray] = []
    mean_mel: np.ndarray | None = None
    std_mel: np.ndarray | None = None
    var_mel: np.ndarray | None = None
    norm_alpha = 0.995  # EMA decay for adaptive normalization
    # Running peak over last ~1 s for MQTT (same idea as phoneme-test segment peak)
    peak_window_frames = 100
    peak_deque: deque[dict] = deque(maxlen=peak_window_frames)

    def on_audio(indata: np.ndarray, _frames: int, _time_info: object, _status: object) -> None:
        nonlocal frame_count, mean_mel, std_mel, var_mel
        chunk = indata[:, 0].astype(np.float64) / 32768.0
        buffer.extend(chunk.tolist())
        while len(buffer) >= n_fft:
            arr = np.array(buffer[:n_fft], dtype=np.float32)
            mel_frame = waveform_to_mel_frame(arr, mel_transform, amp_to_db, device)
            if mean_mel is None:
                warmup_mels.append(mel_frame)
                if len(warmup_mels) >= warmup_frames:
                    stack = np.stack(warmup_mels)
                    mean_mel = np.mean(stack, axis=0)
                    std_mel = np.clip(np.std(stack, axis=0), 1e-8, None)
                    var_mel = (std_mel.astype(np.float64) ** 2).copy()
            else:
                # Adaptive normalization: EMA so stats track recent speech
                diff = mel_frame.astype(np.float64) - mean_mel
                mean_mel = norm_alpha * mean_mel + (1 - norm_alpha) * mel_frame.astype(np.float64)
                var_mel = norm_alpha * var_mel + (1 - norm_alpha) * (diff ** 2)
                std_mel = np.sqrt(np.maximum(var_mel, 1e-8)).astype(np.float32)
                normalized = ((mel_frame - mean_mel) / std_mel).astype(np.float32)
                mel_context.append(normalized)
                if len(mel_context) >= context_frames:
                    seq = np.array(list(mel_context), dtype=np.float32)
                    if seq.shape[0] < context_frames:
                        pad = np.repeat(seq[:1], context_frames - seq.shape[0], axis=0)
                        seq = np.concatenate([pad, seq], axis=0)
                    normalized_batch = seq[-context_frames:].reshape(1, context_frames, n_mels)
                    logits = sess.run(None, {input_name: normalized_batch})[0]
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits[0, -1, :], -50, 50)))
                    raw_visemes = {name: float(probs[i]) for i, name in enumerate(viseme_names)}
                    total = sum(raw_visemes.values())
                    n = len(viseme_names)
                    visemes_norm = {name: (v / total if total > 0 else 1.0 / n) for name, v in raw_visemes.items()}
                    # Correct float error so sum is exactly 1.0
                    _s = sum(visemes_norm.values())
                    if abs(_s - 1.0) > 1e-9:
                        _max_name = max(visemes_norm, key=visemes_norm.get)
                        visemes_norm[_max_name] += 1.0 - _s
                    payload = {
                        "t": time.time(),
                        "frame": frame_count,
                        "client_id": client_id,
                        "visemes": visemes_norm,
                    }
                    peak_deque.append(raw_visemes.copy())
                    peak_raw = {name: max(d[name] for d in peak_deque) for name in viseme_names}
                    peak_total = sum(peak_raw.values())
                    payload["visemes_peak"] = {name: (v / peak_total if peak_total > 0 else 1.0 / n) for name, v in peak_raw.items()}
                    _sp = sum(payload["visemes_peak"].values())
                    if abs(_sp - 1.0) > 1e-9:
                        _max_name = max(payload["visemes_peak"], key=payload["visemes_peak"].get)
                        payload["visemes_peak"][_max_name] += 1.0 - _sp
                    if phoneme_test and segment_state is not None:
                        idx = segment_state["segment_index"]
                        if idx < len(segment_state["segment_values"]):
                            segment_state["segment_values"][idx].append(payload["visemes"].copy())
                        if time.time() - segment_state["segment_start"] >= segment_duration:
                            # Summarize this segment
                            vals = segment_state["segment_values"][idx]
                            expected_viseme = PHONEME_PROMPTS[idx][1]
                            expected_phoneme = PHONEME_PROMPTS[idx][2]
                            if vals:
                                # Use peak (max) over segment so levels reflect when the model fired, not diluted by silence
                                maxs = {k: max(d[k] for d in vals) for k in viseme_names}
                                top = sorted(maxs.items(), key=lambda x: -x[1])[:3]
                                in_top = expected_viseme in [t[0] for t in top[:2]]
                                sym = "✓" if in_top else "✗"
                                phoneme_hints = " | ".join(f"{t[0]}→{VISEME_TO_PHONEMES.get(t[0], '?')}" for t in top)
                                print(f"  → Expected viseme {expected_viseme} (phoneme {expected_phoneme}). Top 3 (peak): {top[0][0]}={top[0][1]:.2f} {top[1][0]}={top[1][1]:.2f} {top[2][0]}={top[2][1]:.2f}  [{sym}]")
                                print(f"     Phonemes for top 3: {phoneme_hints}")
                            else:
                                print(f"  → No frames in segment [expected {expected_viseme} / {expected_phoneme}]")
                            segment_state["segment_index"] = idx + 1
                            segment_state["segment_start"] = time.time()
                            if segment_state["segment_index"] >= len(PHONEME_PROMPTS):
                                segment_state["done"] = True
                                print("\n--- Phoneme check done ---")
                            else:
                                ni = segment_state["segment_index"]
                                p, v, ph = PHONEME_PROMPTS[ni]
                                print(f"\nSegment {ni + 1}/{len(PHONEME_PROMPTS)}: playing sound for {p} (expect viseme: {v}, phoneme: {ph})\n")
                                if speak_prompts:
                                    _start_phoneme_playback(ni + 1)
                    else:
                        if publish_interval_frames <= 0 or frame_count % publish_interval_frames == 0:
                            if client is not None:
                                client.publish(topic, json.dumps(payload), qos=0)
                            if not phoneme_test:
                                parts = [f"{name}={payload['visemes'][name]:.2f}" for name in viseme_names]
                                total_display = sum(payload["visemes"].values())
                                print(f"[frame {frame_count:5d}] " + " ".join(parts) + f"  sum={total_display:.2f}", flush=True)
            frame_count += 1
            buffer[:] = buffer[hop:]

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=hop,
        device=device_name,
        callback=on_audio,
    )
    stream.start()
    if phoneme_test:
        print(f"Warmup: {warmup_seconds:.1f}s. Then {len(PHONEME_PROMPTS)} segments of {segment_duration:.0f}s each.")
    else:
        print(f"Client ID: {client_id}")
        print(f"Streaming mic -> visemes -> MQTT {broker}:{port} topic {topic}")
        print("Warmup: {:.1f}s then publishing every {} frame(s). Ctrl+C to stop.".format(warmup_seconds, publish_interval_frames if publish_interval_frames > 0 else 1))
    try:
        if phoneme_test and segment_state is not None:
            while not segment_state["done"]:
                time.sleep(0.2)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        if client is not None:
            client.loop_stop()
            client.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-time microphone -> visemes -> MQTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to export dir containing model.onnx and config.json (default: newest under export/)",
    )
    parser.add_argument("--broker", type=str, default="mqtt.dynamicdevices.co.uk", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--topic", type=str, default="openlipsync/visemes", help="MQTT topic prefix; client ID is appended (e.g. openlipsync/visemes/<client_id>)")
    parser.add_argument("--client-id", type=str, default=None, help="MQTT client ID (default: stable per device from ~/.openlipsync-mqtt-client-id or MAC)")
    parser.add_argument("--warmup", type=float, default=1.0, dest="warmup_seconds", help="Warmup seconds for normalization stats")
    parser.add_argument(
        "--publish-every",
        type=int,
        default=1,
        dest="publish_interval_frames",
        help="Publish every N frames (1 = every 10ms; 5 = 50ms)",
    )
    parser.add_argument("--device", type=str, default=None, help="Sounddevice input device name or index")
    parser.add_argument("--phoneme-test", action="store_true", help="Run phoneme check: prompt for each sound and report if expected viseme is in top 2")
    parser.add_argument("--segment-duration", type=float, default=5.0, help="Seconds per segment in phoneme-test (default: 5)")
    parser.add_argument("--speak", action="store_true", dest="speak_prompts", help="In phoneme-test, play the test phoneme (or silence) each segment in background while capturing; check expected viseme")
    args = parser.parse_args()

    model_dir = args.model_dir or find_newest_export_dir()
    if not (model_dir / "model.onnx").exists():
        print(f"Error: model.onnx not found in {model_dir}", file=sys.stderr)
        return 1

    try:
        run_realtime(
            model_dir=model_dir,
            broker=args.broker,
            port=args.port,
            topic_base=args.topic,
            client_id=args.client_id,
            warmup_seconds=args.warmup_seconds,
            publish_interval_frames=args.publish_interval_frames,
            device_name=args.device,
            phoneme_test=args.phoneme_test,
            segment_duration=args.segment_duration,
            speak_prompts=args.speak_prompts,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
