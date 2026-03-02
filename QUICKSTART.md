# Try OpenLipSync on your laptop

Minimal steps to run a short training job (small dataset, few epochs) on your machine.

## 0. Clone and Git LFS (if the repo has LFS-tracked files)

If the repo uses Git LFS for exported models (e.g. `export/*/model.onnx`):

1. **Install Git LFS** once per machine: [git-lfs.github.com](https://git-lfs.github.com) or `sudo apt install git-lfs` / `brew install git-lfs`.
2. **After clone**, run in the repo:
   ```bash
   git lfs install
   git lfs pull
   ```
   so that LFS files (e.g. `.onnx`) are downloaded instead of left as pointer files. Without this, the C# inference app may fail to load the model.

If `export/` is not in the repo (e.g. it’s in `.gitignore`), you don’t need LFS for export; you’ll need to run training/export locally or get the export folder from someone.

## 1. Prerequisites

- **Python 3.13+** (project uses `requires-python = ">=3.13"` in `pyproject.toml`)
- **uv** (install: <https://docs.astral.sh/uv/getting-started/installation/>)
- **ffmpeg** (for converting LibriSpeech FLAC → WAV)
- **micromamba** (for Montreal Forced Aligner; install: <https://mamba.readthedocs.io/en/latest/installation.html>)

## 2. Install project dependencies

From the **project root** (the `OpenLipSync` directory):

```bash
uv sync
```

This creates a virtual environment and installs PyTorch, torchaudio, and other deps.

## 3. Set up Montreal Forced Aligner (MFA)

MFA is used to get phoneme timings from speech; the training pipeline needs it for viseme labels.

```bash
micromamba create -n mfa -c conda-forge python=3.12 montreal-forced-aligner
micromamba activate mfa

mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

Then `micromamba deactivate` (training runs in your normal shell with `uv`).

## 4. Run a quick training job

From the **project root**:

```bash
uv run python training/train.py --config training/recipes/tcn_quick_laptop.toml
```

- **First run:** The script will ask: *"Download and prepare missing datasets? [y/N]"*. Answer **`y`**. It will download **dev-clean** (~337 MB), convert to WAV, run MFA alignment, then start training.
- **Later runs:** If `training/data/prepared/dev-clean` is already there, it will start training without prompting.

The quick recipe uses **dev-clean only**, **CPU**, and **4 data workers**. A full epoch on a laptop may take on the order of tens of minutes depending on CPU.

## 4b. UK English

To train for **UK English** (British phoneme set and viseme mapping):

1. **Download UK MFA models** (in the same `mfa` micromamba env). There is no UK-only acoustic model; use the generic English acoustic model with the UK dictionary:

   ```bash
   micromamba activate mfa
   mfa model download dictionary english_uk_mfa
   mfa model download g2p english_uk_mfa
   mfa model download acoustic english_mfa
   micromamba deactivate
   ```

2. **Run training with the UK recipe** (from project root). Set env vars so alignment uses the UK dictionary (and UK phone set) with the generic English acoustic model:

   ```bash
   MFA_ACOUSTIC_MODEL=english_mfa MFA_DICTIONARY_MODEL=english_uk_mfa uv run python training/train.py --config training/recipes/tcn_quick_laptop_uk.toml
   ```

   The UK recipe uses `training/configs/viseme_map_en_uk_mfa.json`, which maps the UK MFA phone set (IPA-style symbols) to the same 15 visemes. When prompted to download/prepare data, answer **`y`**; alignment will run with the UK model.

## 5. Optional: use GPU

Edit `training/recipes/tcn_quick_laptop.toml` and set:

```toml
[hardware]
device = "cuda"   # or "mps" on Apple Silicon
```

If CUDA/MPS isn’t available, the trainer falls back to CPU and logs a warning.

## 6. Where outputs go

- **Checkpoints:** `training/runs/<run_name>/checkpoints/` (e.g. `best_model.pt`)
- **TensorBoard:** `training/runs/<run_name>/`. View with:
  ```bash
  uv run tensorboard --logdir training/runs
  ```

## 7. Run C# inference with training audio

**Just checking visemes (e.g. with your own WAVs)?** You do **not** need dev-clean, Python, MFA, or any training setup. You only need:

1. **Clone the repo** (and `git checkout develop` if that’s the branch with the export).
2. **.NET 8 SDK** – install from [dotnet.microsoft.com/download](https://dotnet.microsoft.com/download) or `sudo apt install dotnet-sdk-8` / `brew install dotnet@8`. Check with `dotnet --version`.
3. Run the command from the **project root** with a path to your WAV (see “Checking visemes” below). The repo already includes an ONNX model under `export/`, so no LFS or training is required.

Dev-clean / training data is only needed if you want to run training yourself or use a dev-clean WAV as a sanity check (see “Example WAV” later in this section).

---

After exporting a model to ONNX (e.g. from the training UI or export script), you can run the C# test app from the **project root** so it finds the `export/` directory and optional training WAVs.

- **Auto-pick a sample** from `training/data/prepared/dev-clean`:
  ```bash
  dotnet run --project inference/OpenLipSync.Inference/OpenLipSync.Inference.Test/OpenLipSync.Inference.Test.csproj -c Release
  ```
- **Use a specific WAV file** (e.g. from training data):
  ```bash
  dotnet run --project inference/OpenLipSync.Inference/OpenLipSync.Inference.Test/OpenLipSync.Inference.Test.csproj -c Release -- training/data/prepared/dev-clean/3752-4943-0028.wav
  ```

The app loads the newest ONNX model under `export/` (and its `config.json`), applies per-utterance mel normalization, runs the TCN, and prints viseme values at 0%, 25%, 50%, 75%, and 100% of the clip. It also writes **CSV** and **HTML** next to the WAV so you can inspect or chart all frames. Requires .NET 8 SDK.

### Checking visemes (quick run)

**1. Run with your test WAV files**

From the **project root** (so `export/` and the ONNX model are found):

```bash
dotnet run --project inference/OpenLipSync.Inference/OpenLipSync.Inference.Test/OpenLipSync.Inference.Test.csproj -c Release -- /path/to/your-audio.wav
```

Example with a file in Downloads:

```bash
dotnet run --project inference/OpenLipSync.Inference/OpenLipSync.Inference.Test/OpenLipSync.Inference.Test.csproj -c Release -- ~/Downloads/julian-01.wav
```

Outputs are written **next to the WAV** (same folder): `visemes_<basename>.csv` and `visemes_<basename>.html`.

**2. Evaluating the CSV output**

- **Location:** Same directory as the WAV, e.g. `visemes_julian-01.csv`.
- **Format:** Header row: `time_ms,sil,PP,FF,TH,DD,kk,CH,SS,nn,RR,aa,E,ih,oh,ou`. Each following row is one time step (frame); values are 0–1 activations.
- **What to check:** Open in Excel or a text editor. During speech you should see non-zero values in several viseme columns (e.g. `aa`, `E`, `ou`); near silence, `sil` should be high (e.g. > 0.9) and others low. Time should advance by ~21 ms per row (model frame rate). If every row is `sil=1` and the rest 0, something is wrong (e.g. model not loaded or normalization missing).

**3. Evaluating the HTML output**

- **Location:** Same directory as the WAV, e.g. `visemes_julian-01.html`.
- **How to use:** Open the file in a browser (double-click or File → Open). You get a line chart: x-axis = time (seconds), y-axis = activation (0–1), one series per viseme.
- **What to check:** During speech, multiple lines should move (e.g. sil dips, aa/E/ou rise). Use **All on** / **All off** to show or hide all series; click a viseme name in the legend to toggle that line. If the chart is flat (only silence) for a file you know has speech, the pipeline or model is wrong.

**4. Console output**

The app also prints per-utterance normalization (mean, std) and five sample lines at 0%, 25%, 50%, 75%, and 100% of the clip (e.g. `sil=0.14 aa=0.43 E=0.26`). Use these for a quick sanity check without opening CSV/HTML.

**Example WAV with known-ish outputs (sanity check)**

The repo does **not** ship a golden WAV with committed “expected” visemes (training data is gitignored). You can still sanity-check the pipeline:

- **If you have prepared training data:** After running section 4 (training) once, you’ll have `training/data/prepared/dev-clean/` with WAVs and MFA alignment. Run the C# app on one of those WAVs, e.g.:
  ```bash
  dotnet run --project inference/OpenLipSync.Inference/OpenLipSync.Inference.Test/OpenLipSync.Inference.Test.csproj -c Release -- training/data/prepared/dev-clean/1272-128104-0000.wav
  ```
  The model was trained on this data, so you should see varied visemes during speech (not all silence). The MFA alignment in the matching `.json` is the “ground truth” used for training; you can compare the CSV roughly to that (same time ranges should show the corresponding visemes).

- **Without training data:** Use any of your own WAVs and inspect the CSV/HTML as above; there’s no committed reference output to diff against.

If the app says **"Model not found"**, ensure there is an ONNX export under `export/` (e.g. `export/quick_laptop_uk_15ep_*/model.onnx` and `config.json`). The app uses the **newest** `model.onnx` under `export/`.

## 8. Real-time mic → visemes → MQTT (test harness)

A Python test harness streams live microphone audio through the ONNX model and publishes viseme activations as JSON to an MQTT broker.

**Install extra deps (from project root):**

```bash
uv sync --extra realtime
```

**Run (default: newest model under `export/`, broker `mqtt.dynamicdevices.co.uk:1883`, topic `openlipsync/visemes`):**

```bash
uv run python tools/realtime_viseme_mqtt.py
```

**MQTT usage:** Messages are published to `openlipsync/visemes/<client_id>` (client ID is stable per device; override with `--client-id`). Each message is JSON with: `t` (Unix time), `frame`, `client_id`, `visemes` (per-frame activations, normalised to sum=1), and `visemes_peak` (running max over ~1 s, normalised to sum=1). Subscribe to all clients with `mosquitto_sub -h mqtt.dynamicdevices.co.uk -t 'openlipsync/visemes/#'` or to one client with `openlipsync/visemes/<client_id>`.

**Options:**

- `--broker HOST` — MQTT broker host (default: mqtt.dynamicdevices.co.uk)
- `--port PORT` — MQTT port (default: 1883)
- `--topic TOPIC` — MQTT topic prefix; the connection’s client ID is appended so each client has a unique topic (default: openlipsync/visemes → openlipsync/visemes/&lt;client_id&gt;)
- `--client-id ID` — MQTT client ID (default: MAC-derived, e.g. olips-a1b2c3d4e5f6)
- `--warmup SECS` — Seconds of audio used to compute mel normalization stats (default: 1.0)
- `--publish-every N` — Publish every N frames (1 = every ~10 ms; 5 = every ~50 ms)
- `--model-dir PATH` — Use a specific export dir instead of the newest under `export/`
- `--device NAME` — Sounddevice input device (e.g. list with `python -c "import sounddevice; print(sounddevice.query_devices())"`)

**Phoneme check:** Run with `--phoneme-test`. Each segment is either silence (segment 1) or a test phoneme played from pre-generated files (segment 2–15). The mic captures that audio and the harness reports **peak** viseme activations and ✓/✗ (expected in top 2). Add `--speak` to play the clips: generate once with `uv run --extra realtime python tools/generate_phoneme_prompts.py` (writes `data/phoneme_prompts/`: segment_01.wav = silence, segment_02..15.mp3 = TTS e, ah, eh, oh, oo, p, f, thin, t, k, sh, s, n, r, each repeated 5×). Playback runs in the background; need **ffplay**, **afplay**, or **mpv**.

**JSON payload shape:** Each message is per-frame viseme activations **normalised so each set sums to 1** (including silence). Fields: `t`, `frame`, `client_id`, `visemes` (name → 0–1, sum=1), and `visemes_peak` (running max over last ~1 s, then normalised to sum=1). Example:

```json
{"t": 1739123456.78, "frame": 42, "client_id": "openlipsync-a1b2c3d4", "visemes": {"silence": 0.9, "PP": 0.01, "aa": 0.02, ...}, "visemes_peak": {"silence": 0.95, "PP": 0.02, "aa": 0.45, ...}}
```

Normalization uses a short warmup (default 1 s) to estimate mean/std over the mic input; then those stats are fixed for the rest of the session. Stop with Ctrl+C.

**Improving the phoneme test:** The default clips are TTS (Edge TTS). In practice, silence and SS often peak high (playback + mic setup), so the pass rate can be low even when the correct viseme fires. Using **human-recorded** phoneme clips (same filenames in `data/phoneme_prompts/`: segment_01.wav … segment_15.mp3) would improve realism and pass rate. See TODO.md (phoneme test audio).

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **Python version** | Project expects 3.13+. Use `uv sync` so uv picks a matching interpreter, or install Python 3.13 and run with that. |
| **MFA not found** | Ensure the `run_mfa_alignment_prepared.sh` is run from the project root and that the `mfa` env exists (`micromamba env list \| grep mfa`). |
| **No data** | Run from project root so paths like `training/data/` and `training/configs/viseme_map_en_us_arpa.json` resolve. When prompted, choose **y** to download and prepare dev-clean. |
| **Out of memory** | In `tcn_quick_laptop.toml` set `batch_size = 8` and `num_workers = 2`. |

To run a **full** training setup (e.g. train-clean-100 or larger), use `training/recipes/tcn_config.toml` and follow the main README for data splits and disk space.
