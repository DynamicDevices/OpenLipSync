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

From the **project root** (so `export/` and the ONNX model are found), run with your own WAV:

```bash
dotnet run --project inference/OpenLipSync.Inference/OpenLipSync.Inference.Test/OpenLipSync.Inference.Test.csproj -c Release -- /path/to/your-audio.wav
```

Example with a file in Downloads:

```bash
dotnet run --project inference/OpenLipSync.Inference/OpenLipSync.Inference.Test/OpenLipSync.Inference.Test.csproj -c Release -- ~/Downloads/julian-01.wav
```

**What you get**

- **Console:** Per-utterance normalization stats, then viseme lines at five time points (start, 25%, 50%, 75%, end). Each line lists which visemes are active (e.g. `sil=0.14 aa=0.43 E=0.26`).
- **Same folder as the WAV:**
  - `visemes_<basename>.csv` – every frame: `time_ms`, then 15 columns (sil, PP, FF, …). Open in Excel or use for your own plots.
  - `visemes_<basename>.html` – open in a browser: time (s) vs activation (0–1) for all 15 visemes. Use **All on** / **All off** or click the legend to show/hide series.

If the app says "Model not found", ensure there is an ONNX export under `export/` (e.g. `export/quick_laptop_uk_15ep_*/model.onnx` and `config.json`). The app uses the **newest** `model.onnx` under `export/`.

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **Python version** | Project expects 3.13+. Use `uv sync` so uv picks a matching interpreter, or install Python 3.13 and run with that. |
| **MFA not found** | Ensure the `run_mfa_alignment_prepared.sh` is run from the project root and that the `mfa` env exists (`micromamba env list \| grep mfa`). |
| **No data** | Run from project root so paths like `training/data/` and `training/configs/viseme_map_en_us_arpa.json` resolve. When prompted, choose **y** to download and prepare dev-clean. |
| **Out of memory** | In `tcn_quick_laptop.toml` set `batch_size = 8` and `num_workers = 2`. |

To run a **full** training setup (e.g. train-clean-100 or larger), use `training/recipes/tcn_config.toml` and follow the main README for data splits and disk space.
