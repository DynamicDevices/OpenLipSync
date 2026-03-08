# OpenLipSync

**Experimental work-in-progress project**

An open-source, cross-platform project that converts audio input into realistic facial expressions in real-time following the [MPEG-4 (FBA)](https://visagetechnologies.com/uploads/2012/08/MPEG-4FBAOverview.pdf) standard.

**→ To try it on your laptop (small data, few epochs), see [QUICKSTART.md](QUICKSTART.md).**

## Setup for model training

Core (uv)

```bash
uv sync
```

MFA (micromamba)

```bash
micromamba create -n mfa -c conda-forge python=3.12 montreal-forced-aligner
micromamba activate mfa

mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

Dataset Download is now integrated in the training script.

**Quick (laptop) training:**  
`uv run python training/train.py --config training/recipes/tcn_quick_laptop.toml`

**Full training (production ONNX):** See [QUICKSTART.md](QUICKSTART.md) section 4c. Use `training/recipes/tcn_config.toml` (US) or `training/recipes/tcn_full_uk.toml` (UK), then export with `training/tools/export_onnx.py --run <run_name> --checkpoint best`.


This project uses the [LibriSpeech ASR corpus](https://openslr.org/12/) (CC BY 4.0 license).
