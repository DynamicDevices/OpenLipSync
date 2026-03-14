# TODO

## UK visemes

**Support UK visemes in addition to US.** Currently using US visemes (`viseme_map_en_us_arpa.json`). UK recipe (`tcn_full_uk.toml`) exists and uses `viseme_map_en_uk_mfa.json` with the same LibriSpeech data—run with UK MFA models for alignment. Goal: ensure both US and UK viseme models are maintained and easily trainable.

## Phoneme test audio

**Improve phoneme test pass rate and realism** by replacing TTS-generated clips with human-recorded phoneme clips.

- **Current:** `tools/generate_phoneme_prompts.py` creates segment_01.wav (silence) and segment_02..15.mp3 via Edge TTS. Playback + mic often causes silence/SS to dominate, so pass rate is low.
- **Proposed:** Have someone record the 15 segments (silence + e, ah, eh, oh, oo, p, f, thin, t, k, sh, s, n, r, each repeated several times). Save as the same filenames in `data/phoneme_prompts/` (segment_01.wav, segment_02.mp3 … segment_15.mp3). No harness changes needed.
- **Refs:** QUICKSTART “Improving the phoneme test”; `tools/generate_phoneme_prompts.py` docstring TODO.
