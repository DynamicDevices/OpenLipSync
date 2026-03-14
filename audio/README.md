# Mel Spectrogram C Implementation

A C implementation of the mel spectrogram computation used by OpenLipSync, matching **torchaudio** `MelSpectrogram` (power=2.0) and `AmplitudeToDB` (stype="power", top_db=80).

## Parameters (OpenLipSync default)

| Parameter | Value | Description |
|-----------|-------|-------------|
| sample_rate | 16000 | Hz |
| hop_length_samples | 160 | 10 ms at 16 kHz |
| window_length_samples | 400 | 25 ms at 16 kHz |
| n_fft | 1024 | FFT size (power of 2) |
| n_mels | 80 | Mel frequency bands |
| fmin | 50 | Lowest mel frequency (Hz) |
| fmax | 8000 | Highest mel frequency (Hz) |
| top_db | 80 | Dynamic range clamp (dB) |

## Algorithm

1. **Hann window**: `w[n] = 0.5 * (1 - cos(2πn/(N-1)))`
2. **FFT**: Radix-2 Cooley-Tukey (in-place)
3. **Power spectrum**: `|FFT|²` (magnitude squared)
4. **Mel filterbank**: Triangular filters, HTK mel scale (`m = 2595*log10(1+f/700)`)
5. **Power to dB**: `10*log10(max(x,ref)) - 10*log10(ref)` with `ref = max(1e-10, max_val*10^(-top_db/10))`

## Custom FFT

You can plug in your own FFT via:

```c
mel_spectrogram_set_fft_callback(my_fft_func, userdata);
```

The callback receives a complex buffer of length `n_fft` (real = waveform, imag = 0). It should compute the forward FFT in-place. If not set, the built-in radix-2 FFT is used.

## Build and test

```bash
make
./example_mel
```

## Validation

To compare against torchaudio, run from project root:

```bash
make -C audio mel_from_raw
uv run python audio/validate_mel.py
```

This generates a test tone, computes mel with both the C library and torchaudio (center=False for frame alignment), and reports the max absolute difference. Small differences (< 1 dB) may occur due to FFT implementation details; use `mel_spectrogram_set_fft_callback()` to plug in your own FFT for exact matching.

## Files

- `mel_spectrogram.h` - API and config struct
- `mel_spectrogram.c` - Implementation (FFT, mel filterbank, dB)
- `example_mel.c` - Minimal test
- `validate_mel.py` - Python validation script
