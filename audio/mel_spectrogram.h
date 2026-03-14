/**
 * mel_spectrogram.h - Mel spectrogram computation for OpenLipSync
 *
 * Converts audio waveform to mel-scale log-power spectrogram features,
 * matching torchaudio.transforms.MelSpectrogram + AmplitudeToDB behavior.
 *
 * Parameters align with training/config:
 *   sample_rate=16000, hop_length=160 (10ms), window_length=400 (25ms),
 *   n_fft=1024, n_mels=80, fmin=50, fmax=8000, power=2.0, top_db=80
 *
 * The FFT can be provided by the caller via mel_spectrogram_set_fft_callback().
 * If not set, a built-in radix-2 FFT is used.
 */

#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration matching OpenLipSync training */
typedef struct {
    int sample_rate;           /* e.g. 16000 */
    int hop_length_samples;     /* e.g. 160 (10ms at 16kHz) */
    int window_length_samples; /* e.g. 400 (25ms at 16kHz) */
    int n_fft;                  /* FFT size, power of 2, e.g. 1024 */
    int n_mels;                 /* Mel bands, e.g. 80 */
    float fmin;                  /* Lowest mel frequency in Hz, e.g. 50 */
    float fmax;                 /* Highest mel frequency in Hz, e.g. 8000 */
    float top_db;               /* Dynamic range clamp in dB, e.g. 80 */
} MelSpectrogramConfig;

/* Complex number for FFT */
typedef struct {
    float real;
    float imag;
} MelComplex;

/**
 * FFT callback: compute forward FFT in-place.
 * Input: complex buffer of length n_fft (real part = waveform, imag = 0).
 * Output: complex spectrum in same buffer.
 * The callback may modify the buffer in-place.
 */
typedef void (*MelFFTCallback)(MelComplex *buffer, int n_fft, void *userdata);

/**
 * Initialize mel spectrogram state. Allocates window, filterbank, and buffers.
 * Returns 0 on success, -1 on error (invalid config or allocation failure).
 */
int mel_spectrogram_init(const MelSpectrogramConfig *config);

/**
 * Free allocated resources. Safe to call even if init failed.
 */
void mel_spectrogram_free(void);

/**
 * Set custom FFT callback. If NULL, uses built-in radix-2 FFT.
 * Must be called before mel_spectrogram_process().
 */
void mel_spectrogram_set_fft_callback(MelFFTCallback callback, void *userdata);

/**
 * Process a single frame of audio.
 * Input: window_length_samples of float audio (mono, [-1,1] range).
 * Output: n_mels floats written to mel_out (log-power in dB).
 * Returns 0 on success, -1 on error.
 */
int mel_spectrogram_process_frame(const float *audio, float *mel_out);

/**
 * Process a full waveform and produce mel spectrogram.
 * Input: num_samples of float audio.
 * Output: mel_out must hold (num_frames * n_mels) floats, row-major (time, mel).
 * Returns number of frames produced, or -1 on error.
 * num_frames = 1 + (num_samples - window_length_samples) / hop_length_samples
 */
int mel_spectrogram_process(const float *audio, size_t num_samples,
                            float *mel_out, size_t *num_frames_out);

/**
 * Get config values (e.g. n_mels for output buffer sizing).
 */
void mel_spectrogram_get_config(MelSpectrogramConfig *out);

#ifdef __cplusplus
}
#endif

#endif /* MEL_SPECTROGRAM_H */
