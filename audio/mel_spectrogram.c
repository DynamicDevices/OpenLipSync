/**
 * mel_spectrogram.c - Mel spectrogram implementation for OpenLipSync
 *
 * Matches torchaudio MelSpectrogram (power=2.0) + AmplitudeToDB (stype="power", top_db=80).
 * Uses Hann window, HTK mel scale, and triangular filterbank.
 */

#include "mel_spectrogram.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AMIN 1e-10f

/* State (singleton for simplicity; could be made opaque with context ptr) */
static MelSpectrogramConfig s_config;
static float *s_window = NULL;
static float *s_filterbank = NULL;  /* n_mels * (n_fft/2 + 1), row-major */
static MelComplex *s_fft_buffer = NULL;
static float *s_power_spectrum = NULL;
static MelFFTCallback s_fft_callback = NULL;
static void *s_fft_userdata = NULL;
static int s_initialized = 0;

/* Forward declarations */
static void builtin_fft(MelComplex *buf, int n);
static float hz_to_mel(float hz);
static float mel_to_hz(float mel);

/* --- Hann window --- */
static void create_hann_window(void)
{
    int n = s_config.window_length_samples;
    s_window = (float *)malloc((size_t)n * sizeof(float));
    if (!s_window) return;

    for (int i = 0; i < n; i++) {
        s_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1)));
    }
}

/* --- Mel scale (HTK/Slaney) --- */
static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* --- Mel filterbank ---
 * Triangular filters between mel-spaced center frequencies.
 * Matches torchaudio default (no Slaney norm) and C# MelSpectrogramProcessor.
 */
static int create_mel_filterbank(void)
{
    int n_mels = s_config.n_mels;
    int n_fft = s_config.n_fft;
    int n_bins = n_fft / 2 + 1;
    float sr = (float)s_config.sample_rate;
    float fmin = s_config.fmin;
    float fmax = s_config.fmax;

    s_filterbank = (float *)malloc((size_t)n_mels * (size_t)n_bins * sizeof(float));
    if (!s_filterbank) return -1;
    memset(s_filterbank, 0, (size_t)n_mels * (size_t)n_bins * sizeof(float));

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    /* n_mels+2 points linearly spaced in mel */
    float *mel_pts = (float *)malloc((size_t)(n_mels + 2) * sizeof(float));
    float *hz_pts = (float *)malloc((size_t)(n_mels + 2) * sizeof(float));
    float *bin_pts = (float *)malloc((size_t)(n_mels + 2) * sizeof(float));
    if (!mel_pts || !hz_pts || !bin_pts) {
        free(mel_pts);
        free(hz_pts);
        free(bin_pts);
        return -1;
    }

    /* FFT bin k corresponds to frequency k * sample_rate / n_fft */
    for (int i = 0; i < n_mels + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mels + 1);
        hz_pts[i] = mel_to_hz(mel_pts[i]);
        bin_pts[i] = (float)n_fft * hz_pts[i] / sr;
    }

    for (int mel = 0; mel < n_mels; mel++) {
        float left = bin_pts[mel];
        float center = bin_pts[mel + 1];
        float right = bin_pts[mel + 2];
        float eps = 1e-6f;

        for (int bin = 0; bin < n_bins; bin++) {
            float b = (float)bin;
            if (b >= left && b <= center) {
                float denom = center - left;
                s_filterbank[mel * n_bins + bin] = (denom > eps) ? (b - left) / denom : 0.0f;
            } else if (b > center && b <= right) {
                float denom = right - center;
                s_filterbank[mel * n_bins + bin] = (denom > eps) ? (right - b) / denom : 0.0f;
            }
        }
    }

    free(mel_pts);
    free(hz_pts);
    free(bin_pts);
    return 0;
}

/* --- Built-in radix-2 Cooley-Tukey FFT (in-place) --- */
static void builtin_fft(MelComplex *buf, int n)
{
    if (n <= 1) return;

    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j) {
            MelComplex tmp = buf[i];
            buf[i] = buf[j];
            buf[j] = tmp;
        }
    }

    /* Cooley-Tukey */
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * (float)M_PI / (float)len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);

        for (int i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                MelComplex u = buf[i + j];
                MelComplex v;
                v.real = wr * buf[i + j + len/2].real - wi * buf[i + j + len/2].imag;
                v.imag = wr * buf[i + j + len/2].imag + wi * buf[i + j + len/2].real;

                buf[i + j].real = u.real + v.real;
                buf[i + j].imag = u.imag + v.imag;
                buf[i + j + len/2].real = u.real - v.real;
                buf[i + j + len/2].imag = u.imag - v.imag;

                float nwr = wr * wlen_r - wi * wlen_i;
                float nwi = wr * wlen_i + wi * wlen_r;
                wr = nwr;
                wi = nwi;
            }
        }
    }
}

/* --- Power spectrum from FFT output (first n_fft/2+1 bins) --- */
static void compute_power_spectrum(const MelComplex *fft_out, float *power, int n_fft)
{
    int n_bins = n_fft / 2 + 1;
    for (int i = 0; i < n_bins; i++) {
        float re = fft_out[i].real;
        float im = fft_out[i].imag;
        power[i] = re * re + im * im;
    }
}

/* --- Amplitude to dB (power scale, top_db clamp) ---
 * Matches torchaudio: max value -> 0 dB, min clamped to -top_db.
 * ref = max(amin, max_val * 10^(-top_db/10))
 * out = 10 * log10(max(x, ref)) - 10 * log10(max_val)
 */
static void power_to_db(float *mel_frame, int n_mels, float top_db)
{
    float max_val = AMIN;
    for (int i = 0; i < n_mels; i++) {
        if (mel_frame[i] > max_val) max_val = mel_frame[i];
    }

    float ref = max_val * powf(10.0f, -top_db / 10.0f);
    if (ref < AMIN) ref = AMIN;

    float log_max = 10.0f * log10f(max_val);
    for (int i = 0; i < n_mels; i++) {
        float x = mel_frame[i];
        if (x < ref) x = ref;
        mel_frame[i] = 10.0f * log10f(x) - log_max;
    }
}

/* --- Public API --- */
int mel_spectrogram_init(const MelSpectrogramConfig *config)
{
    if (s_initialized) {
        mel_spectrogram_free();
    }

    if (!config) return -1;
    if (config->n_fft <= 0 || (config->n_fft & (config->n_fft - 1)) != 0)
        return -1;
    if (config->window_length_samples <= 0 || config->hop_length_samples <= 0)
        return -1;
    if (config->n_mels <= 0) return -1;
    if (config->fmax <= config->fmin) return -1;

    memcpy(&s_config, config, sizeof(MelSpectrogramConfig));

    create_hann_window();
    if (!s_window) return -1;

    if (create_mel_filterbank() != 0) return -1;

    s_fft_buffer = (MelComplex *)malloc((size_t)config->n_fft * sizeof(MelComplex));
    s_power_spectrum = (float *)malloc((size_t)(config->n_fft / 2 + 1) * sizeof(float));
    if (!s_fft_buffer || !s_power_spectrum) return -1;

    s_initialized = 1;
    return 0;
}

void mel_spectrogram_free(void)
{
    free(s_window);
    s_window = NULL;
    free(s_filterbank);
    s_filterbank = NULL;
    free(s_fft_buffer);
    s_fft_buffer = NULL;
    free(s_power_spectrum);
    s_power_spectrum = NULL;
    s_fft_callback = NULL;
    s_fft_userdata = NULL;
    s_initialized = 0;
}

void mel_spectrogram_set_fft_callback(MelFFTCallback callback, void *userdata)
{
    s_fft_callback = callback;
    s_fft_userdata = userdata;
}

int mel_spectrogram_process_frame(const float *audio, float *mel_out)
{
    if (!s_initialized || !audio || !mel_out) return -1;

    int n_fft = s_config.n_fft;
    int n_mels = s_config.n_mels;
    int n_bins = n_fft / 2 + 1;
    int win_len = s_config.window_length_samples;

    /* Copy, window first win_len samples, zero-pad to n_fft (matches torch.stft) */
    memset(s_fft_buffer, 0, (size_t)n_fft * sizeof(MelComplex));
    for (int i = 0; i < win_len; i++) {
        s_fft_buffer[i].real = audio[i] * s_window[i];
        s_fft_buffer[i].imag = 0.0f;
    }
    /* samples win_len..n_fft-1 remain zero (audio not used beyond win_len) */

    /* FFT */
    if (s_fft_callback) {
        s_fft_callback(s_fft_buffer, n_fft, s_fft_userdata);
    } else {
        builtin_fft(s_fft_buffer, n_fft);
    }

    /* Power spectrum */
    compute_power_spectrum(s_fft_buffer, s_power_spectrum, n_fft);

    /* Mel filterbank */
    for (int mel = 0; mel < n_mels; mel++) {
        float sum = 0.0f;
        for (int bin = 0; bin < n_bins; bin++) {
            sum += s_power_spectrum[bin] * s_filterbank[mel * n_bins + bin];
        }
        mel_out[mel] = sum;
    }

    /* Power to dB */
    power_to_db(mel_out, n_mels, s_config.top_db);

    return 0;
}

int mel_spectrogram_process(const float *audio, size_t num_samples,
                            float *mel_out, size_t *num_frames_out)
{
    if (!s_initialized || !audio || !mel_out || !num_frames_out) return -1;

    int hop = s_config.hop_length_samples;
    int n_fft = s_config.n_fft;
    int n_mels = s_config.n_mels;

    /* Frame extent is n_fft (matches torch.stft) */
    if (num_samples < (size_t)n_fft) {
        *num_frames_out = 0;
        return 0;
    }

    size_t num_frames = 1 + (num_samples - (size_t)n_fft) / (size_t)hop;
    *num_frames_out = num_frames;

    for (size_t t = 0; t < num_frames; t++) {
        const float *frame = audio + t * (size_t)hop;
        float *out = mel_out + t * (size_t)n_mels;
        if (mel_spectrogram_process_frame(frame, out) != 0)
            return -1;
    }

    return (int)num_frames;
}

void mel_spectrogram_get_config(MelSpectrogramConfig *out)
{
    if (out) memcpy(out, &s_config, sizeof(MelSpectrogramConfig));
}
