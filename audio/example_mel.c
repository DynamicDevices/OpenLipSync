/**
 * example_mel.c - Example usage and minimal test of mel_spectrogram
 *
 * Build: gcc -o example_mel example_mel.c mel_spectrogram.c -lm
 * Run:   ./example_mel
 *
 * Generates 1 second of test tone, computes mel spectrogram, prints first frame.
 */

#include "mel_spectrogram.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void)
{
    MelSpectrogramConfig config = {
        .sample_rate = 16000,
        .hop_length_samples = 160,
        .window_length_samples = 400,
        .n_fft = 1024,
        .n_mels = 80,
        .fmin = 50.0f,
        .fmax = 8000.0f,
        .top_db = 80.0f,
    };

    if (mel_spectrogram_init(&config) != 0) {
        fprintf(stderr, "mel_spectrogram_init failed\n");
        return 1;
    }

    /* 1 second of 440 Hz test tone */
    size_t num_samples = 16000;
    float *audio = (float *)malloc(num_samples * sizeof(float));
    if (!audio) {
        mel_spectrogram_free();
        return 1;
    }

    for (size_t i = 0; i < num_samples; i++) {
        audio[i] = 0.3f * sinf(2.0f * (float)M_PI * 440.0f * (float)i / 16000.0f);
    }

    /* Compute mel spectrogram */
    size_t num_frames;
    size_t max_frames = 1 + (num_samples - config.window_length_samples) / config.hop_length_samples;
    float *mel = (float *)malloc(max_frames * config.n_mels * sizeof(float));
    if (!mel) {
        free(audio);
        mel_spectrogram_free();
        return 1;
    }

    int n = mel_spectrogram_process(audio, num_samples, mel, &num_frames);
    if (n < 0) {
        fprintf(stderr, "mel_spectrogram_process failed\n");
        free(audio);
        free(mel);
        mel_spectrogram_free();
        return 1;
    }

    printf("Computed %zu mel frames from %zu samples\n", num_frames, num_samples);
    printf("First frame (first 10 of %d mel bins):\n", config.n_mels);
    for (int i = 0; i < 10 && i < config.n_mels; i++) {
        printf("  mel[%d] = %.4f\n", i, mel[i]);
    }

    free(audio);
    free(mel);
    mel_spectrogram_free();
    printf("OK\n");
    return 0;
}
