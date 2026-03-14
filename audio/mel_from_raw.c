/**
 * mel_from_raw.c - Read raw float32 audio, write mel spectrogram to file.
 * Used by validate_mel.py to compare C output with torchaudio.
 *
 * Usage: mel_from_raw < input.raw > output.raw
 *   or:  mel_from_raw input.raw output.raw
 *
 * Input: raw float32, mono, 16000 Hz (or config default)
 * Output: raw float32, row-major (num_frames, n_mels)
 */

#include "mel_spectrogram.h"
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_SAMPLE_RATE 16000
#define BUF_SAMPLES 16000  /* 1 second chunks for reading */

static int read_raw_audio(FILE *f, float **out, size_t *num_samples)
{
    size_t cap = BUF_SAMPLES;
    size_t n = 0;
    float *buf = (float *)malloc(cap * sizeof(float));
    if (!buf) return -1;

    while (fread(buf + n, sizeof(float), 1, f) == 1) {
        n++;
        if (n >= cap) {
            cap *= 2;
            float *tmp = (float *)realloc(buf, cap * sizeof(float));
            if (!tmp) { free(buf); return -1; }
            buf = tmp;
        }
    }

    *out = buf;
    *num_samples = n;
    return 0;
}

int main(int argc, char **argv)
{
    MelSpectrogramConfig config = {
        .sample_rate = DEFAULT_SAMPLE_RATE,
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

    float *audio = NULL;
    size_t num_samples = 0;

    if (argc >= 3) {
        FILE *fin = fopen(argv[1], "rb");
        if (!fin) { perror(argv[1]); mel_spectrogram_free(); return 1; }
        if (read_raw_audio(fin, &audio, &num_samples) != 0) {
            fclose(fin); mel_spectrogram_free(); return 1;
        }
        fclose(fin);

        size_t num_frames;
        size_t max_frames = 1 + (num_samples - config.window_length_samples) / config.hop_length_samples;
        float *mel = (float *)malloc(max_frames * config.n_mels * sizeof(float));
        if (!mel) { free(audio); mel_spectrogram_free(); return 1; }

        int n = mel_spectrogram_process(audio, num_samples, mel, &num_frames);
        free(audio);
        if (n < 0) { free(mel); mel_spectrogram_free(); return 1; }

        FILE *fout = fopen(argv[2], "wb");
        if (!fout) { perror(argv[2]); free(mel); mel_spectrogram_free(); return 1; }
        fwrite(mel, sizeof(float), num_frames * config.n_mels, fout);
        fclose(fout);
        free(mel);
    } else {
        /* stdin -> stdout */
        if (read_raw_audio(stdin, &audio, &num_samples) != 0) {
            mel_spectrogram_free(); return 1;
        }

        size_t num_frames;
        size_t max_frames = 1 + (num_samples - config.window_length_samples) / config.hop_length_samples;
        float *mel = (float *)malloc(max_frames * config.n_mels * sizeof(float));
        if (!mel) { free(audio); mel_spectrogram_free(); return 1; }

        int n = mel_spectrogram_process(audio, num_samples, mel, &num_frames);
        free(audio);
        if (n < 0) { free(mel); mel_spectrogram_free(); return 1; }

        fwrite(mel, sizeof(float), num_frames * config.n_mels, stdout);
        free(mel);
    }

    mel_spectrogram_free();
    return 0;
}
