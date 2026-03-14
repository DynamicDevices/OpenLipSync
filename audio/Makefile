# Mel spectrogram C implementation - OpenLipSync
# Build: make
# Test:  make test && ./example_mel

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

.PHONY: all clean test

all: example_mel mel_from_raw

example_mel: example_mel.c mel_spectrogram.c mel_spectrogram.h
	$(CC) $(CFLAGS) -o $@ example_mel.c mel_spectrogram.c $(LDFLAGS)

mel_from_raw: mel_from_raw.c mel_spectrogram.c mel_spectrogram.h
	$(CC) $(CFLAGS) -o $@ mel_from_raw.c mel_spectrogram.c $(LDFLAGS)

clean:
	rm -f example_mel mel_from_raw

test: example_mel
	./example_mel
