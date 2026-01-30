#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <complex>

struct wav_file {
    std::vector<float> data;
    int sample_rate;
    int channels;
    int bits_per_sample;
};

bool read_wav(const std::string & fname, wav_file & wav);

// Simple FFT implementation
void fft(std::vector<std::complex<float>>& x);

// Mel filterbank features
std::vector<float> compute_fbank(const std::vector<float>& waveform, int sample_rate, int n_mels=80);
