#include "common_audio.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cstdint>

bool read_wav(const std::string & fname, wav_file & wav) {
    std::ifstream f(fname, std::ios::binary);
    if (!f.is_open()) return false;

    char buf[4];
    f.read(buf, 4); if (strncmp(buf, "RIFF", 4) != 0) return false;
    f.ignore(4);
    f.read(buf, 4); if (strncmp(buf, "WAVE", 4) != 0) return false;
    f.read(buf, 4); if (strncmp(buf, "fmt ", 4) != 0) return false;
    
    uint32_t fmt_chunk_size;
    f.read((char*)&fmt_chunk_size, 4);
    
    uint16_t audio_format, channels, bits_per_sample;
    uint32_t sample_rate;
    f.read((char*)&audio_format, 2);
    f.read((char*)&channels, 2);
    f.read((char*)&sample_rate, 4);
    f.ignore(6);
    f.read((char*)&bits_per_sample, 2);
    
    if (fmt_chunk_size > 16) f.ignore(fmt_chunk_size - 16);
    
    while (f.read(buf, 4)) {
        uint32_t chunk_size;
        f.read((char*)&chunk_size, 4);
        
        if (strncmp(buf, "data", 4) == 0) {
            wav.sample_rate = sample_rate;
            wav.channels = channels;
            wav.bits_per_sample = bits_per_sample;
            
            int num_samples = chunk_size / (channels * bits_per_sample / 8);
            wav.data.resize(num_samples * channels);
            
            if (bits_per_sample == 16) {
                for (int i = 0; i < num_samples * channels; ++i) {
                    int16_t val;
                    f.read((char*)&val, 2);
                    wav.data[i] = val / 32768.0f;
                }
            } else {
                fprintf(stderr, "Unsupported bit depth: %d\n", bits_per_sample);
                return false;
            }
            return true;
        } else {
            f.ignore(chunk_size);
        }
    }
    return false;
}

// Radix-2 FFT (in-place, Cooley-Tukey) - more efficient than recursive version
void fft(std::vector<std::complex<float>>& x) {
    int n = x.size();
    if (n <= 1) return;

    // Bit-reversal permutation
    int log2n = 0;
    while ((1 << log2n) < n) log2n++;
    
    for (int i = 0; i < n; ++i) {
        int j = 0;
        for (int k = 0; k < log2n; ++k) {
            j |= ((i >> k) & 1) << (log2n - 1 - k);
        }
        if (j > i) std::swap(x[i], x[j]);
    }
    
    // Cooley-Tukey iterative FFT
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        std::complex<float> wm = std::polar(1.0f, -2.0f * (float)M_PI / m);
        for (int k = 0; k < n; k += m) {
            std::complex<float> w = 1.0f;
            for (int j = 0; j < m / 2; ++j) {
                std::complex<float> t = w * x[k + j + m/2];
                std::complex<float> u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m/2] = u - t;
                w *= wm;
            }
        }
    }
}

// Kaldi-compatible fbank implementation
// Matches torchaudio.compliance.kaldi.fbank with:
//   sample_frequency=16000, frame_length=25.0, frame_shift=10.0,
//   num_mel_bins=80, window_type='hamming', use_energy=False,
//   htk_compat=True, use_log_fbank=True, subtract_mean=False (in Kaldi fbank)
// BUT pyannote applies CMVN (mean subtraction) after fbank computation!
std::vector<float> compute_fbank(const std::vector<float>& waveform_in, int sample_rate, int n_mels) {
    // WeSpeaker scales input by 2^15
    std::vector<float> waveform = waveform_in;
    for (float & v : waveform) {
        v *= 32768.0f;
    }

    // Frame parameters (matching Kaldi defaults)
    float frame_length_ms = 25.0f;
    float frame_shift_ms = 10.0f;
    float preemphasis_coeff = 0.97f;
    float low_freq = 20.0f;
    float high_freq = 0.0f;  // 0 means Nyquist
    bool htk_compat = true;
    bool remove_dc_offset = true; // subtract-mean=true in feature extraction usually
    bool use_power = true;
    
    int frame_length = static_cast<int>(frame_length_ms * sample_rate / 1000.0f);
    int frame_shift = static_cast<int>(frame_shift_ms * sample_rate / 1000.0f);
    
    // Round to power of 2 for FFT
    int fft_size = 1;
    while (fft_size < frame_length) fft_size *= 2;
    
    // Compute number of frames (snip_edges=True behavior)
    int num_frames = 0;
    if ((int)waveform.size() >= frame_length) {
        num_frames = 1 + (waveform.size() - frame_length) / frame_shift;
    }
    
    if (num_frames == 0) {
        return std::vector<float>();
    }
    
    std::vector<float> fbank(num_frames * n_mels);
    
    // Compute actual high frequency
    float nyquist = sample_rate / 2.0f;
    if (high_freq <= 0.0f) {
        high_freq = nyquist + high_freq;
    }
    if (high_freq > nyquist) high_freq = nyquist;
    
    // HTK-style mel conversion (htk_compat=True uses this)
    auto hz_to_mel = [htk_compat](float hz) -> float {
        if (htk_compat) {
            return 2595.0f * log10f(1.0f + hz / 700.0f);
        } else {
            return 1127.0f * logf(1.0f + hz / 700.0f);
        }
    };
    
    auto mel_to_hz = [htk_compat](float mel) -> float {
        if (htk_compat) {
            return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
        } else {
            return 700.0f * (expf(mel / 1127.0f) - 1.0f);
        }
    };
    
    // Compute mel center frequencies in Hz (n_mels + 2 points for edges)
    float low_mel = hz_to_mel(low_freq);
    float high_mel = hz_to_mel(high_freq);
    
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        float mel = low_mel + (high_mel - low_mel) * i / (n_mels + 1);
        hz_points[i] = mel_to_hz(mel);
    }
    
    int num_fft_bins = fft_size / 2 + 1;
    
    // Kaldi-style mel filterbank: use continuous Hz values for weight computation
    std::vector<std::vector<float>> filterbank(n_mels);
    for (int i = 0; i < n_mels; ++i) {
        filterbank[i].resize(num_fft_bins, 0.0f);
        float left_hz = hz_points[i];
        float center_hz = hz_points[i + 1];
        float right_hz = hz_points[i + 2];
        
        for (int j = 0; j < num_fft_bins; ++j) {
            float freq = (float)j * sample_rate / fft_size;
            
            if (freq >= left_hz && freq < center_hz) {
                // Rising slope
                if (center_hz != left_hz) {
                    filterbank[i][j] = (freq - left_hz) / (center_hz - left_hz);
                }
            } else if (freq >= center_hz && freq <= right_hz) {
                // Falling slope
                if (right_hz != center_hz) {
                    filterbank[i][j] = (right_hz - freq) / (right_hz - center_hz);
                }
            }
        }
    }
    
    // Hamming window (matching window_type='hamming')
    std::vector<float> window(frame_length);
    for (int i = 0; i < frame_length; ++i) {
        // Kaldi periodic Hamming window
        window[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / frame_length);
    }
    
    // Process each frame
    for (int t = 0; t < num_frames; ++t) {
        int start = t * frame_shift;
        
        // Extract frame and optionally remove DC offset
        std::vector<float> frame(frame_length);
        for (int i = 0; i < frame_length; ++i) {
            frame[i] = waveform[start + i];
        }
        
        // Remove DC offset (subtract mean)
        if (remove_dc_offset) {
            float mean = 0.0f;
            for (int i = 0; i < frame_length; ++i) {
                mean += frame[i];
            }
            mean /= frame_length;
            for (int i = 0; i < frame_length; ++i) {
                frame[i] -= mean;
            }
        }
        
        // Pre-emphasis
        for (int i = frame_length - 1; i > 0; --i) {
            frame[i] -= preemphasis_coeff * frame[i - 1];
        }
        frame[0] *= (1.0f - preemphasis_coeff);
        
        // Apply window
        for (int i = 0; i < frame_length; ++i) {
            frame[i] *= window[i];
        }
        
        // Zero-pad to FFT size
        std::vector<std::complex<float>> fft_frame(fft_size, 0.0f);
        for (int i = 0; i < frame_length; ++i) {
            fft_frame[i] = frame[i];
        }
        
        // FFT
        fft(fft_frame);
        
        // Power spectrum (or magnitude squared)
        std::vector<float> power_spec(num_fft_bins);
        for (int i = 0; i < num_fft_bins; ++i) {
            if (use_power) {
                power_spec[i] = std::norm(fft_frame[i]);  // |x|^2
            } else {
                power_spec[i] = std::abs(fft_frame[i]);   // |x|
            }
        }
        
        // Apply mel filterbank and compute log
        for (int i = 0; i < n_mels; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < num_fft_bins; ++j) {
                sum += power_spec[j] * filterbank[i][j];
            }
            
            // Log with floor for numerical stability (Kaldi uses FLT_EPSILON)
            float energy_floor = 1.192092896e-07f;  // FLT_EPSILON
            if (sum < energy_floor) sum = energy_floor;
            fbank[t * n_mels + i] = logf(sum);
        }
    }
    
    // Apply CMVN (Mean Normalization) over time
    // For each mel bin, calculate mean over time and subtract
    for (int i = 0; i < n_mels; ++i) {
        float mean = 0.0f;
        for (int t = 0; t < num_frames; ++t) {
            mean += fbank[t * n_mels + i];
        }
        mean /= num_frames;
        for (int t = 0; t < num_frames; ++t) {
            fbank[t * n_mels + i] -= mean;
        }
    }
    
    return fbank;
}
