#include "pyannote/diarization.h"
#include "vbx.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kChunkSamples = 160000;
constexpr int kStepSamples = 16000;
constexpr int kFramesPerChunk = 589;
constexpr int kPowersetClasses = 7;
constexpr int kLocalSpeakers = 3;
constexpr int kEmbeddingDim = 256;
constexpr int kPldaDim = 128;
constexpr int kMelBins = 80;
constexpr int kSegGraphNodes = 8192;
constexpr int kEmbGraphNodes = 4096;

constexpr double kFrameDuration = 0.0619375;
constexpr double kFrameStep = 0.016875;
constexpr double kChunkDuration = 10.0;
constexpr double kChunkStep = 1.0;

constexpr float kLeakyReluSlope = 0.01f;
constexpr float kInstanceNormEps = 1e-5f;
constexpr float kBatchNormEps = 1e-5f;
constexpr double kGreedyClusterThreshold = 0.55;

struct SlidingWindowParams {
    double start = 0.0;
    double duration = 0.0;
    double step = 0.0;
};

struct RTTMSegment {
    double start = 0.0;
    double duration = 0.0;
    std::string speaker;
};

struct WavData {
    uint32_t sample_rate = 0;
    int channels = 0;
    std::vector<float> mono;
};

struct PldaModel {
    std::vector<double> mean1;
    std::vector<double> mean2;
    std::vector<double> lda;
    std::vector<double> mu;
    std::vector<double> tr;
    std::vector<double> psi;
    bool loaded = false;
};

struct BackendState {
    std::vector<ggml_backend_t> backends;
    ggml_backend_t preferred_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> graph_meta;
};

struct WeightStore {
    ggml_context * ctx = nullptr;
    gguf_context * gguf = nullptr;
    std::vector<ggml_backend_buffer_t> buffers;
    std::vector<ggml_backend_t> backends;
};

struct SegmentationModel {
    WeightStore weights;

    ggml_tensor * wav_norm_weight = nullptr;
    ggml_tensor * wav_norm_bias = nullptr;
    ggml_tensor * sinc_conv_weight[3] = {nullptr, nullptr, nullptr};
    ggml_tensor * sinc_conv_bias[3] = {nullptr, nullptr, nullptr};
    ggml_tensor * sinc_norm_weight[3] = {nullptr, nullptr, nullptr};
    ggml_tensor * sinc_norm_bias[3] = {nullptr, nullptr, nullptr};

    ggml_tensor * lstm_weight_ih[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * lstm_weight_hh[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * lstm_bias_ih[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * lstm_bias_hh[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * lstm_weight_ih_rev[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * lstm_weight_hh_rev[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * lstm_bias_ih_rev[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * lstm_bias_hh_rev[4] = {nullptr, nullptr, nullptr, nullptr};

    ggml_tensor * linear_weight[2] = {nullptr, nullptr};
    ggml_tensor * linear_bias[2] = {nullptr, nullptr};
    ggml_tensor * classifier_weight = nullptr;
    ggml_tensor * classifier_bias = nullptr;
};

struct EmbeddingModel {
    WeightStore weights;

    ggml_tensor * conv1_weight = nullptr;
    ggml_tensor * bn1_weight = nullptr;
    ggml_tensor * bn1_bias = nullptr;
    ggml_tensor * bn1_mean = nullptr;
    ggml_tensor * bn1_var = nullptr;

    ggml_tensor * layer_conv1_weight[4][6] = {};
    ggml_tensor * layer_bn1_weight[4][6] = {};
    ggml_tensor * layer_bn1_bias[4][6] = {};
    ggml_tensor * layer_bn1_mean[4][6] = {};
    ggml_tensor * layer_bn1_var[4][6] = {};

    ggml_tensor * layer_conv2_weight[4][6] = {};
    ggml_tensor * layer_bn2_weight[4][6] = {};
    ggml_tensor * layer_bn2_bias[4][6] = {};
    ggml_tensor * layer_bn2_mean[4][6] = {};
    ggml_tensor * layer_bn2_var[4][6] = {};

    ggml_tensor * shortcut_conv_weight[4] = {};
    ggml_tensor * shortcut_bn_weight[4] = {};
    ggml_tensor * shortcut_bn_bias[4] = {};
    ggml_tensor * shortcut_bn_mean[4] = {};
    ggml_tensor * shortcut_bn_var[4] = {};

    ggml_tensor * seg1_weight = nullptr;
    ggml_tensor * seg1_bias = nullptr;
};

struct FbankData {
    std::vector<float> data;
    int frames = 0;
};

void apply_cmn(float * data, int frames);

enum : uintptr_t {
    kSegLstmFlagCoop      = 1u << 0,
    kSegLstmFlagWarp      = 1u << 1,
    kSegLstmFlagWarpNoSh  = 1u << 2,
    kSegLstmFlagBidir     = 1u << 3,
    kSegLstmWarpShift     = 8,
    kSegLstmWarpMask      = 0xffu << kSegLstmWarpShift,
};

uintptr_t pack_seg_lstm_cuda_options(const DiarizationConfig & config) {
    uintptr_t packed = 0;
    if (config.seg_lstm_coop) {
        packed |= kSegLstmFlagCoop;
    }
    if (config.seg_lstm_coop_warp) {
        packed |= kSegLstmFlagWarp;
    }
    if (config.seg_lstm_coop_warp_nosh) {
        packed |= kSegLstmFlagWarpNoSh;
    }
    if (config.seg_lstm_coop_bidir) {
        packed |= kSegLstmFlagBidir;
    }
    const int warps = std::clamp(config.seg_lstm_coop_warps, 1, 255);
    packed |= static_cast<uintptr_t>(warps) << kSegLstmWarpShift;
    return packed;
}

std::string basename_without_ext(const std::string & path) {
    std::string name = path;
    const size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    const size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) {
        name = name.substr(0, dot);
    }
    return name.empty() ? "audio" : name;
}

bool write_rttm_lines(const std::vector<RTTMSegment> & segments,
                      const std::string & uri,
                      const std::string & output_path) {
    FILE * fp = output_path.empty() ? stdout : std::fopen(output_path.c_str(), "w");
    if (!fp) {
        std::fprintf(stderr, "Error: cannot open RTTM output '%s'\n", output_path.c_str());
        return false;
    }

    for (const auto & seg : segments) {
        std::fprintf(fp,
                     "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n",
                     uri.c_str(),
                     seg.start,
                     seg.duration,
                     seg.speaker.c_str());
    }

    if (fp != stdout) {
        std::fclose(fp);
    }
    return true;
}

bool load_wav_file(const std::string & path, WavData & wav) {
    struct Header {
        char riff[4];
        uint32_t file_size;
        char wave[4];
    };
    struct ChunkHeader {
        char id[4];
        uint32_t size;
    };
    struct FmtChunk {
        uint16_t audio_format;
        uint16_t num_channels;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample;
    };

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::fprintf(stderr, "Error: failed to open WAV '%s'\n", path.c_str());
        return false;
    }

    Header header{};
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in || std::strncmp(header.riff, "RIFF", 4) != 0 || std::strncmp(header.wave, "WAVE", 4) != 0) {
        std::fprintf(stderr, "Error: invalid WAV container '%s'\n", path.c_str());
        return false;
    }

    FmtChunk fmt{};
    bool have_fmt = false;
    std::vector<int16_t> pcm;

    while (in) {
        ChunkHeader chunk{};
        in.read(reinterpret_cast<char *>(&chunk), sizeof(chunk));
        if (!in) {
            break;
        }

        if (std::strncmp(chunk.id, "fmt ", 4) == 0) {
            if (chunk.size < sizeof(FmtChunk)) {
                std::fprintf(stderr, "Error: malformed fmt chunk in '%s'\n", path.c_str());
                return false;
            }
            in.read(reinterpret_cast<char *>(&fmt), sizeof(FmtChunk));
            if (chunk.size > sizeof(FmtChunk)) {
                in.seekg(chunk.size - sizeof(FmtChunk), std::ios::cur);
            }
            have_fmt = true;
        } else if (std::strncmp(chunk.id, "data", 4) == 0) {
            pcm.resize(chunk.size / sizeof(int16_t));
            in.read(reinterpret_cast<char *>(pcm.data()), static_cast<std::streamsize>(chunk.size));
        } else {
            in.seekg(chunk.size, std::ios::cur);
        }

        if ((chunk.size & 1U) != 0U) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (!have_fmt || pcm.empty()) {
        std::fprintf(stderr, "Error: missing fmt/data chunk in '%s'\n", path.c_str());
        return false;
    }
    if (fmt.audio_format != 1 || fmt.bits_per_sample != 16) {
        std::fprintf(stderr, "Error: only 16-bit PCM WAV is supported\n");
        return false;
    }
    if (fmt.sample_rate != static_cast<uint32_t>(kSampleRate)) {
        std::fprintf(stderr, "Error: expected %d Hz WAV, got %u Hz\n", kSampleRate, fmt.sample_rate);
        return false;
    }

    wav.sample_rate = fmt.sample_rate;
    wav.channels = fmt.num_channels;
    const size_t frames = pcm.size() / std::max<int>(1, fmt.num_channels);
    wav.mono.assign(frames, 0.0f);
    for (size_t i = 0; i < frames; ++i) {
        double sum = 0.0;
        for (int ch = 0; ch < fmt.num_channels; ++ch) {
            sum += static_cast<double>(pcm[i * fmt.num_channels + ch]);
        }
        wav.mono[i] = static_cast<float>(sum / (32768.0 * fmt.num_channels));
    }
    return true;
}

double hz_to_mel(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

std::vector<float> build_hamming_window(int frame_size) {
    std::vector<float> window(frame_size);
    for (int i = 0; i < frame_size; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(static_cast<float>(2.0 * M_PI * i) / static_cast<float>(frame_size - 1));
    }
    return window;
}

std::vector<float> build_mel_filterbank(int fft_size, int sample_rate, int mel_bins) {
    const int fft_bins = fft_size / 2 + 1;
    const double mel_lo = hz_to_mel(20.0);
    const double mel_hi = hz_to_mel(sample_rate * 0.5);

    std::vector<double> mel_points(mel_bins + 2);
    for (int i = 0; i < mel_bins + 2; ++i) {
        mel_points[i] = mel_lo + (mel_hi - mel_lo) * static_cast<double>(i) / static_cast<double>(mel_bins + 1);
    }

    std::vector<int> bins(mel_bins + 2);
    for (int i = 0; i < mel_bins + 2; ++i) {
        const double hz = mel_to_hz(mel_points[i]);
        bins[i] = static_cast<int>(std::floor((fft_size + 1) * hz / sample_rate));
        bins[i] = std::clamp(bins[i], 0, fft_bins - 1);
    }

    std::vector<float> fb(static_cast<size_t>(mel_bins) * fft_bins, 0.0f);
    for (int m = 1; m <= mel_bins; ++m) {
        const int left = bins[m - 1];
        const int center = bins[m];
        const int right = bins[m + 1];
        if (center <= left || right <= center) {
            continue;
        }
        for (int k = left; k < center; ++k) {
            fb[static_cast<size_t>(m - 1) * fft_bins + k] =
                static_cast<float>(k - left) / static_cast<float>(center - left);
        }
        for (int k = center; k < right; ++k) {
            fb[static_cast<size_t>(m - 1) * fft_bins + k] =
                static_cast<float>(right - k) / static_cast<float>(right - center);
        }
    }
    return fb;
}

FbankData compute_fbank(const float * audio, int n_samples, bool apply_cmn_to_output = true) {
    FbankData out;
    if (!audio || n_samples <= 0) {
        return out;
    }

    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = static_cast<float>(kSampleRate);
    opts.frame_opts.frame_length_ms = 25.0f;
    opts.frame_opts.frame_shift_ms = 10.0f;
    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.remove_dc_offset = false;
    opts.frame_opts.preemph_coeff = 0.97f;
    opts.frame_opts.window_type = "hamming";
    opts.frame_opts.round_to_power_of_two = true;
    opts.frame_opts.snip_edges = true;
    opts.mel_opts.num_bins = kMelBins;
    opts.mel_opts.low_freq = 20.0f;
    opts.mel_opts.high_freq = 0.0f;
    opts.use_log_fbank = true;
    opts.use_power = true;

    knf::OnlineFbank fbank(opts);
    std::vector<float> scaled(static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        scaled[static_cast<size_t>(i)] = audio[i] * 32768.0f;
    }
    fbank.AcceptWaveform(static_cast<float>(kSampleRate), scaled.data(), n_samples);
    fbank.InputFinished();

    out.frames = fbank.NumFramesReady();
    out.data.assign(static_cast<size_t>(out.frames) * kMelBins, 0.0f);
    for (int t = 0; t < out.frames; ++t) {
        const float * frame = fbank.GetFrame(t);
        std::memcpy(out.data.data() + static_cast<size_t>(t) * kMelBins,
                    frame,
                    static_cast<size_t>(kMelBins) * sizeof(float));
    }

    if (apply_cmn_to_output) {
        apply_cmn(out.data.data(), out.frames);
    }

    return out;
}

void apply_cmn(float * data, int frames) {
    if (frames <= 0) {
        return;
    }
    for (int b = 0; b < kMelBins; ++b) {
        double mean = 0.0;
        for (int t = 0; t < frames; ++t) {
            mean += data[static_cast<size_t>(t) * kMelBins + b];
        }
        mean /= frames;
        for (int t = 0; t < frames; ++t) {
            data[static_cast<size_t>(t) * kMelBins + b] -= static_cast<float>(mean);
        }
    }
}

ggml_tensor * get_tensor(ggml_context * ctx, const char * name, bool required = true) {
    ggml_tensor * tensor = ggml_get_tensor(ctx, name);
    if (!tensor && required) {
        std::fprintf(stderr, "Error: missing tensor '%s'\n", name);
    }
    return tensor;
}

bool load_weight_store(const std::string & path, WeightStore & store) {
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx =*/ &store.ctx,
    };
    store.gguf = gguf_init_from_file(path.c_str(), params);
    if (!store.gguf || !store.ctx) {
        std::fprintf(stderr, "Error: failed to open GGUF '%s'\n", path.c_str());
        return false;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "Error: failed to initialize CPU backend for weights\n");
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(store.ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "Error: failed to allocate weight buffer\n");
        ggml_backend_free(backend);
        return false;
    }

    store.backends.push_back(backend);
    store.buffers.push_back(buffer);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "Error: failed to reopen GGUF '%s'\n", path.c_str());
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(store.gguf);
    std::vector<uint8_t> tmp;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(store.gguf, i);
        ggml_tensor * tensor = ggml_get_tensor(store.ctx, name);
        if (!tensor) {
            continue;
        }
        const size_t offset = gguf_get_data_offset(store.gguf) + gguf_get_tensor_offset(store.gguf, i);
        const size_t nbytes = ggml_nbytes(tensor);
        tmp.resize(nbytes);
        std::fseek(fp, static_cast<long>(offset), SEEK_SET);
        if (std::fread(tmp.data(), 1, nbytes, fp) != nbytes) {
            std::fprintf(stderr, "Error: failed to read tensor '%s'\n", name);
            std::fclose(fp);
            return false;
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, nbytes);
    }

    std::fclose(fp);
    return true;
}

bool load_weight_store(const std::string & path,
                       WeightStore & store,
                       const std::string & backend_name,
                       int gpu_device) {
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx =*/ &store.ctx,
    };
    store.gguf = gguf_init_from_file(path.c_str(), params);
    if (!store.gguf || !store.ctx) {
        std::fprintf(stderr, "Error: failed to open GGUF '%s'\n", path.c_str());
        return false;
    }

    const bool prefer_gpu = backend_name == "cuda" || backend_name == "auto";
    const bool allow_cpu_fallback = backend_name != "cuda";
    ggml_backend_t backend = nullptr;

    if (prefer_gpu) {
        char device_desc[32];
        std::snprintf(device_desc, sizeof(device_desc), "%d", gpu_device);
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, device_desc);
        if (!backend && !allow_cpu_fallback) {
            std::fprintf(stderr, "Error: failed to initialize GPU weight backend for '%s'\n", path.c_str());
            return false;
        }
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!backend) {
        std::fprintf(stderr, "Error: failed to initialize weight backend for '%s'\n", path.c_str());
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(store.ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "Error: failed to allocate weight buffer\n");
        ggml_backend_free(backend);
        return false;
    }

    store.backends.push_back(backend);
    store.buffers.push_back(buffer);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "Error: failed to reopen GGUF '%s'\n", path.c_str());
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(store.gguf);
    std::vector<uint8_t> tmp;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(store.gguf, i);
        ggml_tensor * tensor = ggml_get_tensor(store.ctx, name);
        if (!tensor) {
            continue;
        }
        const size_t offset = gguf_get_data_offset(store.gguf) + gguf_get_tensor_offset(store.gguf, i);
        const size_t nbytes = ggml_nbytes(tensor);
        tmp.resize(nbytes);
        std::fseek(fp, static_cast<long>(offset), SEEK_SET);
        if (std::fread(tmp.data(), 1, nbytes, fp) != nbytes) {
            std::fprintf(stderr, "Error: failed to read tensor '%s'\n", name);
            std::fclose(fp);
            return false;
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, nbytes);
    }

    std::fclose(fp);
    return true;
}

void free_weight_store(WeightStore & store) {
    for (ggml_backend_buffer_t buffer : store.buffers) {
        ggml_backend_buffer_free(buffer);
    }
    store.buffers.clear();

    for (ggml_backend_t backend : store.backends) {
        ggml_backend_free(backend);
    }
    store.backends.clear();

    if (store.gguf) {
        gguf_free(store.gguf);
        store.gguf = nullptr;
    }
    if (store.ctx) {
        ggml_free(store.ctx);
        store.ctx = nullptr;
    }
}

bool load_segmentation_model(const std::string & path,
                             SegmentationModel & model,
                             const std::string & backend_name,
                             int gpu_device) {
    if (!load_weight_store(path, model.weights, backend_name, gpu_device)) {
        return false;
    }

    ggml_context * ctx = model.weights.ctx;
    model.wav_norm_weight = get_tensor(ctx, "sincnet.wav_norm.weight", false);
    model.wav_norm_bias = get_tensor(ctx, "sincnet.wav_norm.bias", false);
    model.sinc_conv_weight[0] = get_tensor(ctx, "sincnet.0.conv.weight");
    model.sinc_conv_weight[1] = get_tensor(ctx, "sincnet.1.conv.weight");
    model.sinc_conv_weight[2] = get_tensor(ctx, "sincnet.2.conv.weight");
    model.sinc_conv_bias[1] = get_tensor(ctx, "sincnet.1.conv.bias");
    model.sinc_conv_bias[2] = get_tensor(ctx, "sincnet.2.conv.bias");

    for (int i = 0; i < 3; ++i) {
        char name[64];
        std::snprintf(name, sizeof(name), "sincnet.%d.norm.weight", i);
        model.sinc_norm_weight[i] = get_tensor(ctx, name);
        std::snprintf(name, sizeof(name), "sincnet.%d.norm.bias", i);
        model.sinc_norm_bias[i] = get_tensor(ctx, name);
    }

    for (int i = 0; i < 4; ++i) {
        char name[64];
        std::snprintf(name, sizeof(name), "lstm.weight_ih_l%d", i);
        model.lstm_weight_ih[i] = get_tensor(ctx, name);
        std::snprintf(name, sizeof(name), "lstm.weight_hh_l%d", i);
        model.lstm_weight_hh[i] = get_tensor(ctx, name);
        std::snprintf(name, sizeof(name), "lstm.bias_ih_l%d", i);
        model.lstm_bias_ih[i] = get_tensor(ctx, name);
        std::snprintf(name, sizeof(name), "lstm.bias_hh_l%d", i);
        model.lstm_bias_hh[i] = get_tensor(ctx, name);

        std::snprintf(name, sizeof(name), "lstm.weight_ih_l%d_reverse", i);
        model.lstm_weight_ih_rev[i] = get_tensor(ctx, name);
        std::snprintf(name, sizeof(name), "lstm.weight_hh_l%d_reverse", i);
        model.lstm_weight_hh_rev[i] = get_tensor(ctx, name);
        std::snprintf(name, sizeof(name), "lstm.bias_ih_l%d_reverse", i);
        model.lstm_bias_ih_rev[i] = get_tensor(ctx, name);
        std::snprintf(name, sizeof(name), "lstm.bias_hh_l%d_reverse", i);
        model.lstm_bias_hh_rev[i] = get_tensor(ctx, name);
    }

    model.linear_weight[0] = get_tensor(ctx, "linear.0.weight");
    model.linear_weight[1] = get_tensor(ctx, "linear.1.weight");
    model.linear_bias[0] = get_tensor(ctx, "linear.0.bias");
    model.linear_bias[1] = get_tensor(ctx, "linear.1.bias");
    model.classifier_weight = get_tensor(ctx, "classifier.weight");
    model.classifier_bias = get_tensor(ctx, "classifier.bias");

    return true;
}

bool load_embedding_model(const std::string & path,
                          EmbeddingModel & model,
                          const std::string & backend_name,
                          int gpu_device) {
    if (!load_weight_store(path, model.weights, backend_name, gpu_device)) {
        return false;
    }

    ggml_context * ctx = model.weights.ctx;
    model.conv1_weight = get_tensor(ctx, "resnet.conv1.weight");
    model.bn1_weight = get_tensor(ctx, "resnet.bn1.weight");
    model.bn1_bias = get_tensor(ctx, "resnet.bn1.bias");
    model.bn1_mean = get_tensor(ctx, "resnet.bn1.running_mean");
    model.bn1_var = get_tensor(ctx, "resnet.bn1.running_var");

    const int blocks[4] = {3, 4, 6, 3};
    for (int l = 0; l < 4; ++l) {
        for (int b = 0; b < blocks[l]; ++b) {
            char name[128];
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.conv1.weight", l + 1, b);
            model.layer_conv1_weight[l][b] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.bn1.weight", l + 1, b);
            model.layer_bn1_weight[l][b] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.bn1.bias", l + 1, b);
            model.layer_bn1_bias[l][b] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.bn1.running_mean", l + 1, b);
            model.layer_bn1_mean[l][b] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.bn1.running_var", l + 1, b);
            model.layer_bn1_var[l][b] = get_tensor(ctx, name);

            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.conv2.weight", l + 1, b);
            model.layer_conv2_weight[l][b] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.bn2.weight", l + 1, b);
            model.layer_bn2_weight[l][b] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.bn2.bias", l + 1, b);
            model.layer_bn2_bias[l][b] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.bn2.running_mean", l + 1, b);
            model.layer_bn2_mean[l][b] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.%d.bn2.running_var", l + 1, b);
            model.layer_bn2_var[l][b] = get_tensor(ctx, name);
        }

        if (l >= 1) {
            char name[128];
            std::snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.0.weight", l + 1);
            model.shortcut_conv_weight[l] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.1.weight", l + 1);
            model.shortcut_bn_weight[l] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.1.bias", l + 1);
            model.shortcut_bn_bias[l] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.1.running_mean", l + 1);
            model.shortcut_bn_mean[l] = get_tensor(ctx, name);
            std::snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.1.running_var", l + 1);
            model.shortcut_bn_var[l] = get_tensor(ctx, name);
        }
    }

    model.seg1_weight = get_tensor(ctx, "resnet.seg_1.weight");
    model.seg1_bias = get_tensor(ctx, "resnet.seg_1.bias");
    return true;
}

bool init_backend_state(BackendState & state,
                        const std::string & backend_name,
                        int gpu_device,
                        size_t graph_nodes) {
    const bool want_cpu = backend_name == "cpu" || backend_name == "auto";
    const bool want_cuda = backend_name == "cuda" || backend_name == "auto";
    const bool want_metal = backend_name == "metal" || backend_name == "auto";

    ggml_backend_t cpu_main = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_main) {
        std::fprintf(stderr, "Error: failed to initialize CPU backend\n");
        return false;
    }
    state.backends.push_back(cpu_main);

    if (want_cuda) {
        char device_desc[32];
        std::snprintf(device_desc, sizeof(device_desc), "CUDA%d", gpu_device);
        ggml_backend_t cuda_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, device_desc);
        if (cuda_backend) {
            state.backends.push_back(cuda_backend);
        } else if (backend_name == "cuda") {
            std::fprintf(stderr, "Error: CUDA backend requested but unavailable\n");
            return false;
        }
    }

    if (want_metal) {
        ggml_backend_t metal_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, "Metal");
        if (metal_backend) {
            state.backends.push_back(metal_backend);
        } else if (backend_name == "metal") {
            std::fprintf(stderr, "Error: Metal backend requested but unavailable\n");
            return false;
        }
    }

    if (state.backends.size() > 1 || want_cpu) {
        ggml_backend_t cpu_fallback = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cpu_fallback) {
            state.backends.push_back(cpu_fallback);
        }
    }

    for (ggml_backend_t backend : state.backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            state.preferred_gpu = backend;
            break;
        }
    }

    if (!state.preferred_gpu && backend_name != "auto" && backend_name != "cpu") {
        std::fprintf(stderr, "Error: requested backend '%s' unavailable\n", backend_name.c_str());
        return false;
    }

    if (state.preferred_gpu && state.backends.front() != state.preferred_gpu) {
        const auto it = std::find(state.backends.begin(), state.backends.end(), state.preferred_gpu);
        if (it != state.backends.end()) {
            std::rotate(state.backends.begin(), it, it + 1);
        }
    }

    const size_t meta_size = ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false);
    state.graph_meta.resize(meta_size);
    state.sched = ggml_backend_sched_new(state.backends.data(), nullptr, static_cast<int>(state.backends.size()), graph_nodes, false, true);
    if (!state.sched) {
        std::fprintf(stderr, "Error: failed to create backend scheduler\n");
        return false;
    }

    return true;
}

void free_backend_state(BackendState & state) {
    if (state.sched) {
        ggml_backend_sched_free(state.sched);
        state.sched = nullptr;
    }
    for (ggml_backend_t backend : state.backends) {
        ggml_backend_free(backend);
    }
    state.backends.clear();
    state.preferred_gpu = nullptr;
    state.graph_meta.clear();
}

void prefer_gpu_nodes(BackendState & state, ggml_cgraph * graph) {
    if (!state.sched || !state.preferred_gpu || !graph) {
        return;
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node && ggml_backend_supports_op(state.preferred_gpu, node)) {
            ggml_backend_sched_set_tensor_backend(state.sched, node, state.preferred_gpu);
        }
    }
}

ggml_tensor * instance_norm_1d(ggml_context * ctx,
                               ggml_tensor * x,
                               ggml_tensor * weight,
                               ggml_tensor * bias) {
    ggml_tensor * y = ggml_norm(ctx, x, kInstanceNormEps);
    if (weight) {
        ggml_tensor * w = ggml_reshape_3d(ctx, weight, 1, weight->ne[0], 1);
        y = ggml_mul(ctx, y, w);
    }
    if (bias) {
        ggml_tensor * b = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
        y = ggml_add(ctx, y, b);
    }
    return y;
}

ggml_tensor * segmentation_conv_stage(ggml_context * ctx,
                                      ggml_tensor * input,
                                      ggml_tensor * weight,
                                      ggml_tensor * bias,
                                      ggml_tensor * norm_weight,
                                      ggml_tensor * norm_bias,
                                      int stride,
                                      bool apply_abs) {
    ggml_tensor * y = ggml_conv_1d(ctx, weight, input, stride, 0, 1);
    if (bias) {
        ggml_tensor * b = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
        y = ggml_add(ctx, y, b);
    }
    if (apply_abs) {
        y = ggml_abs(ctx, y);
    }
    y = ggml_pool_1d(ctx, y, GGML_OP_POOL_MAX, 3, 3, 0);
    y = instance_norm_1d(ctx, y, norm_weight, norm_bias);
    y = ggml_leaky_relu(ctx, y, kLeakyReluSlope, false);
    return y;
}

void cpu_lstm_direction(float * dst,
                        int dst_offset,
                        const float * input,
                        int seq_len,
                        int input_size,
                        int hidden,
                        bool reverse,
                        ggml_tensor * weight_ih,
                        ggml_tensor * weight_hh,
                        ggml_tensor * bias_ih,
                        ggml_tensor * bias_hh) {
    const int gates = 4 * hidden;
    std::vector<float> w_ih(static_cast<size_t>(gates) * input_size);
    std::vector<float> w_hh(static_cast<size_t>(gates) * hidden);

    const ggml_fp16_t * src_ih = reinterpret_cast<const ggml_fp16_t *>(weight_ih->data);
    const ggml_fp16_t * src_hh = reinterpret_cast<const ggml_fp16_t *>(weight_hh->data);
    for (size_t i = 0; i < w_ih.size(); ++i) {
        w_ih[i] = ggml_fp16_to_fp32(src_ih[i]);
    }
    for (size_t i = 0; i < w_hh.size(); ++i) {
        w_hh[i] = ggml_fp16_to_fp32(src_hh[i]);
    }

    std::vector<float> ih_all(static_cast<size_t>(seq_len) * gates, 0.0f);
    for (int g = 0; g < gates; ++g) {
        const float * row = w_ih.data() + static_cast<size_t>(g) * input_size;
        for (int t = 0; t < seq_len; ++t) {
            const float * x = input + static_cast<size_t>(t);
            double acc = 0.0;
            for (int i = 0; i < input_size; ++i) {
                acc += static_cast<double>(row[i]) * static_cast<double>(x[static_cast<size_t>(i) * seq_len]);
            }
            ih_all[static_cast<size_t>(t) * gates + g] = static_cast<float>(acc);
        }
    }

    std::vector<float> h(hidden, 0.0f);
    std::vector<float> c(hidden, 0.0f);
    std::vector<float> gate_values(gates, 0.0f);

    const bool bias_f16 = bias_ih->type == GGML_TYPE_F16;
    const ggml_fp16_t * b_ih_f16 = bias_f16 ? reinterpret_cast<const ggml_fp16_t *>(bias_ih->data) : nullptr;
    const ggml_fp16_t * b_hh_f16 = bias_f16 ? reinterpret_cast<const ggml_fp16_t *>(bias_hh->data) : nullptr;
    const float * b_ih_f32 = !bias_f16 ? reinterpret_cast<const float *>(bias_ih->data) : nullptr;
    const float * b_hh_f32 = !bias_f16 ? reinterpret_cast<const float *>(bias_hh->data) : nullptr;

    for (int step = 0; step < seq_len; ++step) {
        const int t = reverse ? (seq_len - 1 - step) : step;
        for (int g = 0; g < gates; ++g) {
            const float * row = w_hh.data() + static_cast<size_t>(g) * hidden;
            double acc = ih_all[static_cast<size_t>(t) * gates + g];
            for (int i = 0; i < hidden; ++i) {
                acc += static_cast<double>(row[i]) * static_cast<double>(h[i]);
            }
            const float b1 = bias_f16 ? ggml_fp16_to_fp32(b_ih_f16[g]) : b_ih_f32[g];
            const float b2 = bias_f16 ? ggml_fp16_to_fp32(b_hh_f16[g]) : b_hh_f32[g];
            gate_values[g] = static_cast<float>(acc) + b1 + b2;
        }
        for (int i = 0; i < hidden; ++i) {
            const float in_gate = 1.0f / (1.0f + std::exp(-gate_values[i]));
            const float forget_gate = 1.0f / (1.0f + std::exp(-gate_values[hidden + i]));
            const float cand = std::tanh(gate_values[2 * hidden + i]);
            const float out_gate = 1.0f / (1.0f + std::exp(-gate_values[3 * hidden + i]));
            c[i] = forget_gate * c[i] + in_gate * cand;
            h[i] = out_gate * std::tanh(c[i]);
            dst[static_cast<size_t>(dst_offset + i) * seq_len + t] = h[i];
        }
    }
}

void pyannote_seg_bilstm_op(ggml_tensor * dst, int ith, int nth, void *) {
    if (ith != 0) {
        return;
    }

    ggml_tensor * input = dst->src[0];
    ggml_tensor * w_ih = dst->src[1];
    ggml_tensor * w_hh = dst->src[2];
    ggml_tensor * b_ih = dst->src[3];
    ggml_tensor * b_hh = dst->src[4];
    ggml_tensor * w_ih_r = dst->src[5];
    ggml_tensor * w_hh_r = dst->src[6];
    ggml_tensor * b_ih_r = dst->src[7];
    ggml_tensor * b_hh_r = dst->src[8];
    (void) nth;

    const int seq_len = static_cast<int>(input->ne[0]);
    const int input_size = static_cast<int>(input->ne[1]);
    const int hidden = static_cast<int>(w_hh->ne[0]);
    float * out = reinterpret_cast<float *>(dst->data);
    const float * x = reinterpret_cast<const float *>(input->data);

    cpu_lstm_direction(out, 0, x, seq_len, input_size, hidden, false, w_ih, w_hh, b_ih, b_hh);
    cpu_lstm_direction(out, hidden, x, seq_len, input_size, hidden, true, w_ih_r, w_hh_r, b_ih_r, b_hh_r);
}

ggml_tensor * bidirectional_lstm_layer(ggml_context * ctx,
                                       ggml_tensor * input,
                                       ggml_tensor * w_ih,
                                       ggml_tensor * w_hh,
                                       ggml_tensor * b_ih,
                                       ggml_tensor * b_hh,
                                       ggml_tensor * w_ih_r,
                                       ggml_tensor * w_hh_r,
                                       ggml_tensor * b_ih_r,
                                       ggml_tensor * b_hh_r,
                                       uintptr_t cuda_options) {
    ggml_tensor * srcs[] = {input, w_ih, w_hh, b_ih, b_hh, w_ih_r, w_hh_r, b_ih_r, b_hh_r};
    const int seq_len = static_cast<int>(input->ne[0]);
    const int hidden = static_cast<int>(w_hh->ne[0]);
    ggml_tensor * out = ggml_custom_4d(ctx,
                                       GGML_TYPE_F32,
                                       seq_len,
                                       2 * hidden,
                                       1,
                                       1,
                                       srcs,
                                       9,
                                       pyannote_seg_bilstm_op,
                                       2,
                                       reinterpret_cast<void *>(cuda_options));
    ggml_set_name(out, "pyannote_seg_bilstm");
    return out;
}

ggml_tensor * linear_layer(ggml_context * ctx,
                           ggml_tensor * x,
                           ggml_tensor * weight,
                           ggml_tensor * bias,
                           bool apply_activation,
                           const char * name) {
    const int64_t seq_len = x->ne[0];
    const int64_t input_dim = x->ne[1];
    const int64_t output_dim = weight->ne[1];

    ggml_tensor * x_t = ggml_permute(ctx, x, 1, 0, 2, 3);
    x_t = ggml_cont(ctx, x_t);
    ggml_tensor * x_2d = ggml_reshape_2d(ctx, x_t, input_dim, seq_len);
    x_2d = ggml_cont(ctx, x_2d);

    ggml_tensor * y = ggml_mul_mat(ctx, weight, x_2d);
    if (name) {
        ggml_set_name(y, name);
    }
    y = ggml_add(ctx, y, bias);
    if (apply_activation) {
        y = ggml_leaky_relu(ctx, y, kLeakyReluSlope, true);
    }
    y = ggml_transpose(ctx, y);
    y = ggml_cont(ctx, y);
    return ggml_reshape_3d(ctx, y, seq_len, output_dim, 1);
}

ggml_tensor * classifier_layer(ggml_context * ctx,
                               ggml_tensor * x,
                               ggml_tensor * weight,
                               ggml_tensor * bias) {
    const int64_t seq_len = x->ne[0];
    const int64_t input_dim = x->ne[1];
    const int64_t num_classes = weight->ne[1];

    ggml_tensor * x_t = ggml_permute(ctx, x, 1, 0, 2, 3);
    x_t = ggml_cont(ctx, x_t);
    ggml_tensor * x_2d = ggml_reshape_2d(ctx, x_t, input_dim, seq_len);
    x_2d = ggml_cont(ctx, x_2d);

    ggml_tensor * logits = ggml_mul_mat(ctx, weight, x_2d);
    ggml_set_name(logits, "classifier_mm");
    logits = ggml_add(ctx, logits, bias);
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    ggml_tensor * log_probs = ggml_log(ctx, probs);
    ggml_tensor * out = ggml_transpose(ctx, log_probs);
    out = ggml_cont(ctx, out);
    out = ggml_reshape_3d(ctx, out, seq_len, num_classes, 1);
    ggml_set_name(out, "classifier_out");
    return out;
}

ggml_tensor * build_segmentation_forward(ggml_context * ctx,
                                         const SegmentationModel & model,
                                         ggml_tensor * waveform,
                                         uintptr_t seg_lstm_cuda_options) {
    ggml_tensor * x = waveform;
    if (model.wav_norm_weight && model.wav_norm_bias) {
        x = instance_norm_1d(ctx, x, model.wav_norm_weight, model.wav_norm_bias);
    }

    x = segmentation_conv_stage(ctx, x, model.sinc_conv_weight[0], model.sinc_conv_bias[0], model.sinc_norm_weight[0], model.sinc_norm_bias[0], 10, true);
    x = segmentation_conv_stage(ctx, x, model.sinc_conv_weight[1], model.sinc_conv_bias[1], model.sinc_norm_weight[1], model.sinc_norm_bias[1], 1, false);
    x = segmentation_conv_stage(ctx, x, model.sinc_conv_weight[2], model.sinc_conv_bias[2], model.sinc_norm_weight[2], model.sinc_norm_bias[2], 1, false);

    for (int i = 0; i < 4; ++i) {
        x = bidirectional_lstm_layer(ctx,
                                     x,
                                     model.lstm_weight_ih[i],
                                     model.lstm_weight_hh[i],
                                     model.lstm_bias_ih[i],
                                     model.lstm_bias_hh[i],
                                     model.lstm_weight_ih_rev[i],
                                     model.lstm_weight_hh_rev[i],
                                     model.lstm_bias_ih_rev[i],
                                     model.lstm_bias_hh_rev[i],
                                     seg_lstm_cuda_options);
    }

    x = ggml_cont(ctx, x);
    ggml_set_name(x, "lstm_out_cont");
    x = linear_layer(ctx, x, model.linear_weight[0], model.linear_bias[0], true, "linear1_mm");
    x = linear_layer(ctx, x, model.linear_weight[1], model.linear_bias[1], true, "linear2_mm");
    return classifier_layer(ctx, x, model.classifier_weight, model.classifier_bias);
}

ggml_cgraph * build_segmentation_graph(const SegmentationModel & model,
                                       BackendState & state,
                                       uintptr_t seg_lstm_cuda_options) {
    ggml_init_params params = {
        /*.mem_size   =*/ state.graph_meta.size(),
        /*.mem_buffer =*/ state.graph_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return nullptr;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kSegGraphNodes, false);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kChunkSamples, 1, 1);
    ggml_set_name(input, "waveform");
    ggml_set_input(input);
    ggml_tensor * output = build_segmentation_forward(ctx, model, input, seg_lstm_cuda_options);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_free(ctx);
    return graph;
}

bool infer_segmentation(const SegmentationModel & model,
                        BackendState & state,
                        const float * chunk,
                        uintptr_t seg_lstm_cuda_options,
                        float * out_logits) {
    ggml_cgraph * graph = build_segmentation_graph(model, state, seg_lstm_cuda_options);
    if (!graph) {
        std::fprintf(stderr, "Error: failed to build segmentation graph\n");
        return false;
    }
    prefer_gpu_nodes(state, graph);
    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        std::fprintf(stderr, "Error: failed to allocate segmentation graph\n");
        return false;
    }

    ggml_tensor * input = ggml_graph_get_tensor(graph, "waveform");
    ggml_tensor * output = ggml_graph_get_tensor(graph, "classifier_out");
    if (!input || !output) {
        std::fprintf(stderr, "Error: missing segmentation IO tensors\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    ggml_backend_tensor_set(input, chunk, 0, static_cast<size_t>(kChunkSamples) * sizeof(float));
    if (ggml_backend_sched_graph_compute(state.sched, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "Error: segmentation graph compute failed\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    std::vector<float> class_major(static_cast<size_t>(kFramesPerChunk) * kPowersetClasses, 0.0f);
    ggml_backend_tensor_get(output, class_major.data(), 0, class_major.size() * sizeof(float));
    for (int cls = 0; cls < kPowersetClasses; ++cls) {
        for (int frame = 0; frame < kFramesPerChunk; ++frame) {
            out_logits[static_cast<size_t>(frame) * kPowersetClasses + cls] =
                class_major[static_cast<size_t>(cls) * kFramesPerChunk + frame];
        }
    }
    ggml_backend_sched_reset(state.sched);
    return true;
}

ggml_tensor * batch_norm_2d(ggml_context * ctx,
                            ggml_tensor * x,
                            ggml_tensor * weight,
                            ggml_tensor * bias,
                            ggml_tensor * mean,
                            ggml_tensor * var,
                            ggml_tensor * eps) {
    const int64_t channels = weight->ne[0];
    ggml_tensor * inv = ggml_sqrt(ctx, ggml_add1(ctx, var, eps));
    ggml_tensor * scale = ggml_div(ctx, weight, inv);
    ggml_tensor * shift = ggml_sub(ctx, bias, ggml_mul(ctx, mean, scale));
    ggml_tensor * scale_4d = ggml_reshape_4d(ctx, scale, 1, 1, channels, 1);
    ggml_tensor * shift_4d = ggml_reshape_4d(ctx, shift, 1, 1, channels, 1);
    return ggml_add(ctx, ggml_mul(ctx, x, scale_4d), shift_4d);
}

ggml_tensor * residual_block(ggml_context * ctx,
                             ggml_tensor * x,
                             ggml_tensor * conv1_w,
                             ggml_tensor * bn1_w,
                             ggml_tensor * bn1_b,
                             ggml_tensor * bn1_m,
                             ggml_tensor * bn1_v,
                             ggml_tensor * conv2_w,
                             ggml_tensor * bn2_w,
                             ggml_tensor * bn2_b,
                             ggml_tensor * bn2_m,
                             ggml_tensor * bn2_v,
                             ggml_tensor * shortcut_conv,
                             ggml_tensor * shortcut_bn_w,
                             ggml_tensor * shortcut_bn_b,
                             ggml_tensor * shortcut_bn_m,
                             ggml_tensor * shortcut_bn_v,
                             ggml_tensor * eps,
                             int stride) {
    ggml_tensor * identity = x;
    ggml_tensor * y = ggml_conv_2d(ctx, conv1_w, x, stride, stride, 1, 1, 1, 1);
    y = batch_norm_2d(ctx, y, bn1_w, bn1_b, bn1_m, bn1_v, eps);
    y = ggml_relu(ctx, y);
    y = ggml_conv_2d(ctx, conv2_w, y, 1, 1, 1, 1, 1, 1);
    y = batch_norm_2d(ctx, y, bn2_w, bn2_b, bn2_m, bn2_v, eps);

    if (shortcut_conv) {
        identity = ggml_conv_2d(ctx, shortcut_conv, x, stride, stride, 0, 0, 1, 1);
        identity = batch_norm_2d(ctx, identity, shortcut_bn_w, shortcut_bn_b, shortcut_bn_m, shortcut_bn_v, eps);
    }
    y = ggml_add(ctx, y, identity);
    y = ggml_relu(ctx, y);
    return y;
}

ggml_tensor * build_embedding_forward(ggml_context * ctx,
                                      const EmbeddingModel & model,
                                      ggml_tensor * input) {
    const int blocks[4] = {3, 4, 6, 3};
    ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(eps, "bn_eps");
    ggml_set_input(eps);

    ggml_tensor * x = ggml_conv_2d(ctx, model.conv1_weight, input, 1, 1, 1, 1, 1, 1);
    x = batch_norm_2d(ctx, x, model.bn1_weight, model.bn1_bias, model.bn1_mean, model.bn1_var, eps);
    x = ggml_relu(ctx, x);

    for (int l = 0; l < 4; ++l) {
        for (int b = 0; b < blocks[l]; ++b) {
            const int stride = (l > 0 && b == 0) ? 2 : 1;
            x = residual_block(ctx,
                               x,
                               model.layer_conv1_weight[l][b],
                               model.layer_bn1_weight[l][b],
                               model.layer_bn1_bias[l][b],
                               model.layer_bn1_mean[l][b],
                               model.layer_bn1_var[l][b],
                               model.layer_conv2_weight[l][b],
                               model.layer_bn2_weight[l][b],
                               model.layer_bn2_bias[l][b],
                               model.layer_bn2_mean[l][b],
                               model.layer_bn2_var[l][b],
                               (b == 0 && l >= 1) ? model.shortcut_conv_weight[l] : nullptr,
                               (b == 0 && l >= 1) ? model.shortcut_bn_weight[l] : nullptr,
                               (b == 0 && l >= 1) ? model.shortcut_bn_bias[l] : nullptr,
                               (b == 0 && l >= 1) ? model.shortcut_bn_mean[l] : nullptr,
                               (b == 0 && l >= 1) ? model.shortcut_bn_var[l] : nullptr,
                               eps,
                               stride);
        }
    }

    const int64_t t8 = x->ne[0];
    const int64_t feat_dim = x->ne[1] * x->ne[2];
    ggml_tensor * flat = ggml_reshape_4d(ctx, x, t8, feat_dim, 1, 1);
    ggml_tensor * flat_2d = ggml_reshape_2d(ctx, flat, t8, feat_dim);
    ggml_tensor * perm = ggml_permute(ctx, flat_2d, 1, 0, 2, 3);
    perm = ggml_cont(ctx, perm);

    ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t8, 1);
    ggml_set_name(ones, "tstp_ones");
    ggml_set_input(ones);
    ggml_tensor * t8_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t8_scalar, "tstp_t8");
    ggml_set_input(t8_scalar);
    ggml_tensor * t8m1_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t8m1_scalar, "tstp_t8m1");
    ggml_set_input(t8m1_scalar);
    ggml_tensor * pool_eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(pool_eps, "tstp_eps");
    ggml_set_input(pool_eps);

    ggml_tensor * perm_t = ggml_permute(ctx, perm, 1, 0, 2, 3);
    perm_t = ggml_cont(ctx, perm_t);
    ggml_tensor * sum = ggml_mul_mat(ctx, perm_t, ones);
    ggml_tensor * mean = ggml_div(ctx, sum, t8_scalar);

    ggml_tensor * sq = ggml_mul(ctx, perm, perm);
    ggml_tensor * sq_t = ggml_permute(ctx, sq, 1, 0, 2, 3);
    sq_t = ggml_cont(ctx, sq_t);
    ggml_tensor * mean_sq = ggml_div(ctx, ggml_mul_mat(ctx, sq_t, ones), t8_scalar);
    ggml_tensor * var = ggml_sub(ctx, mean_sq, ggml_mul(ctx, mean, mean));
    ggml_tensor * unbiased = ggml_mul(ctx, var, ggml_div(ctx, t8_scalar, t8m1_scalar));
    ggml_tensor * stddev = ggml_sqrt(ctx, ggml_add(ctx, ggml_clamp(ctx, unbiased, 0.0f, INFINITY), pool_eps));

    ggml_tensor * mean_1d = ggml_reshape_1d(ctx, mean, feat_dim);
    ggml_tensor * std_1d = ggml_reshape_1d(ctx, stddev, feat_dim);
    ggml_tensor * pooled = ggml_concat(ctx, mean_1d, std_1d, 0);
    ggml_tensor * pooled_2d = ggml_reshape_2d(ctx, pooled, 2 * feat_dim, 1);
    ggml_tensor * embed = ggml_mul_mat(ctx, model.seg1_weight, pooled_2d);
    embed = ggml_add(ctx, embed, model.seg1_bias);
    embed = ggml_reshape_1d(ctx, embed, kEmbeddingDim);
    ggml_set_name(embed, "embedding");
    return embed;
}

ggml_cgraph * build_embedding_graph(const EmbeddingModel & model,
                                    BackendState & state,
                                    int num_frames) {
    ggml_init_params params = {
        /*.mem_size   =*/ state.graph_meta.size(),
        /*.mem_buffer =*/ state.graph_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return nullptr;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kEmbGraphNodes, false);
    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, num_frames, kMelBins, 1, 1);
    ggml_set_name(input, "fbank");
    ggml_set_input(input);
    ggml_tensor * output = build_embedding_forward(ctx, model, input);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_free(ctx);
    return graph;
}

bool infer_embedding(const EmbeddingModel & model,
                     BackendState & state,
                     const float * fbank,
                     int num_frames,
                     float * out_embedding) {
    ggml_cgraph * graph = build_embedding_graph(model, state, num_frames);
    if (!graph) {
        std::fprintf(stderr, "Error: failed to build embedding graph\n");
        return false;
    }
    prefer_gpu_nodes(state, graph);
    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        std::fprintf(stderr, "Error: failed to allocate embedding graph\n");
        return false;
    }

    ggml_tensor * input = ggml_graph_get_tensor(graph, "fbank");
    ggml_tensor * output = ggml_graph_get_tensor(graph, "embedding");
    ggml_tensor * bn_eps = ggml_graph_get_tensor(graph, "bn_eps");
    ggml_tensor * tstp_ones = ggml_graph_get_tensor(graph, "tstp_ones");
    ggml_tensor * tstp_t8 = ggml_graph_get_tensor(graph, "tstp_t8");
    ggml_tensor * tstp_t8m1 = ggml_graph_get_tensor(graph, "tstp_t8m1");
    ggml_tensor * tstp_eps = ggml_graph_get_tensor(graph, "tstp_eps");

    if (!input || !output || !bn_eps || !tstp_ones || !tstp_t8 || !tstp_t8m1 || !tstp_eps) {
        std::fprintf(stderr, "Error: missing embedding IO tensors\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    const float bn_eps_val = kBatchNormEps;
    const float t8_val = static_cast<float>(tstp_ones->ne[0]);
    const float t8m1_val = std::max(1.0f, t8_val - 1.0f);
    const float pool_eps_val = 1e-5f;
    std::vector<float> ones(static_cast<size_t>(tstp_ones->ne[0]), 1.0f);
    std::vector<float> fbank_planar(static_cast<size_t>(num_frames) * kMelBins, 0.0f);

    for (int frame = 0; frame < num_frames; ++frame) {
        for (int bin = 0; bin < kMelBins; ++bin) {
            fbank_planar[static_cast<size_t>(bin) * num_frames + frame] =
                fbank[static_cast<size_t>(frame) * kMelBins + bin];
        }
    }

    ggml_backend_tensor_set(input, fbank_planar.data(), 0, fbank_planar.size() * sizeof(float));
    ggml_backend_tensor_set(bn_eps, &bn_eps_val, 0, sizeof(float));
    ggml_backend_tensor_set(tstp_ones, ones.data(), 0, ones.size() * sizeof(float));
    ggml_backend_tensor_set(tstp_t8, &t8_val, 0, sizeof(float));
    ggml_backend_tensor_set(tstp_t8m1, &t8m1_val, 0, sizeof(float));
    ggml_backend_tensor_set(tstp_eps, &pool_eps_val, 0, sizeof(float));

    if (ggml_backend_sched_graph_compute(state.sched, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "Error: embedding graph compute failed\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    ggml_backend_tensor_get(output, out_embedding, 0, static_cast<size_t>(kEmbeddingDim) * sizeof(float));
    ggml_backend_sched_reset(state.sched);
    return true;
}

void powerset_to_multilabel(const float * logits, float * out_multi) {
    static constexpr float mapping[7][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 1.0f},
    };
    for (int i = 0; i < kFramesPerChunk; ++i) {
        const float * frame = logits + static_cast<size_t>(i) * kPowersetClasses;
        int best = 0;
        for (int k = 1; k < kPowersetClasses; ++k) {
            if (frame[k] > frame[best]) {
                best = k;
            }
        }
        float * dst = out_multi + static_cast<size_t>(i) * kLocalSpeakers;
        std::memcpy(dst, mapping[best], sizeof(mapping[best]));
    }
}

int closest_frame(double t, const SlidingWindowParams & sw) {
    return static_cast<int>(std::lround((t - sw.start - 0.5 * sw.duration) / sw.step));
}

void aggregate_chunks(const float * chunks,
                      int num_chunks,
                      int num_frames,
                      int num_classes,
                      const SlidingWindowParams & chunk_window,
                      const SlidingWindowParams & frame_window,
                      std::vector<float> & output,
                      int & total_frames,
                      bool skip_average,
                      float missing) {
    const SlidingWindowParams frames = {chunk_window.start, frame_window.duration, frame_window.step};
    const double end_time = chunk_window.start + chunk_window.duration + (num_chunks - 1) * chunk_window.step + 0.5 * frames.duration;
    total_frames = closest_frame(end_time, frames) + 1;

    output.assign(static_cast<size_t>(total_frames) * num_classes, 0.0f);
    std::vector<float> counts(static_cast<size_t>(total_frames) * num_classes, 0.0f);
    std::vector<float> mask(static_cast<size_t>(total_frames) * num_classes, 0.0f);

    for (int c = 0; c < num_chunks; ++c) {
        const int start = closest_frame(chunk_window.start + c * chunk_window.step + 0.5 * frames.duration, frames);
        const float * chunk = chunks + static_cast<size_t>(c) * num_frames * num_classes;
        for (int f = 0; f < num_frames; ++f) {
            const int dst_frame = start + f;
            if (dst_frame < 0 || dst_frame >= total_frames) {
                continue;
            }
            for (int k = 0; k < num_classes; ++k) {
                const float value = chunk[static_cast<size_t>(f) * num_classes + k];
                const float valid = std::isnan(value) ? 0.0f : 1.0f;
                const float clean = std::isnan(value) ? 0.0f : value;
                output[static_cast<size_t>(dst_frame) * num_classes + k] += clean * valid;
                counts[static_cast<size_t>(dst_frame) * num_classes + k] += valid;
                mask[static_cast<size_t>(dst_frame) * num_classes + k] =
                    std::max(mask[static_cast<size_t>(dst_frame) * num_classes + k], valid);
            }
        }
    }

    if (!skip_average) {
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] /= std::max(counts[i], 1e-12f);
        }
    }
    for (size_t i = 0; i < output.size(); ++i) {
        if (mask[i] == 0.0f) {
            output[i] = missing;
        }
    }
}

void compute_speaker_count(const float * binarized,
                           int num_chunks,
                           const SlidingWindowParams & chunk_window,
                           const SlidingWindowParams & frame_window,
                           std::vector<int> & count,
                           int & total_frames) {
    std::vector<float> summed(static_cast<size_t>(num_chunks) * kFramesPerChunk, 0.0f);
    for (int c = 0; c < num_chunks; ++c) {
        for (int f = 0; f < kFramesPerChunk; ++f) {
            float sum = 0.0f;
            for (int s = 0; s < kLocalSpeakers; ++s) {
                sum += binarized[(static_cast<size_t>(c) * kFramesPerChunk + f) * kLocalSpeakers + s];
            }
            summed[static_cast<size_t>(c) * kFramesPerChunk + f] = sum;
        }
    }

    std::vector<float> aggregated;
    aggregate_chunks(summed.data(), num_chunks, kFramesPerChunk, 1, chunk_window, frame_window, aggregated, total_frames, false, 0.0f);
    count.resize(total_frames);
    for (int i = 0; i < total_frames; ++i) {
        count[i] = static_cast<int>(std::lround(aggregated[i]));
    }
}

double cosine_similarity(const float * a, const float * b, int dim) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (int i = 0; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 1e-30 || nb <= 1e-30) {
        return -1.0;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

void normalize_rows(std::vector<double> & data, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        double norm = 0.0;
        for (int d = 0; d < dim; ++d) {
            const double v = data[static_cast<size_t>(r) * dim + d];
            norm += v * v;
        }
        norm = std::sqrt(std::max(norm, 1e-30));
        for (int d = 0; d < dim; ++d) {
            data[static_cast<size_t>(r) * dim + d] /= norm;
        }
    }
}

bool load_f64_tensor(ggml_context * ctx,
                     gguf_context * gguf_ctx,
                     FILE * fp,
                     const char * name,
                     std::vector<double> & out,
                     size_t expected) {
    ggml_tensor * tensor = ggml_get_tensor(ctx, name);
    if (!tensor || static_cast<size_t>(ggml_nelements(tensor)) != expected) {
        return false;
    }
    const int idx = gguf_find_tensor(gguf_ctx, name);
    if (idx < 0) {
        return false;
    }
    out.resize(expected);
    const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, idx);
    std::fseek(fp, static_cast<long>(offset), SEEK_SET);
    return std::fread(out.data(), sizeof(double), expected, fp) == expected;
}

bool load_plda(const std::string & path, PldaModel & plda) {
    ggml_context * ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx =*/ &ctx,
    };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx || !ctx) {
        return false;
    }
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        gguf_free(gguf_ctx);
        ggml_free(ctx);
        return false;
    }

    const bool ok =
        load_f64_tensor(ctx, gguf_ctx, fp, "plda.mean1", plda.mean1, kEmbeddingDim) &&
        load_f64_tensor(ctx, gguf_ctx, fp, "plda.mean2", plda.mean2, kPldaDim) &&
        load_f64_tensor(ctx, gguf_ctx, fp, "plda.lda", plda.lda, static_cast<size_t>(kEmbeddingDim) * kPldaDim) &&
        load_f64_tensor(ctx, gguf_ctx, fp, "plda.mu", plda.mu, kPldaDim) &&
        load_f64_tensor(ctx, gguf_ctx, fp, "plda.tr", plda.tr, static_cast<size_t>(kPldaDim) * kPldaDim) &&
        load_f64_tensor(ctx, gguf_ctx, fp, "plda.psi", plda.psi, kPldaDim);

    std::fclose(fp);
    gguf_free(gguf_ctx);
    ggml_free(ctx);
    plda.loaded = ok;
    return ok;
}

void plda_transform(const PldaModel & plda,
                    const std::vector<float> & src,
                    int rows,
                    std::vector<double> & out) {
    std::vector<double> centered(static_cast<size_t>(rows) * kEmbeddingDim, 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < kEmbeddingDim; ++d) {
            centered[static_cast<size_t>(r) * kEmbeddingDim + d] =
                static_cast<double>(src[static_cast<size_t>(r) * kEmbeddingDim + d]) - plda.mean1[d];
        }
    }
    normalize_rows(centered, rows, kEmbeddingDim);
    const double scale1 = std::sqrt(static_cast<double>(kEmbeddingDim));
    for (double & v : centered) {
        v *= scale1;
    }

    out.assign(static_cast<size_t>(rows) * kPldaDim, 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < kPldaDim; ++j) {
            double acc = 0.0;
            for (int i = 0; i < kEmbeddingDim; ++i) {
                acc += centered[static_cast<size_t>(r) * kEmbeddingDim + i] *
                       plda.lda[static_cast<size_t>(i) * kPldaDim + j];
            }
            out[static_cast<size_t>(r) * kPldaDim + j] = acc - plda.mean2[j];
        }
    }
    normalize_rows(out, rows, kPldaDim);
    const double scale2 = std::sqrt(static_cast<double>(kPldaDim));
    for (double & v : out) {
        v *= scale2;
    }

    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < kPldaDim; ++j) {
            out[static_cast<size_t>(r) * kPldaDim + j] -= plda.mu[j];
        }
    }

    std::vector<double> transformed(static_cast<size_t>(rows) * kPldaDim, 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < kPldaDim; ++j) {
            double acc = 0.0;
            for (int i = 0; i < kPldaDim; ++i) {
                acc += out[static_cast<size_t>(r) * kPldaDim + i] *
                       plda.tr[static_cast<size_t>(j) * kPldaDim + i];
            }
            transformed[static_cast<size_t>(r) * kPldaDim + j] = acc;
        }
    }
    out.swap(transformed);
}

vbx_model to_vbx_model(const PldaModel & plda) {
    vbx_model model{};
    model.D_emb = kEmbeddingDim;
    model.D_lda = kPldaDim;
    model.mean1.assign(plda.mean1.begin(), plda.mean1.end());
    model.mean2.assign(plda.mean2.begin(), plda.mean2.end());
    model.lda.assign(plda.lda.begin(), plda.lda.end());
    model.plda_mu.assign(plda.mu.begin(), plda.mu.end());
    model.plda_tr.assign(plda.tr.begin(), plda.tr.end());
    model.plda_psi.assign(plda.psi.begin(), plda.psi.end());
    return model;
}

void filter_embeddings(const std::vector<float> & embeddings,
                       const std::vector<float> & binarized,
                       int num_chunks,
                       std::vector<float> & filtered,
                       std::vector<int> & chunk_idx,
                       std::vector<int> & speaker_idx) {
    filtered.clear();
    chunk_idx.clear();
    speaker_idx.clear();
    const float active_threshold = 0.05f * kFramesPerChunk;

    for (int c = 0; c < num_chunks; ++c) {
        const float * seg_chunk = binarized.data() + static_cast<size_t>(c) * kFramesPerChunk * kLocalSpeakers;
        float clean[kLocalSpeakers] = {0.0f, 0.0f, 0.0f};
        for (int f = 0; f < kFramesPerChunk; ++f) {
            const float * frame = seg_chunk + static_cast<size_t>(f) * kLocalSpeakers;
            const float sum = frame[0] + frame[1] + frame[2];
            if (sum == 1.0f) {
                for (int s = 0; s < kLocalSpeakers; ++s) {
                    clean[s] += frame[s];
                }
            }
        }
        for (int s = 0; s < kLocalSpeakers; ++s) {
            const float * emb = embeddings.data() + (static_cast<size_t>(c) * kLocalSpeakers + s) * kEmbeddingDim;
            bool valid = clean[s] >= active_threshold;
            for (int d = 0; valid && d < kEmbeddingDim; ++d) {
                valid = std::isfinite(emb[d]);
            }
            if (!valid) {
                continue;
            }
            chunk_idx.push_back(c);
            speaker_idx.push_back(s);
            filtered.insert(filtered.end(), emb, emb + kEmbeddingDim);
        }
    }
}

std::vector<int> greedy_cluster(const std::vector<double> & features, int rows, int dim) {
    if (rows <= 0) {
        return {};
    }
    std::vector<int> labels(rows, -1);
    std::vector<double> centroids;
    std::vector<int> counts;

    for (int r = 0; r < rows; ++r) {
        const double * feat = features.data() + static_cast<size_t>(r) * dim;
        int best_cluster = -1;
        double best_score = -DBL_MAX;
        for (int c = 0; c < static_cast<int>(counts.size()); ++c) {
            const double * centroid = centroids.data() + static_cast<size_t>(c) * dim;
            double dot = 0.0;
            double nf = 0.0;
            double nc = 0.0;
            for (int d = 0; d < dim; ++d) {
                dot += feat[d] * centroid[d];
                nf += feat[d] * feat[d];
                nc += centroid[d] * centroid[d];
            }
            const double score = dot / (std::sqrt(std::max(nf, 1e-30)) * std::sqrt(std::max(nc, 1e-30)));
            if (score > best_score) {
                best_score = score;
                best_cluster = c;
            }
        }

        if (best_cluster < 0 || best_score < kGreedyClusterThreshold) {
            best_cluster = static_cast<int>(counts.size());
            counts.push_back(0);
            centroids.resize(static_cast<size_t>(best_cluster + 1) * dim, 0.0);
        }

        labels[r] = best_cluster;
        double * centroid = centroids.data() + static_cast<size_t>(best_cluster) * dim;
        const int new_count = ++counts[best_cluster];
        for (int d = 0; d < dim; ++d) {
            centroid[d] += (feat[d] - centroid[d]) / static_cast<double>(new_count);
        }
    }

    return labels;
}

std::vector<float> compute_centroids_from_labels(const std::vector<float> & filtered_embeddings,
                                                 const std::vector<int> & labels) {
    int clusters = 0;
    for (int label : labels) {
        clusters = std::max(clusters, label + 1);
    }
    std::vector<float> centroids(static_cast<size_t>(clusters) * kEmbeddingDim, 0.0f);
    std::vector<int> counts(clusters, 0);
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        const int cluster = labels[i];
        ++counts[cluster];
        const float * src = filtered_embeddings.data() + static_cast<size_t>(i) * kEmbeddingDim;
        float * dst = centroids.data() + static_cast<size_t>(cluster) * kEmbeddingDim;
        for (int d = 0; d < kEmbeddingDim; ++d) {
            dst[d] += src[d];
        }
    }
    for (int c = 0; c < clusters; ++c) {
        float * dst = centroids.data() + static_cast<size_t>(c) * kEmbeddingDim;
        const float inv = 1.0f / std::max(1, counts[c]);
        for (int d = 0; d < kEmbeddingDim; ++d) {
            dst[d] *= inv;
        }
    }
    return centroids;
}

void assign_chunk_speakers(const std::vector<float> & embeddings,
                           int num_chunks,
                           const std::vector<float> & centroids,
                           int num_clusters,
                           std::vector<int> & hard_clusters) {
    hard_clusters.assign(static_cast<size_t>(num_chunks) * kLocalSpeakers, -1);
    if (num_clusters <= 0) {
        return;
    }

    std::vector<float> scores(static_cast<size_t>(kLocalSpeakers) * num_clusters, -1e9f);
    for (int c = 0; c < num_chunks; ++c) {
        for (int s = 0; s < kLocalSpeakers; ++s) {
            const float * emb = embeddings.data() + (static_cast<size_t>(c) * kLocalSpeakers + s) * kEmbeddingDim;
            bool valid = true;
            for (int d = 0; d < kEmbeddingDim; ++d) {
                valid = valid && std::isfinite(emb[d]);
            }
            if (!valid) {
                continue;
            }
            for (int k = 0; k < num_clusters; ++k) {
                const float * centroid = centroids.data() + static_cast<size_t>(k) * kEmbeddingDim;
                scores[static_cast<size_t>(s) * num_clusters + k] = static_cast<float>(cosine_similarity(emb, centroid, kEmbeddingDim));
            }
        }

        int best_perm[3] = {0, 0, 0};
        double best_total = -DBL_MAX;
        for (int a = 0; a < num_clusters; ++a) {
            const double sa = scores[0 * num_clusters + a];
            for (int b = 0; b < num_clusters; ++b) {
                if (num_clusters >= 2 && b == a) {
                    continue;
                }
                const double sb = scores[1 * num_clusters + b];
                for (int d = 0; d < num_clusters; ++d) {
                    if ((num_clusters >= 2 && d == a) || (num_clusters >= 3 && d == b)) {
                        continue;
                    }
                    const double sd = scores[2 * num_clusters + d];
                    const double total = sa + sb + sd;
                    if (total > best_total) {
                        best_total = total;
                        best_perm[0] = a;
                        best_perm[1] = b;
                        best_perm[2] = d;
                    }
                }
            }
        }
        for (int s = 0; s < kLocalSpeakers; ++s) {
            hard_clusters[static_cast<size_t>(c) * kLocalSpeakers + s] = best_perm[s];
        }
    }
}

void build_clustered_segmentations(const std::vector<float> & binarized,
                                   const std::vector<int> & hard_clusters,
                                   int num_chunks,
                                   int num_clusters,
                                   std::vector<float> & clustered) {
    clustered.assign(static_cast<size_t>(num_chunks) * kFramesPerChunk * num_clusters, std::nanf(""));
    for (int c = 0; c < num_chunks; ++c) {
        const float * seg_chunk = binarized.data() + static_cast<size_t>(c) * kFramesPerChunk * kLocalSpeakers;
        for (int s = 0; s < kLocalSpeakers; ++s) {
            const int cluster = hard_clusters[static_cast<size_t>(c) * kLocalSpeakers + s];
            if (cluster < 0 || cluster >= num_clusters) {
                continue;
            }
            for (int f = 0; f < kFramesPerChunk; ++f) {
                const float value = seg_chunk[static_cast<size_t>(f) * kLocalSpeakers + s];
                float & dst = clustered[(static_cast<size_t>(c) * kFramesPerChunk + f) * num_clusters + cluster];
                dst = std::isnan(dst) ? value : std::max(dst, value);
            }
        }
    }
}

void to_diarization(const std::vector<float> & clustered,
                    int num_chunks,
                    int num_clusters,
                    const std::vector<int> & speaker_count,
                    int total_frames,
                    const SlidingWindowParams & chunk_window,
                    const SlidingWindowParams & frame_window,
                    std::vector<float> & discrete) {
    std::vector<float> activations;
    int activation_frames = 0;
    aggregate_chunks(clustered.data(), num_chunks, kFramesPerChunk, num_clusters, chunk_window, frame_window, activations, activation_frames, true, 0.0f);
    const int frames = std::min(total_frames, activation_frames);
    discrete.assign(static_cast<size_t>(frames) * num_clusters, 0.0f);

    std::vector<int> order(num_clusters, 0);
    for (int t = 0; t < frames; ++t) {
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return activations[static_cast<size_t>(t) * num_clusters + a] >
                   activations[static_cast<size_t>(t) * num_clusters + b];
        });
        const int keep = std::min<int>(speaker_count[t], num_clusters);
        for (int i = 0; i < keep; ++i) {
            discrete[static_cast<size_t>(t) * num_clusters + order[i]] = 1.0f;
        }
    }
}

std::vector<RTTMSegment> diarization_to_rttm(const std::vector<float> & discrete,
                                             int num_clusters) {
    const int frames = num_clusters == 0 ? 0 : static_cast<int>(discrete.size()) / num_clusters;
    std::vector<RTTMSegment> segments;

    for (int k = 0; k < num_clusters; ++k) {
        const std::string speaker = "SPEAKER_" + (k < 10 ? std::string("0") : std::string()) + std::to_string(k);
        bool active = false;
        int start = 0;
        for (int f = 0; f <= frames; ++f) {
            const bool value = (f < frames) && (discrete[static_cast<size_t>(f) * num_clusters + k] == 1.0f);
            if (value && !active) {
                active = true;
                start = f;
            } else if (!value && active) {
                const double begin = start * kFrameStep + 0.5 * kFrameDuration;
                const double duration = (f - start) * kFrameStep;
                if (duration > 0.0) {
                    segments.push_back({begin, duration, speaker});
                }
                active = false;
            }
        }
    }

    std::sort(segments.begin(), segments.end(), [](const RTTMSegment & a, const RTTMSegment & b) {
        return a.start < b.start;
    });
    return segments;
}

int compute_num_chunks(int n_samples) {
    if (n_samples <= kChunkSamples) {
        return 1;
    }
    return 1 + (n_samples - kChunkSamples + kStepSamples - 1) / kStepSamples;
}

bool extract_embeddings_for_chunks(const std::vector<float> & audio,
                                   int num_chunks,
                                   const std::vector<float> & binarized,
                                   const EmbeddingModel & model,
                                   BackendState & backend_state,
                                   std::vector<float> & embeddings) {
    embeddings.assign(static_cast<size_t>(num_chunks) * kLocalSpeakers * kEmbeddingDim, std::nanf(""));

    const int padded_len = (num_chunks - 1) * kStepSamples + kChunkSamples;
    std::vector<float> padded(static_cast<size_t>(padded_len), 0.0f);
    std::copy(audio.begin(), audio.end(), padded.begin());
    FbankData global_fbank = compute_fbank(padded.data(), padded_len, false);
    if (global_fbank.frames <= 0) {
        std::fprintf(stderr, "Error: failed to compute fbank features\n");
        return false;
    }

    const int frames_per_shift = kStepSamples / 160;
    const int frames_per_chunk = std::min(global_fbank.frames, 1 + (kChunkSamples - 400) / 160);
    std::vector<float> chunk_fbank(static_cast<size_t>(frames_per_chunk) * kMelBins, 0.0f);
    std::vector<float> masked(static_cast<size_t>(frames_per_chunk) * kMelBins, 0.0f);

    for (int c = 0; c < num_chunks; ++c) {
        const int frame_offset = c * frames_per_shift;
        const int usable_frames = std::min(frames_per_chunk, global_fbank.frames - frame_offset);
        std::memcpy(chunk_fbank.data(),
                    global_fbank.data.data() + static_cast<size_t>(frame_offset) * kMelBins,
                    static_cast<size_t>(usable_frames) * kMelBins * sizeof(float));
        apply_cmn(chunk_fbank.data(), usable_frames);

        const float * seg_chunk = binarized.data() + static_cast<size_t>(c) * kFramesPerChunk * kLocalSpeakers;
        for (int s = 0; s < kLocalSpeakers; ++s) {
            bool any_active = false;
            for (int f = 0; f < kFramesPerChunk; ++f) {
                any_active = any_active || (seg_chunk[static_cast<size_t>(f) * kLocalSpeakers + s] != 0.0f);
            }
            if (!any_active) {
                continue;
            }

            for (int ft = 0; ft < usable_frames; ++ft) {
                int seg_frame = static_cast<int>((static_cast<long long>(ft) * kFramesPerChunk) / std::max(1, usable_frames));
                seg_frame = std::min(seg_frame, kFramesPerChunk - 1);
                const float mask = seg_chunk[static_cast<size_t>(seg_frame) * kLocalSpeakers + s];
                float * dst_row = masked.data() + static_cast<size_t>(ft) * kMelBins;
                const float * src_row = chunk_fbank.data() + static_cast<size_t>(ft) * kMelBins;
                if (mask == 0.0f) {
                    std::fill(dst_row, dst_row + kMelBins, 0.0f);
                } else {
                    std::memcpy(dst_row, src_row, static_cast<size_t>(kMelBins) * sizeof(float));
                }
            }

            float * out = embeddings.data() + (static_cast<size_t>(c) * kLocalSpeakers + s) * kEmbeddingDim;
            if (!infer_embedding(model, backend_state, masked.data(), usable_frames, out)) {
                return false;
            }
        }
    }

    return true;
}

bool diarize_impl(const DiarizationConfig & config,
                  const float * audio,
                  int n_samples,
                  DiarizationResult & result) {
    if (!audio || n_samples <= 0) {
        std::fprintf(stderr, "Error: empty audio input\n");
        return false;
    }

    SegmentationModel seg_model;
    EmbeddingModel emb_model;
    BackendState seg_state;
    BackendState emb_state;
    PldaModel plda;

    const auto cleanup = [&]() {
        free_backend_state(seg_state);
        free_backend_state(emb_state);
        free_weight_store(seg_model.weights);
        free_weight_store(emb_model.weights);
    };

    std::string seg_backend = config.ggml_backend;
    std::string emb_backend = config.ggml_backend;

    if (!load_segmentation_model(config.seg_model_path, seg_model, seg_backend, config.ggml_gpu_device)) {
        cleanup();
        return false;
    }
    if (!load_embedding_model(config.emb_model_path, emb_model, emb_backend, config.ggml_gpu_device)) {
        cleanup();
        return false;
    }
    if (!config.plda_path.empty()) {
        if (!load_plda(config.plda_path, plda)) {
            std::fprintf(stderr, "Warning: failed to load PLDA, falling back to cosine clustering\n");
        }
    }

    if (!init_backend_state(seg_state, seg_backend, config.ggml_gpu_device, kSegGraphNodes) ||
        !init_backend_state(emb_state, emb_backend, config.ggml_gpu_device, kEmbGraphNodes)) {
        cleanup();
        return false;
    }

    const int num_chunks = compute_num_chunks(n_samples);
    const uintptr_t seg_lstm_cuda_options = pack_seg_lstm_cuda_options(config);
    std::vector<float> padded_audio(static_cast<size_t>((num_chunks - 1) * kStepSamples + kChunkSamples), 0.0f);
    std::memcpy(padded_audio.data(), audio, static_cast<size_t>(n_samples) * sizeof(float));

    std::vector<float> chunk_logits(static_cast<size_t>(num_chunks) * kFramesPerChunk * kPowersetClasses, 0.0f);
    std::vector<float> binarized(static_cast<size_t>(num_chunks) * kFramesPerChunk * kLocalSpeakers, 0.0f);
    std::vector<float> chunk(kChunkSamples, 0.0f);

    for (int c = 0; c < num_chunks; ++c) {
        std::memcpy(chunk.data(), padded_audio.data() + static_cast<size_t>(c) * kStepSamples, static_cast<size_t>(kChunkSamples) * sizeof(float));
        float * logits = chunk_logits.data() + static_cast<size_t>(c) * kFramesPerChunk * kPowersetClasses;
        if (!infer_segmentation(seg_model, seg_state, chunk.data(), seg_lstm_cuda_options, logits)) {
            cleanup();
            return false;
        }
        powerset_to_multilabel(logits, binarized.data() + static_cast<size_t>(c) * kFramesPerChunk * kLocalSpeakers);
    }

    const SlidingWindowParams chunk_window{0.0, kChunkDuration, kChunkStep};
    const SlidingWindowParams frame_window{0.0, kFrameDuration, kFrameStep};
    std::vector<int> speaker_count;
    int total_frames = 0;
    compute_speaker_count(binarized.data(), num_chunks, chunk_window, frame_window, speaker_count, total_frames);
    const int max_count = speaker_count.empty() ? 0 : *std::max_element(speaker_count.begin(), speaker_count.end());

    if (max_count == 0) {
        result.segments.clear();
        if (!config.output_path.empty()) {
            write_rttm_lines({}, basename_without_ext(config.audio_path), config.output_path);
        }
        cleanup();
        return true;
    }

    std::vector<float> embeddings;
    if (!config.bypass_embeddings) {
        if (!extract_embeddings_for_chunks(std::vector<float>(audio, audio + n_samples), num_chunks, binarized, emb_model, emb_state, embeddings)) {
            cleanup();
            return false;
        }
    } else {
        embeddings.assign(static_cast<size_t>(num_chunks) * kLocalSpeakers * kEmbeddingDim, 0.0f);
    }

    std::vector<int> hard_clusters(static_cast<size_t>(num_chunks) * kLocalSpeakers, 0);
    int num_clusters = config.bypass_embeddings ? kLocalSpeakers : 1;

    if (!config.bypass_embeddings) {
        std::vector<float> filtered_embeddings;
        std::vector<int> filtered_chunks;
        std::vector<int> filtered_speakers;
        filter_embeddings(embeddings, binarized, num_chunks, filtered_embeddings, filtered_chunks, filtered_speakers);

        if (filtered_chunks.size() >= 2) {
            if (plda.loaded) {
                std::vector<std::vector<float>> raw_embeddings(filtered_chunks.size(), std::vector<float>(kEmbeddingDim));
                for (size_t i = 0; i < filtered_chunks.size(); ++i) {
                    const float * src = filtered_embeddings.data() + i * kEmbeddingDim;
                    std::copy(src, src + kEmbeddingDim, raw_embeddings[i].begin());
                }

                const vbx_model vbx = to_vbx_model(plda);
                const std::vector<std::vector<float>> vbx_embeddings =
                    vbx_transform_embeddings_batch(vbx, raw_embeddings, false);
                vbx_params params;
                params.threshold = 0.6f;
                const std::vector<int> labels = cluster_vbx(vbx, vbx_embeddings, raw_embeddings, params);

                std::vector<float> centroids = compute_centroids_from_labels(filtered_embeddings, labels);
                num_clusters = centroids.empty() ? 1 : static_cast<int>(centroids.size() / kEmbeddingDim);
                assign_chunk_speakers(embeddings, num_chunks, centroids, num_clusters, hard_clusters);
                for (size_t i = 0; i < labels.size(); ++i) {
                    hard_clusters[static_cast<size_t>(filtered_chunks[i]) * kLocalSpeakers + filtered_speakers[i]] = labels[i];
                }
            } else {
                std::vector<double> clustering_features(static_cast<size_t>(filtered_chunks.size()) * kEmbeddingDim);
                clustering_features.resize(static_cast<size_t>(filtered_chunks.size()) * kEmbeddingDim);
                for (size_t i = 0; i < clustering_features.size(); ++i) {
                    clustering_features[i] = filtered_embeddings[i];
                }
                normalize_rows(clustering_features, static_cast<int>(filtered_chunks.size()), kEmbeddingDim);

                const std::vector<int> labels = greedy_cluster(clustering_features, static_cast<int>(filtered_chunks.size()), kEmbeddingDim);
                std::vector<float> centroids = compute_centroids_from_labels(filtered_embeddings, labels);
                num_clusters = centroids.empty() ? 1 : static_cast<int>(centroids.size() / kEmbeddingDim);
                assign_chunk_speakers(embeddings, num_chunks, centroids, num_clusters, hard_clusters);
                for (size_t i = 0; i < labels.size(); ++i) {
                    hard_clusters[static_cast<size_t>(filtered_chunks[i]) * kLocalSpeakers + filtered_speakers[i]] = labels[i];
                }
            }
        } else {
            std::fill(hard_clusters.begin(), hard_clusters.end(), 0);
            num_clusters = 1;
        }
    } else {
        for (int c = 0; c < num_chunks; ++c) {
            for (int s = 0; s < kLocalSpeakers; ++s) {
                hard_clusters[static_cast<size_t>(c) * kLocalSpeakers + s] = s;
            }
        }
    }

    for (int c = 0; c < num_chunks; ++c) {
        const float * seg_chunk = binarized.data() + static_cast<size_t>(c) * kFramesPerChunk * kLocalSpeakers;
        for (int s = 0; s < kLocalSpeakers; ++s) {
            float sum = 0.0f;
            for (int f = 0; f < kFramesPerChunk; ++f) {
                sum += seg_chunk[static_cast<size_t>(f) * kLocalSpeakers + s];
            }
            if (sum == 0.0f) {
                hard_clusters[static_cast<size_t>(c) * kLocalSpeakers + s] = -1;
            }
        }
    }

    std::vector<float> clustered;
    build_clustered_segmentations(binarized, hard_clusters, num_chunks, num_clusters, clustered);

    std::vector<float> discrete;
    to_diarization(clustered, num_chunks, num_clusters, speaker_count, total_frames, chunk_window, frame_window, discrete);

    const std::vector<RTTMSegment> rttm = diarization_to_rttm(discrete, num_clusters);
    result.segments.clear();
    result.segments.reserve(rttm.size());
    for (const auto & seg : rttm) {
        result.segments.push_back({seg.start, seg.duration, seg.speaker});
    }

    if (!config.output_path.empty()) {
        if (!write_rttm_lines(rttm, basename_without_ext(config.audio_path), config.output_path)) {
            cleanup();
            return false;
        }
    }

    cleanup();
    return true;
}

}  // namespace

bool diarize(const DiarizationConfig & config, DiarizationResult & result) {
    WavData wav;
    if (!load_wav_file(config.audio_path, wav)) {
        return false;
    }
    return diarize_impl(config, wav.mono.data(), static_cast<int>(wav.mono.size()), result);
}

bool diarize_from_samples(const DiarizationConfig & config,
                          const float * audio,
                          int n_samples,
                          DiarizationResult & result) {
    return diarize_impl(config, audio, n_samples, result);
}
