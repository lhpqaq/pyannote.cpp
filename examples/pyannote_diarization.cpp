#include "pyannote_diarization.h"

#include "common_audio.h"
#include "pyannote/diarization.h"

#include <cstdio>
#include <string>
#include <vector>

namespace pyannote_example {

namespace {

constexpr int kExpectedSampleRate = 16000;

std::string extract_uri(const std::string & audio_path) {
    std::string uri = audio_path;

    const size_t last_slash = uri.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        uri = uri.substr(last_slash + 1);
    }

    const size_t last_dot = uri.find_last_of('.');
    if (last_dot != std::string::npos) {
        uri = uri.substr(0, last_dot);
    }

    return uri.empty() ? "audio" : uri;
}

const char * null_sink_path() {
#ifdef _WIN32
    return "NUL";
#else
    return "/dev/null";
#endif
}

bool load_audio_mono_16khz(const std::string & audio_path, std::vector<float> & mono_audio) {
    wav_file wav;
    if (!read_wav(audio_path, wav)) {
        std::fprintf(stderr, "Error: failed to read WAV file '%s'\n", audio_path.c_str());
        return false;
    }

    if (wav.sample_rate != kExpectedSampleRate) {
        std::fprintf(stderr, "Error: expected %d Hz WAV, got %d Hz\n",
                     kExpectedSampleRate, wav.sample_rate);
        return false;
    }

    if (wav.channels <= 0) {
        std::fprintf(stderr, "Error: invalid channel count %d\n", wav.channels);
        return false;
    }

    if (wav.channels == 1) {
        mono_audio = std::move(wav.data);
        return true;
    }

    const size_t frames = wav.data.size() / static_cast<size_t>(wav.channels);
    mono_audio.assign(frames, 0.0f);

    for (size_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < wav.channels; ++ch) {
            sum += wav.data[i * static_cast<size_t>(wav.channels) + static_cast<size_t>(ch)];
        }
        mono_audio[i] = sum / static_cast<float>(wav.channels);
    }

    return true;
}

bool write_rttm(const DiarizationResult & result,
                const std::string & audio_path,
                const std::string & output_path) {
    const std::string uri = extract_uri(audio_path);
    FILE * file = output_path.empty() ? stdout : std::fopen(output_path.c_str(), "w");

    if (!file) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", output_path.c_str());
        return false;
    }

    for (const auto & segment : result.segments) {
        std::fprintf(file,
                     "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n",
                     uri.c_str(),
                     segment.start,
                     segment.duration,
                     segment.speaker.c_str());
    }

    if (file != stdout) {
        std::fclose(file);
    }

    return true;
}

::DiarizationConfig to_internal_config(const DiarizationConfig & config) {
    ::DiarizationConfig internal;
    internal.seg_model_path = config.seg_model_path;
    internal.emb_model_path = config.emb_model_path;
    internal.audio_path = config.audio_path;
    internal.plda_path = config.plda_path;
    internal.coreml_path = config.coreml_path;
    internal.seg_coreml_path = config.seg_coreml_path;
    internal.output_path = null_sink_path();
    internal.dump_stage = config.dump_stage;
    internal.ggml_backend = config.ggml_backend;
    internal.ggml_gpu_device = config.ggml_gpu_device;
    return internal;
}

void copy_result(const ::DiarizationResult & internal, DiarizationResult & result) {
    result.segments.clear();
    result.segments.reserve(internal.segments.size());

    for (const auto & segment : internal.segments) {
        result.segments.push_back({segment.start, segment.duration, segment.speaker});
    }
}

}  // namespace

bool diarize_file(const DiarizationConfig & config, DiarizationResult & result) {
    std::vector<float> audio;
    if (!load_audio_mono_16khz(config.audio_path, audio)) {
        return false;
    }

    const ::DiarizationConfig internal_config = to_internal_config(config);
    ::DiarizationResult internal_result;

    if (!::diarize_from_samples(internal_config, audio.data(), static_cast<int>(audio.size()), internal_result)) {
        return false;
    }

    copy_result(internal_result, result);
    return write_rttm(result, config.audio_path, config.output_path);
}

}  // namespace pyannote_example
