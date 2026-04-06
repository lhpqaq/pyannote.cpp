#pragma once

#include <string>
#include <vector>

struct DiarizationConfig {
    std::string seg_model_path;
    std::string emb_model_path;
    std::string audio_path;
    std::string plda_path;
    std::string coreml_path;
    std::string seg_coreml_path;
    std::string output_path;
    std::string dump_stage;
    std::string ggml_backend = "cpu";
    int ggml_gpu_device = 0;
    bool bypass_embeddings = false;
};

struct DiarizationResult {
    struct Segment {
        double start = 0.0;
        double duration = 0.0;
        std::string speaker;
    };

    std::vector<Segment> segments;
};

bool diarize(const DiarizationConfig & config, DiarizationResult & result);
bool diarize_from_samples(const DiarizationConfig & config,
                          const float * audio,
                          int n_samples,
                          DiarizationResult & result);
