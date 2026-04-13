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
    bool seg_lstm_coop = true;
    bool seg_lstm_coop_warp = true;
    int seg_lstm_coop_warps = 4;
    bool seg_lstm_coop_warp_nosh = true;
    bool seg_lstm_coop_bidir = true;
};

struct DiarizationResult {
    struct Stats {
        double segmentation_ms = 0.0;
        double embedding_ms = 0.0;
        int segment_count = 0;
        int speaker_count = 0;
    };

    struct Segment {
        double start = 0.0;
        double duration = 0.0;
        std::string speaker;
    };

    Stats stats;
    std::vector<Segment> segments;
};

bool diarize(const DiarizationConfig & config, DiarizationResult & result);
bool diarize_from_samples(const DiarizationConfig & config,
                          const float * audio,
                          int n_samples,
                          DiarizationResult & result);
