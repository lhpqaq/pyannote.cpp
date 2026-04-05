#pragma once

#include <string>
#include <vector>

namespace pyannote_example {

struct DiarizationConfig {
    std::string seg_model_path;
    std::string emb_model_path;
    std::string audio_path;
    std::string plda_path;
    std::string coreml_path;
    std::string seg_coreml_path;
    std::string output_path;
    std::string dump_stage;
};

struct DiarizationResult {
    struct Segment {
        double start = 0.0;
        double duration = 0.0;
        std::string speaker;
    };

    std::vector<Segment> segments;
};

bool diarize_file(const DiarizationConfig & config, DiarizationResult & result);

}  // namespace pyannote_example
