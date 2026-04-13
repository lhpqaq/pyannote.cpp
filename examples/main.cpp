#include "pyannote_diarization.h"

#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s <seg.gguf> <emb.gguf> <audio.wav> [options]\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Positional arguments:\n");
    fprintf(stderr, "  seg.gguf              Path to segmentation model (GGUF)\n");
    fprintf(stderr, "  emb.gguf              Path to embedding model (GGUF)\n");
    fprintf(stderr, "  audio.wav             Path to audio file (16kHz mono PCM WAV)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --plda <path>         Path to PLDA GGUF file\n");
    fprintf(stderr, "  --coreml <path>       Path to CoreML embedding model (.mlpackage)\n");
    fprintf(stderr, "  --seg-coreml <path>   Path to CoreML segmentation model (.mlpackage)\n");
    fprintf(stderr, "  --backend <name>      GGML backend: cpu | cuda | auto\n");
    fprintf(stderr, "  --gpu-device <id>     CUDA device index (default: 0)\n");
    fprintf(stderr, "  --seg-lstm-coop-warps <n>\n");
    fprintf(stderr, "                        CUDA bidirectional LSTM warp groups (default: 4)\n");
    fprintf(stderr, "  --no-seg-lstm-coop    Disable CUDA fused segmentation LSTM custom op\n");
    fprintf(stderr, "  --no-seg-lstm-coop-warp\n");
    fprintf(stderr, "                        Disable warp-tuned CUDA launch for segmentation LSTM\n");
    fprintf(stderr, "  --no-seg-lstm-coop-warp-nosh\n");
    fprintf(stderr, "                        Disable the no-shared-memory launch preference flag\n");
    fprintf(stderr, "  --no-seg-lstm-coop-bidir\n");
    fprintf(stderr, "                        Disable CUDA bidirectional fused handling for segmentation LSTM\n");
    fprintf(stderr, "  -o, --output <path>   Output RTTM file (default: stdout)\n");
    fprintf(stderr, "  --dump-stage <name>   Dump intermediate stage to binary file\n");
    fprintf(stderr, "  --help                Print this help message\n");
}

static bool consume_value(int argc, char ** argv, int & i, const char * option, std::string & value) {
    if (i + 1 >= argc) {
        fprintf(stderr, "Error: option '%s' requires a value\n\n", option);
        print_usage(argv[0]);
        return false;
    }

    value = argv[++i];
    return true;
}

static bool consume_int_value(int argc, char ** argv, int & i, const char * option, int & value) {
    if (i + 1 >= argc) {
        fprintf(stderr, "Error: option '%s' requires a value\n\n", option);
        print_usage(argv[0]);
        return false;
    }

    const char * raw = argv[++i];
    char * end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (!end || *end != '\0') {
        fprintf(stderr, "Error: option '%s' expects an integer, got '%s'\n\n", option, raw);
        print_usage(argv[0]);
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    if (argc < 4) {
        fprintf(stderr, "Error: expected 3 positional arguments (seg.gguf, emb.gguf, audio.wav)\n\n");
        print_usage(argv[0]);
        return 1;
    }

    pyannote_example::DiarizationConfig config;
    config.seg_model_path = argv[1];
    config.emb_model_path = argv[2];
    config.audio_path     = argv[3];

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--plda") {
            if (!consume_value(argc, argv, i, "--plda", config.plda_path)) {
                return 1;
            }
        } else if (arg == "--coreml") {
            if (!consume_value(argc, argv, i, "--coreml", config.coreml_path)) {
                return 1;
            }
        } else if (arg == "--seg-coreml") {
            if (!consume_value(argc, argv, i, "--seg-coreml", config.seg_coreml_path)) {
                return 1;
            }
        } else if (arg == "--backend") {
            if (!consume_value(argc, argv, i, "--backend", config.ggml_backend)) {
                return 1;
            }
        } else if (arg == "--gpu-device") {
            if (!consume_int_value(argc, argv, i, "--gpu-device", config.ggml_gpu_device)) {
                return 1;
            }
        } else if (arg == "--seg-lstm-coop-warps") {
            if (!consume_int_value(argc, argv, i, "--seg-lstm-coop-warps", config.seg_lstm_coop_warps)) {
                return 1;
            }
            if (config.seg_lstm_coop_warps <= 0) {
                fprintf(stderr, "Error: option '--seg-lstm-coop-warps' expects a positive integer\n\n");
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--no-seg-lstm-coop") {
            config.seg_lstm_coop = false;
        } else if (arg == "--no-seg-lstm-coop-warp") {
            config.seg_lstm_coop_warp = false;
        } else if (arg == "--no-seg-lstm-coop-warp-nosh") {
            config.seg_lstm_coop_warp_nosh = false;
        } else if (arg == "--no-seg-lstm-coop-bidir") {
            config.seg_lstm_coop_bidir = false;
        } else if (arg == "-o" || arg == "--output") {
            if (!consume_value(argc, argv, i, arg.c_str(), config.output_path)) {
                return 1;
            }
        } else if (arg == "--dump-stage") {
            if (!consume_value(argc, argv, i, "--dump-stage", config.dump_stage)) {
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    pyannote_example::DiarizationResult result;
    if (!pyannote_example::diarize_file(config, result)) {
        fprintf(stderr, "Error: diarization failed\n");
        return 1;
    }

    return 0;
}
