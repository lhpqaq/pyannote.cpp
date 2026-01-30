#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "common.h"
#include "common_audio.h"
#include "vbx.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <tuple>

// --- Constants ---
static const int EMBEDDING_DIM = 256;
static const int MAX_EMBEDDING_FRAMES = 100;
static const int SAMPLE_RATE = 16000;
static const float WINDOW_DURATION = 10.0f;
static const float STEP_DURATION = 1.0f;

// --- Structures ---
struct resnet_block {
    struct ggml_tensor * conv1_w;
    struct ggml_tensor * conv1_b;
    struct ggml_tensor * conv2_w;
    struct ggml_tensor * conv2_b;
    struct ggml_tensor * ds_w; 
    struct ggml_tensor * ds_b; 
};

struct pyannote_embedding_model {
    struct ggml_tensor * conv1_w;
    struct ggml_tensor * conv1_b;
    
    std::vector<resnet_block> layer1; 
    std::vector<resnet_block> layer2; 
    std::vector<resnet_block> layer3; 
    std::vector<resnet_block> layer4; 
    
    struct ggml_tensor * seg_1_w;
    struct ggml_tensor * seg_1_b;
    
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

struct pyannote_segmentation_model {
    struct ggml_tensor * wav_norm1d_w;
    struct ggml_tensor * wav_norm1d_b;
    
    std::vector<struct ggml_tensor *> sinc_conv_w;
    std::vector<struct ggml_tensor *> sinc_conv_b;
    
    std::vector<struct ggml_tensor *> sinc_norm_w;
    std::vector<struct ggml_tensor *> sinc_norm_b;
    
    struct lstm_layer {
        struct ggml_tensor * w_ih;
        struct ggml_tensor * w_hh;
        struct ggml_tensor * b_ih;
        struct ggml_tensor * b_hh;
        // Pre-dequantized weights for fast inference
        std::vector<float> w_ih_f32;
        std::vector<float> w_hh_f32;
        std::vector<float> b_ih_f32;
        std::vector<float> b_hh_f32;
    };
    std::vector<lstm_layer> lstm_fwd;
    std::vector<lstm_layer> lstm_rev;
    
    std::vector<struct ggml_tensor *> linear_w;
    std::vector<struct ggml_tensor *> linear_b;
    
    struct ggml_tensor * classifier_w;
    struct ggml_tensor * classifier_b;
    
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

// Helper function to dequantize tensor to f32 vector
static std::vector<float> dequantize_tensor(struct ggml_tensor * t) {
    std::vector<float> res(ggml_nelements(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_cpu_fp16_to_fp32((const ggml_fp16_t *)t->data, res.data(), res.size());
    } else {
        memcpy(res.data(), t->data, res.size() * sizeof(float));
    }
    return res;
}

void load_embedding_model(const std::string & fname, pyannote_embedding_model & model) {
    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &model.ctx,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        exit(1);
    }
    
    auto get_tensor = [&](const std::string & name) {
        struct ggml_tensor * t = ggml_get_tensor(model.ctx, name.c_str());
        if (!t) fprintf(stderr, "Warning: tensor %s not found\n", name.c_str());
        return t;
    };
    
    model.conv1_w = get_tensor("resnet.conv1.weight");
    model.conv1_b = get_tensor("resnet.conv1.bias");
    
    auto load_layer = [&](std::vector<resnet_block> & layer, int num_blocks, int layer_idx) {
        layer.resize(num_blocks);
        for (int i=0; i<num_blocks; i++) {
            std::string prefix = "resnet.layer" + std::to_string(layer_idx) + "." + std::to_string(i) + ".";
            layer[i].conv1_w = get_tensor(prefix + "conv1.weight");
            layer[i].conv1_b = get_tensor(prefix + "conv1.bias");
            layer[i].conv2_w = get_tensor(prefix + "conv2.weight");
            layer[i].conv2_b = get_tensor(prefix + "conv2.bias");
            
            // Shortcut
            layer[i].ds_w = ggml_get_tensor(model.ctx, (prefix + "shortcut.0.weight").c_str());
            if (layer[i].ds_w) {
                layer[i].ds_b = get_tensor(prefix + "shortcut.0.bias");
            } else {
                layer[i].ds_w = nullptr;
                layer[i].ds_b = nullptr;
            }
        }
    };
    
    load_layer(model.layer1, 3, 1);
    load_layer(model.layer2, 4, 2);
    load_layer(model.layer3, 6, 3);
    load_layer(model.layer4, 3, 4);
    
    model.seg_1_w = get_tensor("resnet.seg_1.weight");
    model.seg_1_b = get_tensor("resnet.seg_1.bias");
    
    ggml_backend_t backend = ggml_backend_cpu_init();
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, backend);
    
    FILE * fp = fopen(fname.c_str(), "rb");
    if (!fp) exit(1);
    
    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    
    struct ggml_tensor * t = ggml_get_first_tensor(model.ctx);
    while (t) {
        int idx = gguf_find_tensor(ctx_gguf, t->name);
        if (idx >= 0) {
            size_t offset = data_offset + gguf_get_tensor_offset(ctx_gguf, idx);
            fseek(fp, offset, SEEK_SET);
            void * ptr = t->data;
            size_t ret = fread(ptr, 1, ggml_nbytes(t), fp);
            if (ret != ggml_nbytes(t)) {
                printf("ERROR: fread failed for %s. Expected %zu bytes, got %zu\n", t->name, ggml_nbytes(t), ret);
            }
        } else {
            printf("WARNING: Tensor %s not found in GGUF file\n", t->name);
        }
        t = ggml_get_next_tensor(model.ctx, t);
    }
    
    fclose(fp);
    gguf_free(ctx_gguf);
    ggml_backend_free(backend);
}

void load_segmentation_model(const std::string & fname, pyannote_segmentation_model & model) {
    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &model.ctx,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        exit(1);
    }
    
    model.wav_norm1d_w = ggml_get_tensor(model.ctx, "sincnet.wav_norm1d.weight");
    model.wav_norm1d_b = ggml_get_tensor(model.ctx, "sincnet.wav_norm1d.bias");
    
    model.sinc_conv_w.resize(3);
    model.sinc_conv_b.resize(3);
    
    model.sinc_conv_w[0] = ggml_get_tensor(model.ctx, "sincnet.conv1d.0.weight");
    model.sinc_conv_w[1] = ggml_get_tensor(model.ctx, "sincnet.conv1d.1.weight");
    model.sinc_conv_b[1] = ggml_get_tensor(model.ctx, "sincnet.conv1d.1.bias");
    model.sinc_conv_w[2] = ggml_get_tensor(model.ctx, "sincnet.conv1d.2.weight");
    model.sinc_conv_b[2] = ggml_get_tensor(model.ctx, "sincnet.conv1d.2.bias");
    
    model.sinc_norm_w.resize(3);
    model.sinc_norm_b.resize(3);
    for (int i=0; i<3; i++) {
        model.sinc_norm_w[i] = ggml_get_tensor(model.ctx, ("sincnet.norm1d." + std::to_string(i) + ".weight").c_str());
        model.sinc_norm_b[i] = ggml_get_tensor(model.ctx, ("sincnet.norm1d." + std::to_string(i) + ".bias").c_str());
    }
    
    int num_layers = 4;
    model.lstm_fwd.resize(num_layers);
    model.lstm_rev.resize(num_layers);
    
    for (int i=0; i<num_layers; i++) {
        std::string l = "l" + std::to_string(i);
        model.lstm_fwd[i].w_ih = ggml_get_tensor(model.ctx, ("lstm.weight_ih_" + l).c_str());
        model.lstm_fwd[i].w_hh = ggml_get_tensor(model.ctx, ("lstm.weight_hh_" + l).c_str());
        model.lstm_fwd[i].b_ih = ggml_get_tensor(model.ctx, ("lstm.bias_ih_" + l).c_str());
        model.lstm_fwd[i].b_hh = ggml_get_tensor(model.ctx, ("lstm.bias_hh_" + l).c_str());
        
        model.lstm_rev[i].w_ih = ggml_get_tensor(model.ctx, ("lstm.weight_ih_" + l + "_reverse").c_str());
        model.lstm_rev[i].w_hh = ggml_get_tensor(model.ctx, ("lstm.weight_hh_" + l + "_reverse").c_str());
        model.lstm_rev[i].b_ih = ggml_get_tensor(model.ctx, ("lstm.bias_ih_" + l + "_reverse").c_str());
        model.lstm_rev[i].b_hh = ggml_get_tensor(model.ctx, ("lstm.bias_hh_" + l + "_reverse").c_str());
    }
    
    model.linear_w.resize(2);
    model.linear_b.resize(2);
    for (int i=0; i<2; i++) {
        model.linear_w[i] = ggml_get_tensor(model.ctx, ("linear." + std::to_string(i) + ".weight").c_str());
        model.linear_b[i] = ggml_get_tensor(model.ctx, ("linear." + std::to_string(i) + ".bias").c_str());
    }
    
    model.classifier_w = ggml_get_tensor(model.ctx, "classifier.weight");
    model.classifier_b = ggml_get_tensor(model.ctx, "classifier.bias");
    
    ggml_backend_t backend = ggml_backend_cpu_init();
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, backend);
    
    FILE * fp = fopen(fname.c_str(), "rb");
    if (!fp) exit(1);
    
    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    
    struct ggml_tensor * t = ggml_get_first_tensor(model.ctx);
    while (t) {
        int idx = gguf_find_tensor(ctx_gguf, t->name);
        if (idx >= 0) {
            size_t offset = data_offset + gguf_get_tensor_offset(ctx_gguf, idx);
            fseek(fp, offset, SEEK_SET);
            void * ptr = t->data;
            size_t ret = fread(ptr, 1, ggml_nbytes(t), fp);
            (void)ret;
        }
        t = ggml_get_next_tensor(model.ctx, t);
    }
    fclose(fp);
    gguf_free(ctx_gguf);
    ggml_backend_free(backend);
    
    // Pre-dequantize LSTM weights for fast inference
    for (int i = 0; i < num_layers; i++) {
        model.lstm_fwd[i].w_ih_f32 = dequantize_tensor(model.lstm_fwd[i].w_ih);
        model.lstm_fwd[i].w_hh_f32 = dequantize_tensor(model.lstm_fwd[i].w_hh);
        model.lstm_fwd[i].b_ih_f32 = dequantize_tensor(model.lstm_fwd[i].b_ih);
        model.lstm_fwd[i].b_hh_f32 = dequantize_tensor(model.lstm_fwd[i].b_hh);
        
        model.lstm_rev[i].w_ih_f32 = dequantize_tensor(model.lstm_rev[i].w_ih);
        model.lstm_rev[i].w_hh_f32 = dequantize_tensor(model.lstm_rev[i].w_hh);
        model.lstm_rev[i].b_ih_f32 = dequantize_tensor(model.lstm_rev[i].b_ih);
        model.lstm_rev[i].b_hh_f32 = dequantize_tensor(model.lstm_rev[i].b_hh);
    }
}

std::vector<float> get_embedding(pyannote_embedding_model & model, const std::vector<float> & fbank, int T, int n_mels, bool normalize = true, ggml_backend_t external_backend = nullptr) {
    if (T > MAX_EMBEDDING_FRAMES) {
        std::vector<float> avg_embedding(EMBEDDING_DIM, 0.0f);
        int num_chunks = 0;
        for (int start = 0; start < T; start += MAX_EMBEDDING_FRAMES) {
            int chunk_len = std::min(MAX_EMBEDDING_FRAMES, T - start);
            if (chunk_len < 50) continue;
            std::vector<float> chunk_fbank(chunk_len * n_mels);
            for (int t = 0; t < chunk_len; t++) {
                for (int f = 0; f < n_mels; f++) {
                    chunk_fbank[t * n_mels + f] = fbank[(start + t) * n_mels + f];
                }
            }
            std::vector<float> chunk_emb = get_embedding(model, chunk_fbank, chunk_len, n_mels, false, external_backend);
            for (int i = 0; i < EMBEDDING_DIM; i++) avg_embedding[i] += chunk_emb[i];
            num_chunks++;
        }
        if (num_chunks > 0) for (int i = 0; i < EMBEDDING_DIM; i++) avg_embedding[i] /= (float)num_chunks;
        
        if (normalize) {
            float norm = 0.0f;
            for (int i = 0; i < EMBEDDING_DIM; i++) norm += avg_embedding[i] * avg_embedding[i];
            norm = sqrtf(norm + 1e-7f);
            for (int i = 0; i < EMBEDDING_DIM; i++) avg_embedding[i] /= norm;
        }
        return avg_embedding;
    }
    
    const size_t ctx_mem_size = 4 * 1024 * 1024; // Increased context size
    struct ggml_init_params params = { ctx_mem_size, NULL, true }; 
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return std::vector<float>(EMBEDDING_DIM, 0.0f);
    
    struct ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, n_mels, 1, 1);
    
    struct ggml_tensor * x = ggml_conv_2d(ctx, model.conv1_w, input, 1, 1, 1, 1, 1, 1);
    int n_ch = model.conv1_b->ne[0];
    struct ggml_tensor * b = ggml_reshape_4d(ctx, model.conv1_b, 1, 1, n_ch, 1);
    x = ggml_add(ctx, x, b);
    x = ggml_relu(ctx, x);
    // struct ggml_tensor * conv1_out = x; // Save for debug
    
    auto make_layer = [&](const std::vector<resnet_block> & layer, int stride, const char* name) {
        for (size_t i = 0; i < layer.size(); ++i) {
            struct ggml_tensor * residual = x;
            int s = (i == 0) ? stride : 1;
            int oc = layer[i].conv1_w->ne[3];
            x = ggml_conv_2d(ctx, layer[i].conv1_w, x, s, s, 1, 1, 1, 1);
            struct ggml_tensor * cb1 = ggml_reshape_4d(ctx, layer[i].conv1_b, 1, 1, oc, 1);
            x = ggml_add(ctx, x, cb1);
            x = ggml_relu(ctx, x);
            x = ggml_conv_2d(ctx, layer[i].conv2_w, x, 1, 1, 1, 1, 1, 1);
            struct ggml_tensor * cb2 = ggml_reshape_4d(ctx, layer[i].conv2_b, 1, 1, oc, 1);
            x = ggml_add(ctx, x, cb2);
            if (layer[i].ds_w) {
                residual = ggml_conv_2d(ctx, layer[i].ds_w, residual, s, s, 0, 0, 1, 1);
                struct ggml_tensor * dsb = ggml_reshape_4d(ctx, layer[i].ds_b, 1, 1, oc, 1);
                residual = ggml_add(ctx, residual, dsb);
            }
            x = ggml_add(ctx, x, residual);
            x = ggml_relu(ctx, x);
        }
        // struct ggml_tensor * l_out = x; // Capture for debug
        // printf("DEBUG build: Added %s\n", name);
    };
    
    make_layer(model.layer1, 1, "layer1");
    make_layer(model.layer2, 2, "layer2");
    make_layer(model.layer3, 2, "layer3");
    make_layer(model.layer4, 2, "layer4");
    
    int T_out = x->ne[0];
    int F_out = x->ne[1];
    int C_out = x->ne[2];
    
    x = ggml_permute(ctx, x, 2, 0, 1, 3);
    x = ggml_cont(ctx, x);
    int flat_dim = F_out * C_out;
    x = ggml_reshape_4d(ctx, x, flat_dim, T_out, 1, 1);
    struct ggml_tensor * mean_pool = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, 1, T_out, 1, T_out, 0, 0);
    struct ggml_tensor * x_sq = ggml_sqr(ctx, x);
    struct ggml_tensor * mean_sq_pool = ggml_pool_2d(ctx, x_sq, GGML_OP_POOL_AVG, 1, T_out, 1, T_out, 0, 0);
    struct ggml_tensor * mean_pow2 = ggml_sqr(ctx, mean_pool);
    struct ggml_tensor * var = ggml_sub(ctx, mean_sq_pool, mean_pow2);
    struct ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    var = ggml_add(ctx, var, eps);
    struct ggml_tensor * std_pool = ggml_sqrt(ctx, var);
    struct ggml_tensor * stats = ggml_concat(ctx, mean_pool, std_pool, 0);
    stats = ggml_reshape_1d(ctx, stats, 2 * flat_dim);
    struct ggml_tensor * emb_t = ggml_mul_mat(ctx, model.seg_1_w, stats);
    emb_t = ggml_add(ctx, emb_t, model.seg_1_b);
    
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, emb_t);

    // Use external backend if provided, otherwise create a local one
    ggml_backend_t backend_cpu = external_backend ? external_backend : ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_cpu);

    std::vector<float> in_data(T * n_mels);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < n_mels; f++) {
            in_data[t + f * T] = fbank[t * n_mels + f];
        }
    }
    ggml_backend_tensor_set(input, in_data.data(), 0, T * n_mels * sizeof(float));
    ggml_backend_graph_compute(backend_cpu, gf);
    
    std::vector<float> embedding(EMBEDDING_DIM);
    ggml_backend_tensor_get(emb_t, embedding.data(), 0, EMBEDDING_DIM * sizeof(float));
    
    if (normalize) {
        float norm = 0.0f;
        for (int i = 0; i < EMBEDDING_DIM; i++) norm += embedding[i] * embedding[i];
        norm = sqrtf(norm + 1e-7f);
        if (norm > 1e-8f) {
            for (int i = 0; i < EMBEDDING_DIM; i++) embedding[i] /= norm;
        }
    }
    
    ggml_backend_buffer_free(buf);
    if (!external_backend) {
        ggml_backend_free(backend_cpu);  // Only free if we created it
    }
    ggml_free(ctx);
    return embedding;
}

static std::vector<int> agglomerative_clustering(const std::vector<std::vector<float>> & embeddings, float threshold, int min_cluster_size) {
    const int n = (int)embeddings.size();
    if (n == 0) return {};
    if (n == 1) return {0};

    struct Cluster { std::vector<int> members; };
    std::vector<Cluster> clusters;
    clusters.reserve(n);
    for (int i = 0; i < n; ++i) clusters.push_back({{i}});
    
    auto cosine_distance = [](const std::vector<float> & a, const std::vector<float> & b) {
        float dot = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
        return 1.0f - dot;
    };

    auto average_linkage = [&](const Cluster & c1, const Cluster & c2) {
        float sum = 0.0f;
        int count = 0;
        for (int i : c1.members) {
            for (int j : c2.members) {
                sum += cosine_distance(embeddings[i], embeddings[j]);
                count++;
            }
        }
        return count > 0 ? (sum / count) : 0.0f;
    };

    while (true) {
        float best_dist = std::numeric_limits<float>::infinity();
        int best_i = -1, best_j = -1;
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (size_t j = i + 1; j < clusters.size(); ++j) {
                float dist = average_linkage(clusters[i], clusters[j]);
                if (dist < best_dist) { best_dist = dist; best_i = (int)i; best_j = (int)j; }
            }
        }
        if (best_i < 0 || best_j < 0 || best_dist > threshold) break;
        clusters[best_i].members.insert(clusters[best_i].members.end(), clusters[best_j].members.begin(), clusters[best_j].members.end());
        clusters.erase(clusters.begin() + best_j);
    }

    std::vector<int> labels(n, -1);
    for (size_t k = 0; k < clusters.size(); ++k) {
        for (int idx : clusters[k].members) labels[idx] = (int)k;
    }
    
    return labels;
}

// Convert powerset probs (7 classes) to multilabel probs (3 speakers)
std::vector<float> powerset_to_multilabel(const std::vector<float>& powerset_probs, int num_classes) {
    if (num_classes != 7) {
        if (num_classes == 3) return powerset_probs;
        return std::vector<float>(3, 0.0f);
    }
    std::vector<float> p(3, 0.0f);
    p[0] = powerset_probs[1] + powerset_probs[4] + powerset_probs[5];
    p[1] = powerset_probs[2] + powerset_probs[4] + powerset_probs[6];
    p[2] = powerset_probs[3] + powerset_probs[5] + powerset_probs[6];
    for(int i=0; i<3; i++) if(p[i] > 1.0f) p[i] = 1.0f;
    return p;
}

struct ChunkPrediction {
    int chunk_idx;
    double start_time;
    double end_time;
    std::vector<float> multilabel_probs; 
    int num_frames;
};

struct ChunkEmbedding {
    int chunk_idx;
    int local_speaker; 
    std::vector<float> embedding;
};

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <segmentation.gguf> <input.wav> [embedding.gguf]\n", argv[0]);
        return 1;
    }

    std::string seg_model_path = argv[1];
    std::string wav_path = argv[2];
    std::string emb_model_path = (argc > 3) ? argv[3] : "";
    bool use_embedding = !emb_model_path.empty();

    pyannote_segmentation_model seg_model;
    seg_model.ctx = nullptr; seg_model.buffer = nullptr;
    load_segmentation_model(seg_model_path, seg_model);
    
    pyannote_embedding_model emb_model;
    emb_model.ctx = nullptr; emb_model.buffer = nullptr;
    vbx_model vbx;
    bool use_vbx = false;
    
    if (use_embedding) {
        load_embedding_model(emb_model_path, emb_model);
        if (init_vbx_model_from_gguf(emb_model.ctx, vbx)) {
            use_vbx = true;
            printf("VBx model initialized from GGUF.\n");
        } else {
            printf("VBx model not found in GGUF. Falling back to AHC.\n");
        }
    }
    
    wav_file wav;
    if (!read_wav(wav_path, wav)) {
        fprintf(stderr, "Failed to read WAV file\n");
        return 1;
    }
    
    printf("Loaded WAV: %d Hz, %d channels, %zu samples\n", wav.sample_rate, wav.channels, wav.data.size());
    
    int window_samples = (int)(WINDOW_DURATION * SAMPLE_RATE);
    int step_samples = (int)(STEP_DURATION * SAMPLE_RATE);
    size_t total_samples = wav.data.size();
    
    std::vector<ChunkPrediction> predictions;
    std::vector<ChunkEmbedding> embeddings;
    
    size_t buf_size = 1024 * 1024 * 64; 
    struct ggml_init_params params = { buf_size, NULL, false };
    struct ggml_context * ctx0 = ggml_init(params);
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    
    // For Metal to work, model weights would need to be loaded on Metal buffer
    // This requires modifying load_embedding_model() - complex change for later
    ggml_backend_t emb_backend = use_embedding ? ggml_backend_cpu_init() : nullptr;
    
    int chunk_idx = 0;
    double model_frame_duration = 270.0 / 16000.0;
    
    for (size_t pos = 0; pos < total_samples; pos += step_samples, chunk_idx++) {
        std::vector<float> chunk_wav(window_samples, 0.0f);
        size_t available = std::min((size_t)window_samples, total_samples - pos);
        if (available < (size_t)(0.5 * window_samples)) break; 
        
        memcpy(chunk_wav.data(), &wav.data[pos], available * sizeof(float));
        
        ggml_free(ctx0);
        ctx0 = ggml_init(params);
        
        struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, window_samples);
        memcpy(input->data, chunk_wav.data(), window_samples * sizeof(float));
        
        struct ggml_tensor * x = ggml_norm(ctx0, input, 1e-5f);
        x = ggml_mul(ctx0, x, seg_model.wav_norm1d_w);
        x = ggml_add(ctx0, x, seg_model.wav_norm1d_b);
        x = ggml_reshape_3d(ctx0, x, window_samples, 1, 1);
        x = ggml_conv_1d(ctx0, seg_model.sinc_conv_w[0], x, 10, 0, 1);
        x = ggml_abs(ctx0, x);
        x = ggml_pool_1d(ctx0, x, GGML_OP_POOL_MAX, 3, 3, 0);
        x = ggml_norm(ctx0, x, 1e-5f);
        struct ggml_tensor * w0 = ggml_reshape_2d(ctx0, seg_model.sinc_norm_w[0], 1, 80);
        struct ggml_tensor * b0 = ggml_reshape_2d(ctx0, seg_model.sinc_norm_b[0], 1, 80);
        x = ggml_mul(ctx0, x, w0);
        x = ggml_add(ctx0, x, b0);
        x = ggml_leaky_relu(ctx0, x, 0.01f, true);
        
        x = ggml_conv_1d(ctx0, seg_model.sinc_conv_w[1], x, 1, 0, 1);
        struct ggml_tensor * cb1 = ggml_reshape_2d(ctx0, seg_model.sinc_conv_b[1], 1, 60);
        x = ggml_add(ctx0, x, cb1);
        x = ggml_pool_1d(ctx0, x, GGML_OP_POOL_MAX, 3, 3, 0);
        x = ggml_norm(ctx0, x, 1e-5f);
        struct ggml_tensor * w1 = ggml_reshape_2d(ctx0, seg_model.sinc_norm_w[1], 1, 60);
        struct ggml_tensor * b1 = ggml_reshape_2d(ctx0, seg_model.sinc_norm_b[1], 1, 60);
        x = ggml_mul(ctx0, x, w1);
        x = ggml_add(ctx0, x, b1);
        x = ggml_leaky_relu(ctx0, x, 0.01f, true);
        
        x = ggml_conv_1d(ctx0, seg_model.sinc_conv_w[2], x, 1, 0, 1);
        struct ggml_tensor * cb2 = ggml_reshape_2d(ctx0, seg_model.sinc_conv_b[2], 1, 60);
        x = ggml_add(ctx0, x, cb2);
        x = ggml_pool_1d(ctx0, x, GGML_OP_POOL_MAX, 3, 3, 0);
        x = ggml_norm(ctx0, x, 1e-5f);
        struct ggml_tensor * w2 = ggml_reshape_2d(ctx0, seg_model.sinc_norm_w[2], 1, 60);
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx0, seg_model.sinc_norm_b[2], 1, 60);
        x = ggml_mul(ctx0, x, w2);
        x = ggml_add(ctx0, x, b2);
        x = ggml_leaky_relu(ctx0, x, 0.01f, true);
        
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, x);
        ggml_backend_graph_compute(backend_cpu, gf);
        
        int T = x->ne[0];
        int F = x->ne[1];
        std::vector<float> sinc_out(T * F);
        float * x_ptr = (float*)x->data;
        for (int t=0; t<T; t++) for (int f=0; f<F; f++) sinc_out[t * F + f] = x_ptr[f * T + t];
        
        std::vector<float> h_state = sinc_out;
        int seq_len = T;
        for (int l=0; l<4; l++) {
            int input_dim = (l==0) ? 60 : 256;
            int hidden_dim = 128;
            std::vector<float> output_fwd(seq_len * hidden_dim);
            std::vector<float> output_rev(seq_len * hidden_dim);
            
            // Get weight tensors for ggml batched matmul
            struct ggml_tensor * w_ih_fwd_t = seg_model.lstm_fwd[l].w_ih;  // (4*hidden, input)
            struct ggml_tensor * w_ih_rev_t = seg_model.lstm_rev[l].w_ih;
            
            // Pre-dequantized biases for fast access
            const float * b_ih_fwd = seg_model.lstm_fwd[l].b_ih_f32.data();
            const float * b_hh_fwd = seg_model.lstm_fwd[l].b_hh_f32.data();
            const float * b_ih_rev = seg_model.lstm_rev[l].b_ih_f32.data();
            const float * b_hh_rev = seg_model.lstm_rev[l].b_hh_f32.data();
            
            // Pre-dequantized W_hh for per-timestep hidden state update
            const float * w_hh_fwd = seg_model.lstm_fwd[l].w_hh_f32.data();
            const float * w_hh_rev = seg_model.lstm_rev[l].w_hh_f32.data();
            
            // === BATCH INPUT PROJECTION using ggml ===
            // Compute gates_x = W_ih @ x_all for all timesteps at once
            // x_all: (seq_len, input_dim) -> need as tensor (input_dim, seq_len) for matmul
            ggml_free(ctx0);
            struct ggml_init_params lstm_params = { params.mem_size, NULL, true };  // no_alloc = true for backend alloc
            ctx0 = ggml_init(lstm_params);
            struct ggml_tensor * x_all = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, input_dim, seq_len);
            struct ggml_tensor * gates_x_fwd = ggml_mul_mat(ctx0, w_ih_fwd_t, x_all);  // (4*hidden, seq_len)
            struct ggml_tensor * gates_x_rev = ggml_mul_mat(ctx0, w_ih_rev_t, x_all);
            
            struct ggml_cgraph * gf = ggml_new_graph(ctx0);
            ggml_build_forward_expand(gf, gates_x_fwd);
            ggml_build_forward_expand(gf, gates_x_rev);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, backend_cpu);
            
            // Set input data - ggml uses column-major: x_all(input_dim, seq_len) 
            // data[i + t * input_dim] = h_state[t * input_dim + i]
            // So we can just copy directly since storage matches!
            ggml_backend_tensor_set(x_all, h_state.data(), 0, seq_len * input_dim * sizeof(float));
            ggml_backend_graph_compute(backend_cpu, gf);
            
            // Get batched gate projections
            // Output gates_x_fwd is (4*hidden_dim, seq_len) in column-major
            // data[g + t * (4*hidden_dim)] gives gate g at timestep t
            std::vector<float> gates_fwd_all(4 * hidden_dim * seq_len);
            std::vector<float> gates_rev_all(4 * hidden_dim * seq_len);
            ggml_backend_tensor_get(gates_x_fwd, gates_fwd_all.data(), 0, gates_fwd_all.size() * sizeof(float));
            ggml_backend_tensor_get(gates_x_rev, gates_rev_all.data(), 0, gates_rev_all.size() * sizeof(float));
            ggml_backend_buffer_free(buf);
            
            // === Forward LSTM with precomputed input projections ===
            // Column-major: gates_fwd_all[g + t * (4*hidden_dim)]
            std::vector<float> h(hidden_dim, 0.0f), c(hidden_dim, 0.0f);
            std::vector<float> gates(4 * hidden_dim);
            for (int t = 0; t < seq_len; t++) {
                // gates = gates_x[:, t] + bias_ih + bias_hh + W_hh @ h
                for (int g = 0; g < 4 * hidden_dim; g++) {
                    gates[g] = gates_fwd_all[g + t * (4 * hidden_dim)] + b_ih_fwd[g] + b_hh_fwd[g];
                }
                // W_hh @ h
                for (int g = 0; g < 4 * hidden_dim; g++) {
                    const float * w_row = w_hh_fwd + g * hidden_dim;
                    float sum = 0.0f;
                    for (int j = 0; j < hidden_dim; j++) sum += w_row[j] * h[j];
                    gates[g] += sum;
                }
                // Apply activations
                for (int i = 0; i < hidden_dim; i++) {
                    float in_gate = 1.0f / (1.0f + expf(-gates[i]));
                    float forget_gate = 1.0f / (1.0f + expf(-gates[hidden_dim + i]));
                    float cell_gate = tanhf(gates[2 * hidden_dim + i]);
                    float out_gate = 1.0f / (1.0f + expf(-gates[3 * hidden_dim + i]));
                    c[i] = forget_gate * c[i] + in_gate * cell_gate;
                    h[i] = out_gate * tanhf(c[i]);
                    output_fwd[t * hidden_dim + i] = h[i];
                }
            }
            
            // === Reverse LSTM with precomputed input projections ===
            std::fill(h.begin(), h.end(), 0.0f);
            std::fill(c.begin(), c.end(), 0.0f);
            for (int t = seq_len - 1; t >= 0; t--) {
                for (int g = 0; g < 4 * hidden_dim; g++) {
                    gates[g] = gates_rev_all[g + t * (4 * hidden_dim)] + b_ih_rev[g] + b_hh_rev[g];
                }
                for (int g = 0; g < 4 * hidden_dim; g++) {
                    const float * w_row = w_hh_rev + g * hidden_dim;
                    float sum = 0.0f;
                    for (int j = 0; j < hidden_dim; j++) sum += w_row[j] * h[j];
                    gates[g] += sum;
                }
                for (int i = 0; i < hidden_dim; i++) {
                    float in_gate = 1.0f / (1.0f + expf(-gates[i]));
                    float forget_gate = 1.0f / (1.0f + expf(-gates[hidden_dim + i]));
                    float cell_gate = tanhf(gates[2 * hidden_dim + i]);
                    float out_gate = 1.0f / (1.0f + expf(-gates[3 * hidden_dim + i]));
                    c[i] = forget_gate * c[i] + in_gate * cell_gate;
                    h[i] = out_gate * tanhf(c[i]);
                    output_rev[t * hidden_dim + i] = h[i];
                }
            }
            
            // Concatenate forward and reverse outputs
            std::vector<float> next_input(seq_len * 2 * hidden_dim);
            for (int t=0; t<seq_len; t++) {
                memcpy(&next_input[t*2*hidden_dim], &output_fwd[t*hidden_dim], hidden_dim*sizeof(float));
                memcpy(&next_input[t*2*hidden_dim + hidden_dim], &output_rev[t*hidden_dim], hidden_dim*sizeof(float));
            }
            h_state = next_input;
        }
        
        ggml_free(ctx0); ctx0 = ggml_init(params);
        struct ggml_tensor * lstm_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 256, seq_len);
        memcpy(lstm_out->data, h_state.data(), h_state.size() * sizeof(float));
        struct ggml_tensor * lin0 = ggml_mul_mat(ctx0, seg_model.linear_w[0], lstm_out); 
        struct ggml_tensor * lb0 = ggml_reshape_2d(ctx0, seg_model.linear_b[0], 128, 1);
        lin0 = ggml_add(ctx0, lin0, lb0);
        lin0 = ggml_leaky_relu(ctx0, lin0, 0.01f, true);
        struct ggml_tensor * lin1 = ggml_mul_mat(ctx0, seg_model.linear_w[1], lin0);
        struct ggml_tensor * lb1 = ggml_reshape_2d(ctx0, seg_model.linear_b[1], 128, 1);
        lin1 = ggml_add(ctx0, lin1, lb1);
        lin1 = ggml_leaky_relu(ctx0, lin1, 0.01f, true);
        struct ggml_tensor * logits = ggml_mul_mat(ctx0, seg_model.classifier_w, lin1);
        struct ggml_tensor * cb = ggml_reshape_2d(ctx0, seg_model.classifier_b, seg_model.classifier_w->ne[1], 1); 
        logits = ggml_add(ctx0, logits, cb);
        struct ggml_tensor * probs = ggml_soft_max(ctx0, logits);
        
        gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, probs);
        ggml_backend_graph_compute(backend_cpu, gf);
        
        int num_frames = probs->ne[1];
        int num_classes = probs->ne[0];
        
        std::vector<float> powerset_data(num_frames * num_classes);
        memcpy(powerset_data.data(), probs->data, powerset_data.size() * sizeof(float));
        
        ChunkPrediction pred;
        pred.chunk_idx = chunk_idx;
        pred.start_time = pos / (double)SAMPLE_RATE;
        pred.end_time = pred.start_time + WINDOW_DURATION;
        pred.num_frames = num_frames;
        pred.multilabel_probs.resize(num_frames * 3);
        
        for (int t=0; t<num_frames; t++) {
            std::vector<float> p_frame(num_classes);
            for (int c=0; c<num_classes; c++) p_frame[c] = powerset_data[t*num_classes + c];
            std::vector<float> multi = powerset_to_multilabel(p_frame, num_classes);
            for (int s=0; s<3; s++) pred.multilabel_probs[t*3 + s] = multi[s];
        }
        predictions.push_back(pred);
        
        if (use_embedding) {
            float embed_threshold = 0.5f;
            int frames_per_window = num_frames;
            int frame_samples = window_samples / frames_per_window; 
            
            for (int s=0; s<3; s++) {
                std::vector<std::pair<int, int>> active_segments;
                int start_f = -1;
                for (int t=0; t<num_frames; t++) {
                    if (pred.multilabel_probs[t*3 + s] > embed_threshold) {
                        if (start_f == -1) start_f = t;
                    } else {
                        if (start_f != -1) {
                            active_segments.push_back({start_f, t});
                            start_f = -1;
                        }
                    }
                }
                if (start_f != -1) active_segments.push_back({start_f, num_frames});
                
                for (auto & seg : active_segments) {
                    int len_frames = seg.second - seg.first;
                    if (len_frames < 6) continue;
                    
                    int s0 = seg.first * frame_samples;
                    int s1 = seg.second * frame_samples;
                    if (s1 > (int)chunk_wav.size()) s1 = chunk_wav.size();
                    
                    std::vector<float> sub_wav(s1 - s0);
                    for(int i=0; i<s1-s0; i++) sub_wav[i] = chunk_wav[s0+i];
                    
                    if (sub_wav.size() < (size_t)(0.2 * SAMPLE_RATE)) continue; 
                    
                    std::vector<float> fbank = compute_fbank(sub_wav, SAMPLE_RATE, 80);
                    int T_emb = fbank.size() / 80;
                    if (T_emb < 5) continue;
                    
                    std::vector<float> emb_vec = get_embedding(emb_model, fbank, T_emb, 80, !use_vbx, emb_backend);
                    
                    bool ok = true;
                    for (float v : emb_vec) if (std::isnan(v) || std::isinf(v)) ok = false;
                    if (ok) {
                        embeddings.push_back({chunk_idx, s, emb_vec});
                    }
                }
            }
        }
    }
    
    std::vector<int> global_labels;
    if (use_embedding && !embeddings.empty()) {
        printf("Clustering %zu embeddings...\n", embeddings.size());
        
        std::vector<std::vector<float>> all_embeddings;
        for (const auto & e : embeddings) all_embeddings.push_back(e.embedding);
        
        if (use_vbx) {
            // Use single transformation (verified correct) for each embedding
            std::vector<std::vector<float>> vbx_emb;
            for (const auto & e : all_embeddings) {
                vbx_emb.push_back(vbx_transform_embedding(vbx, e, false));
            }
            vbx_params p; 
            p.threshold = 0.6f; 
            p.Fa = 0.07f; 
            p.Fb = 0.8f;
            global_labels = cluster_vbx(vbx, vbx_emb, all_embeddings, p);
        } else {
            global_labels = agglomerative_clustering(all_embeddings, 0.6f, 1);
        }
    } else {
        global_labels.resize(embeddings.size(), 0);
    }
    
    int num_global_speakers = 0;
    for(int l : global_labels) if(l >= num_global_speakers) num_global_speakers = l + 1;
    
    if (num_global_speakers == 0 && !embeddings.empty()) num_global_speakers = 1; 
    if (!use_embedding) num_global_speakers = 3; 
    
    printf("Reconstructing timeline for %d speakers...\n", num_global_speakers);
    
    double total_dur = total_samples / (double)SAMPLE_RATE;
    int grid_frames = (int)ceil(total_dur / model_frame_duration);
    
    std::vector<std::vector<float>> global_acc(num_global_speakers, std::vector<float>(grid_frames, 0.0f));
    std::vector<std::vector<float>> global_count(num_global_speakers, std::vector<float>(grid_frames, 0.0f));
    
    std::map<std::pair<int,int>, int> assignment_map;
    for (size_t i=0; i<embeddings.size(); i++) {
        assignment_map[{embeddings[i].chunk_idx, embeddings[i].local_speaker}] = global_labels[i];
    }
    
    for (const auto & pred : predictions) {
        int start_grid_frame = (int)round(pred.start_time / model_frame_duration);
        for (int t=0; t<pred.num_frames; t++) {
            int g_t = start_grid_frame + t;
            if (g_t >= grid_frames) break;
            
            for (int s=0; s<3; s++) {
                float prob = pred.multilabel_probs[t*3 + s];
                int global_id = -1;
                if (use_embedding) {
                    if (assignment_map.count({pred.chunk_idx, s})) {
                        global_id = assignment_map[{pred.chunk_idx, s}];
                    }
                } else {
                    global_id = s; 
                }
                
                if (global_id != -1 && global_id < num_global_speakers) {
                    global_acc[global_id][g_t] += prob;
                    global_count[global_id][g_t] += 1.0f;
                }
            }
        }
    }
    
    std::vector<std::vector<std::pair<double, double>>> final_segments(num_global_speakers);
    for (int s=0; s<num_global_speakers; s++) {
        bool active = false;
        double start = 0.0;
        for (int t=0; t<grid_frames; t++) {
            float val = 0.0f;
            if (global_count[s][t] > 0) val = global_acc[s][t] / global_count[s][t];
            
            if (val > 0.5f) { 
                if (!active) {
                    active = true;
                    start = t * model_frame_duration;
                }
            } else {
                if (active) {
                    active = false;
                    final_segments[s].push_back({start, t * model_frame_duration});
                }
            }
        }
        if (active) final_segments[s].push_back({start, grid_frames * model_frame_duration});
    }
    
    struct FinalSeg {
        double start;
        double end;
        int speaker;
    };
    std::vector<FinalSeg> sorted_segments;
    for (int s=0; s<num_global_speakers; s++) {
        for (auto & seg : final_segments[s]) {
            if (seg.second - seg.first > 0.0) {
                 sorted_segments.push_back({seg.first, seg.second, s});
            }
        }
    }
    std::sort(sorted_segments.begin(), sorted_segments.end(), [](const FinalSeg & a, const FinalSeg & b){
        return a.start < b.start;
    });
    
    if (!sorted_segments.empty()) {
        std::vector<FinalSeg> merged;
        merged.push_back(sorted_segments[0]);
        for(size_t i=1; i<sorted_segments.size(); i++) {
            auto & last = merged.back();
            auto & curr = sorted_segments[i];
            if (last.speaker == curr.speaker && (curr.start - last.end) < 0.1) {
                last.end = std::max(last.end, curr.end);
            } else {
                merged.push_back(curr);
            }
        }
        sorted_segments = merged;
    }

    printf("---------------------------------\n");
    std::string filename = wav_path.substr(wav_path.find_last_of("/\\") + 1);
    printf("%s\n", filename.c_str());
    for (const auto & seg : sorted_segments) {
        printf("start=%.1fs stop=%.1fs speaker_SPEAKER_%02d\n", seg.start, seg.end, seg.speaker);
    }
    printf("---------------------------------\n");

    ggml_free(ctx0);
    ggml_backend_free(backend_cpu);
    if (emb_backend) ggml_backend_free(emb_backend);
    
    return 0;
}
