#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpu.h"

#include "common.h"
#include "common_audio.h"

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
#include <deque>
#include <thread>

static const int EMBEDDING_DIM = 256;
static const int NUM_LOCAL_SPEAKERS = 3;
static const int NUM_POWERSET_CLASSES = 7;
static int g_n_threads = 4;

// Global backend for reuse
static ggml_backend_t g_backend_cpu = nullptr;

static void init_global_backend() {
    if (!g_backend_cpu) {
        g_backend_cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(g_backend_cpu, g_n_threads);
    }
}

static void free_global_backend() {
    if (g_backend_cpu) {
        ggml_backend_free(g_backend_cpu);
        g_backend_cpu = nullptr;
    }
}

static std::vector<float> dequantize_tensor(struct ggml_tensor * t) {
    std::vector<float> res(ggml_nelements(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_cpu_fp16_to_fp32((const ggml_fp16_t *)t->data, res.data(), res.size());
    } else {
        memcpy(res.data(), t->data, res.size() * sizeof(float));
    }
    return res;
}

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

struct lstm_layer_weights {
    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
};

struct segmentation_weights_cache {
    std::vector<lstm_layer_weights> fwd;
    std::vector<lstm_layer_weights> rev;
};

static float cosine_distance(const std::vector<float> & a, const std::vector<float> & b) {
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return 1.0f - dot;
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
            (void)ret;
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
}

std::vector<float> get_embedding(pyannote_embedding_model & model, const std::vector<float> & fbank, int T, int n_mels) {
    const size_t ctx_mem_size = 16 * 1024 * 1024;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to allocate ggml context for embedding\n");
        return std::vector<float>(EMBEDDING_DIM, 0.0f);
    }

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, n_mels, 1, 1);

    struct ggml_tensor * x = ggml_conv_2d(ctx, model.conv1_w, input, 1, 1, 1, 1, 1, 1);
    int n_ch = model.conv1_b->ne[0];
    struct ggml_tensor * b = ggml_reshape_4d(ctx, model.conv1_b, 1, 1, n_ch, 1);
    x = ggml_add(ctx, x, b);
    x = ggml_relu(ctx, x);

    auto make_layer = [&](const std::vector<resnet_block> & layer, int stride) {
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
    };

    make_layer(model.layer1, 1);
    make_layer(model.layer2, 2);
    make_layer(model.layer3, 2);
    make_layer(model.layer4, 2);

    int T_out = x->ne[0];
    int F_out = x->ne[1];
    int C_out = x->ne[2];

    // TSTP pooling: permute to (F, C, T, N) for correct flattening
    // 原始: ne[0]=T, ne[1]=F, ne[2]=C, ne[3]=N
    // 目标: ne[0]=F, ne[1]=C, ne[2]=T, ne[3]=N
    x = ggml_permute(ctx, x, 2, 0, 1, 3);
    x = ggml_cont(ctx, x);
    int flat_dim = F_out * C_out;
    x = ggml_reshape_4d(ctx, x, flat_dim, T_out, 1, 1);

    struct ggml_tensor * mean_pool = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, 1, T_out, 1, T_out, 0, 0);
    struct ggml_tensor * x_sq = ggml_sqr(ctx, x);
    struct ggml_tensor * mean_sq_pool = ggml_pool_2d(ctx, x_sq, GGML_OP_POOL_AVG, 1, T_out, 1, T_out, 0, 0);
    struct ggml_tensor * mean_pow2 = ggml_sqr(ctx, mean_pool);
    struct ggml_tensor * var = ggml_sub(ctx, mean_sq_pool, mean_pow2);
    struct ggml_tensor * eps = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1);
    var = ggml_add(ctx, var, eps);
    struct ggml_tensor * std_pool = ggml_sqrt(ctx, var);
    struct ggml_tensor * stats = ggml_concat(ctx, mean_pool, std_pool, 0);
    stats = ggml_reshape_1d(ctx, stats, 2 * flat_dim);

    struct ggml_tensor * emb = ggml_mul_mat(ctx, model.seg_1_w, stats);
    emb = ggml_add(ctx, emb, model.seg_1_b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, emb);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, g_backend_cpu);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate backend buffer for embedding\n");
        ggml_free(ctx);
        return std::vector<float>(EMBEDDING_DIM, 0.0f);
    }

    float * in_data = (float *)input->data;
    for (int f = 0; f < n_mels; f++) {
        for (int t = 0; t < T; t++) {
            in_data[t + f * T] = fbank[t * n_mels + f];
        }
    }
    ggml_set_f32(eps, 1e-7f);

    ggml_backend_graph_compute(g_backend_cpu, gf);

    std::vector<float> embedding(EMBEDDING_DIM);
    memcpy(embedding.data(), emb->data, EMBEDDING_DIM * sizeof(float));
    float norm = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm + 1e-7f);
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        embedding[i] /= norm;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return embedding;
}

static std::vector<std::vector<float>> powerset_to_multilabel(
    const std::vector<float> & probs,
    int num_frames
) {
    static const int mapping[NUM_POWERSET_CLASSES][NUM_LOCAL_SPEAKERS] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
    };

    std::vector<std::vector<float>> multi(num_frames, std::vector<float>(NUM_LOCAL_SPEAKERS, 0.0f));
    for (int t = 0; t < num_frames; ++t) {
        int max_idx = 0;
        float max_val = probs[t * NUM_POWERSET_CLASSES];
        for (int p = 1; p < NUM_POWERSET_CLASSES; ++p) {
            float v = probs[t * NUM_POWERSET_CLASSES + p];
            if (v > max_val) {
                max_val = v;
                max_idx = p;
            }
        }
        for (int s = 0; s < NUM_LOCAL_SPEAKERS; ++s) {
            if (mapping[max_idx][s]) {
                multi[t][s] = 1.0f;
            }
        }
    }
    return multi;
}

// Run SincNet part on GPU-friendly ggml graph, then do LSTM on CPU with cached weights
static std::vector<std::vector<float>> run_segmentation(
    pyannote_segmentation_model & seg_model,
    const segmentation_weights_cache & cache,
    const std::vector<float> & wav,
    int sample_rate,
    int & num_frames
) {
    size_t num_samples = wav.size();
    size_t buf_size = 256 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_samples);
    memcpy(input->data, wav.data(), num_samples * sizeof(float));

    struct ggml_tensor * x = ggml_norm(ctx0, input, 1e-5f);
    x = ggml_mul(ctx0, x, seg_model.wav_norm1d_w);
    x = ggml_add(ctx0, x, seg_model.wav_norm1d_b);

    x = ggml_reshape_3d(ctx0, x, num_samples, 1, 1);
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
    ggml_graph_compute_with_ctx(ctx0, gf, g_n_threads);

    int T = x->ne[0];
    int F = x->ne[1];
    std::vector<float> sinc_out(T * F);
    float * x_ptr = (float*)x->data;
    for (int t=0; t<T; t++) {
        for (int f=0; f<F; f++) {
            sinc_out[t * F + f] = x_ptr[f * T + t];
        }
    }

    std::vector<float> h_state = sinc_out;
    int seq_len = T;

    for (int l=0; l<4; l++) {
        int input_dim = (l==0) ? 60 : 256;
        int hidden_dim = 128;
        std::vector<float> output_fwd(seq_len * hidden_dim);
        std::vector<float> output_rev(seq_len * hidden_dim);
        const auto & w_ih_fwd = cache.fwd[l].w_ih;
        const auto & w_hh_fwd = cache.fwd[l].w_hh;
        const auto & b_ih_fwd = cache.fwd[l].b_ih;
        const auto & b_hh_fwd = cache.fwd[l].b_hh;
        const auto & w_ih_rev = cache.rev[l].w_ih;
        const auto & w_hh_rev = cache.rev[l].w_hh;
        const auto & b_ih_rev = cache.rev[l].b_ih;
        const auto & b_hh_rev = cache.rev[l].b_hh;

        std::vector<float> h(hidden_dim, 0.0f);
        std::vector<float> c(hidden_dim, 0.0f);
        for (int t=0; t<seq_len; t++) {
            float * x_t = &h_state[t * input_dim];
            std::vector<float> gates(4 * hidden_dim);
            for (int i=0; i<4*hidden_dim; i++) {
                float sum = 0;
                for (int j=0; j<input_dim; j++) sum += w_ih_fwd[i*input_dim + j] * x_t[j];
                sum += b_ih_fwd[i];
                for (int j=0; j<hidden_dim; j++) sum += w_hh_fwd[i*hidden_dim + j] * h[j];
                sum += b_hh_fwd[i];
                gates[i] = sum;
            }
            for (int i=0; i<hidden_dim; i++) {
                float in_gate = 1.0f / (1.0f + expf(-gates[0*hidden_dim + i]));
                float forget_gate = 1.0f / (1.0f + expf(-gates[1*hidden_dim + i]));
                float cell_gate = tanhf(gates[2*hidden_dim + i]);
                float out_gate = 1.0f / (1.0f + expf(-gates[3*hidden_dim + i]));
                c[i] = forget_gate * c[i] + in_gate * cell_gate;
                h[i] = out_gate * tanhf(c[i]);
                output_fwd[t*hidden_dim + i] = h[i];
            }
        }

        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        for (int t=seq_len-1; t>=0; t--) {
            float * x_t = &h_state[t * input_dim];
            std::vector<float> gates(4 * hidden_dim);
            for (int i=0; i<4*hidden_dim; i++) {
                float sum = 0;
                for (int j=0; j<input_dim; j++) sum += w_ih_rev[i*input_dim + j] * x_t[j];
                sum += b_ih_rev[i];
                for (int j=0; j<hidden_dim; j++) sum += w_hh_rev[i*hidden_dim + j] * h[j];
                sum += b_hh_rev[i];
                gates[i] = sum;
            }
            for (int i=0; i<hidden_dim; i++) {
                float in_gate = 1.0f / (1.0f + expf(-gates[0*hidden_dim + i]));
                float forget_gate = 1.0f / (1.0f + expf(-gates[1*hidden_dim + i]));
                float cell_gate = tanhf(gates[2*hidden_dim + i]);
                float out_gate = 1.0f / (1.0f + expf(-gates[3*hidden_dim + i]));
                c[i] = forget_gate * c[i] + in_gate * cell_gate;
                h[i] = out_gate * tanhf(c[i]);
                output_rev[t*hidden_dim + i] = h[i];
            }
        }
        std::vector<float> next_input(seq_len * 2 * hidden_dim);
        for (int t=0; t<seq_len; t++) {
            memcpy(&next_input[t*2*hidden_dim], &output_fwd[t*hidden_dim], hidden_dim*sizeof(float));
            memcpy(&next_input[t*2*hidden_dim + hidden_dim], &output_rev[t*hidden_dim], hidden_dim*sizeof(float));
        }
        h_state = next_input;
    }

    int num_classes = seg_model.classifier_w->ne[1];
    ggml_free(ctx0);
    
    // Linear layers using ggml
    size_t linear_buf_size = 16 * 1024 * 1024;
    struct ggml_init_params linear_params = {
        /*.mem_size   =*/ linear_buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    ctx0 = ggml_init(linear_params);
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
    struct ggml_tensor * cb = ggml_reshape_2d(ctx0, seg_model.classifier_b, num_classes, 1);
    logits = ggml_add(ctx0, logits, cb);

    struct ggml_tensor * probs = ggml_soft_max(ctx0, logits);

    gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, probs);
    ggml_graph_compute_with_ctx(ctx0, gf, g_n_threads);

    num_frames = probs->ne[1];
    std::vector<float> result(num_classes * num_frames);
    memcpy(result.data(), probs->data, result.size() * sizeof(float));

    ggml_free(ctx0);

    return powerset_to_multilabel(result, num_frames);
}

struct OnlineSpeakerClustering {
    float tau_active;
    float rho_update;
    float delta_new;
    int max_speakers;
    std::vector<std::vector<float>> centers;
    std::vector<bool> active_centers;

    OnlineSpeakerClustering(float tau, float rho, float delta, int max_spk)
        : tau_active(tau), rho_update(rho), delta_new(delta), max_speakers(max_spk) {
        centers.clear();
        active_centers.assign(max_speakers, false);
    }

    void init_centers(int dim) {
        centers.assign(max_speakers, std::vector<float>(dim, 0.0f));
        active_centers.assign(max_speakers, false);
    }

    int get_next_center() {
        for (int i = 0; i < max_speakers; ++i) {
            if (!active_centers[i]) return i;
        }
        return -1;
    }

    std::vector<int> identify(
        const std::vector<std::vector<float>> & segmentation,
        const std::vector<std::vector<float>> & embeddings
    ) {
        const int num_frames = (int)segmentation.size();
        const int num_local = (int)segmentation[0].size();
        std::vector<int> mapping(num_local, -1);

        std::vector<int> active_speakers;
        std::vector<int> long_speakers;
        for (int s = 0; s < num_local; ++s) {
            float max_val = 0.0f;
            float mean_val = 0.0f;
            for (int t = 0; t < num_frames; ++t) {
                max_val = std::max(max_val, segmentation[t][s]);
                mean_val += segmentation[t][s];
            }
            mean_val /= std::max(1, num_frames);
            if (max_val >= tau_active) active_speakers.push_back(s);
            if (mean_val >= rho_update) long_speakers.push_back(s);
        }

        if (centers.empty()) {
            init_centers((int)embeddings[0].size());
            for (int s : active_speakers) {
                bool has_nan = false;
                for (float v : embeddings[s]) {
                    if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; }
                }
                if (has_nan) continue;
                int c = get_next_center();
                if (c < 0) break;
                centers[c] = embeddings[s];
                active_centers[c] = true;
                mapping[s] = c;
            }
            return mapping;
        }

        struct Pair { int s; int c; float dist; };
        std::vector<Pair> pairs;
        for (int s : active_speakers) {
            bool has_nan = false;
            for (float v : embeddings[s]) {
                if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; }
            }
            if (has_nan) continue;
            for (int c = 0; c < max_speakers; ++c) {
                if (!active_centers[c]) continue;
                pairs.push_back({s, c, cosine_distance(embeddings[s], centers[c])});
            }
        }

        std::sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) {
            return a.dist < b.dist;
        });

        std::vector<bool> used_s(num_local, false);
        std::vector<bool> used_c(max_speakers, false);

        for (const auto & p : pairs) {
            if (p.dist > delta_new) break;
            if (used_s[p.s] || used_c[p.c]) continue;
            mapping[p.s] = p.c;
            used_s[p.s] = true;
            used_c[p.c] = true;
        }

        // Assign remaining active speakers
        for (int s : active_speakers) {
            if (mapping[s] >= 0) continue;
            bool has_nan = false;
            for (float v : embeddings[s]) {
                if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; }
            }
            if (has_nan) continue;

            bool is_long = std::find(long_speakers.begin(), long_speakers.end(), s) != long_speakers.end();
            if (is_long) {
                int c = get_next_center();
                if (c >= 0) {
                    centers[c] = embeddings[s];
                    active_centers[c] = true;
                    mapping[s] = c;
                    continue;
                }
            }

            float best_dist = std::numeric_limits<float>::infinity();
            int best_center = -1;
            for (int c = 0; c < max_speakers; ++c) {
                if (!active_centers[c]) continue;
                float dist = cosine_distance(embeddings[s], centers[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_center = c;
                }
            }
            if (best_center >= 0) {
                mapping[s] = best_center;
            }
        }

        // Update centers with long speakers that were assigned
        for (int s : active_speakers) {
            if (mapping[s] < 0) continue;
            if (std::find(long_speakers.begin(), long_speakers.end(), s) == long_speakers.end()) continue;
            int c = mapping[s];
            for (size_t d = 0; d < centers[c].size(); ++d) {
                centers[c][d] += embeddings[s][d];
            }
        }

        return mapping;
    }
};

static std::vector<std::vector<float>> permute_segmentation(
    const std::vector<std::vector<float>> & segmentation,
    const std::vector<int> & mapping,
    int max_speakers,
    bool fallback_local
) {
    const int T = (int)segmentation.size();
    std::vector<std::vector<float>> out(T, std::vector<float>(max_speakers, 0.0f));
    for (size_t s = 0; s < mapping.size(); ++s) {
        int g = mapping[s];
        if (g < 0 && fallback_local) {
            if ((int)s < max_speakers) g = (int)s;
        }
        if (g < 0) continue;
        for (int t = 0; t < T; ++t) {
            out[t][g] = std::max(out[t][g], segmentation[t][s]);
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <segmentation.gguf> <input.wav> <embedding.gguf> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --duration D     Chunk duration in seconds (default: 5.0)\n");
        fprintf(stderr, "  --step S         Step size in seconds (default: 2.5 for speed)\n");
        fprintf(stderr, "  --latency L      Latency in seconds (default: step)\n");
        fprintf(stderr, "  --tau_active T   Active speaker threshold (default: 0.6)\n");
        fprintf(stderr, "  --tau_embed T    Embedding threshold (default: 0.3)\n");
        fprintf(stderr, "  --rho_update R   Center update threshold (default: 0.3)\n");
        fprintf(stderr, "  --delta_new D    New speaker threshold (default: 1.0)\n");
        fprintf(stderr, "  --max_speakers N Maximum speakers (default: 20)\n");
        fprintf(stderr, "  --embed_every N  Compute embedding every N chunks (default: 1)\n");
        fprintf(stderr, "  --threads N      Number of threads (default: hardware concurrency)\n");
        return 1;
    }

    std::string seg_model_path = argv[1];
    std::string wav_path = argv[2];
    std::string emb_model_path = argv[3];

    // Default parameters optimized for speed
    // Use step=2.5 (half of duration) for good balance of speed and quality
    float duration = 5.0f;
    float step = 2.5f;  // Changed from 0.5 for speed
    float latency = -1.0f;  // Will be set to step if not specified
    float tau_active = 0.6f;
    float tau_embed = 0.3f;
    float rho_update = 0.3f;
    float delta_new = 1.0f;
    int max_speakers = 20;
    int embed_every = 1;
    g_n_threads = (int)std::max(1u, std::thread::hardware_concurrency());

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" && i + 1 < argc) duration = std::stof(argv[++i]);
        else if (arg == "--step" && i + 1 < argc) step = std::stof(argv[++i]);
        else if (arg == "--latency" && i + 1 < argc) latency = std::stof(argv[++i]);
        else if (arg == "--tau_active" && i + 1 < argc) tau_active = std::stof(argv[++i]);
        else if (arg == "--tau_embed" && i + 1 < argc) tau_embed = std::stof(argv[++i]);
        else if (arg == "--rho_update" && i + 1 < argc) rho_update = std::stof(argv[++i]);
        else if (arg == "--delta_new" && i + 1 < argc) delta_new = std::stof(argv[++i]);
        else if (arg == "--max_speakers" && i + 1 < argc) max_speakers = std::stoi(argv[++i]);
        else if (arg == "--embed_every" && i + 1 < argc) embed_every = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--threads" && i + 1 < argc) g_n_threads = std::max(1, std::stoi(argv[++i]));
    }
    
    if (latency < 0) latency = step;

    // Initialize global backend
    init_global_backend();

    pyannote_segmentation_model seg_model;
    load_segmentation_model(seg_model_path, seg_model);

    segmentation_weights_cache seg_cache;
    seg_cache.fwd.resize(4);
    seg_cache.rev.resize(4);
    for (int l = 0; l < 4; ++l) {
        seg_cache.fwd[l].w_ih = dequantize_tensor(seg_model.lstm_fwd[l].w_ih);
        seg_cache.fwd[l].w_hh = dequantize_tensor(seg_model.lstm_fwd[l].w_hh);
        seg_cache.fwd[l].b_ih = dequantize_tensor(seg_model.lstm_fwd[l].b_ih);
        seg_cache.fwd[l].b_hh = dequantize_tensor(seg_model.lstm_fwd[l].b_hh);

        seg_cache.rev[l].w_ih = dequantize_tensor(seg_model.lstm_rev[l].w_ih);
        seg_cache.rev[l].w_hh = dequantize_tensor(seg_model.lstm_rev[l].w_hh);
        seg_cache.rev[l].b_ih = dequantize_tensor(seg_model.lstm_rev[l].b_ih);
        seg_cache.rev[l].b_hh = dequantize_tensor(seg_model.lstm_rev[l].b_hh);
    }

    pyannote_embedding_model emb_model;
    load_embedding_model(emb_model_path, emb_model);

    wav_file wav;
    if (!read_wav(wav_path, wav)) {
        fprintf(stderr, "Failed to read WAV file\n");
        free_global_backend();
        return 1;
    }

    const int sample_rate = wav.sample_rate;
    const int chunk_samples = (int)std::round(duration * sample_rate);
    const int step_samples = (int)std::round(step * sample_rate);
    const int frame_samples = (int)std::round((270.0 / 16000.0) * sample_rate);

    OnlineSpeakerClustering clustering(tau_active, rho_update, delta_new, max_speakers);
    std::vector<bool> active_global(max_speakers, false);
    std::vector<double> active_start(max_speakers, 0.0);
    std::vector<std::vector<float>> last_embeddings(NUM_LOCAL_SPEAKERS, std::vector<float>(EMBEDDING_DIM, NAN));

    struct BufferSeg {
        double start;
        double resolution;
        int num_frames;
        int num_speakers;
        std::vector<float> data; // [num_frames * num_speakers]
    };

    auto hamming = [](int n) {
        std::vector<float> w(n);
        if (n <= 1) {
            if (n == 1) w[0] = 1.0f;
            return w;
        }
        for (int i = 0; i < n; ++i) {
            w[i] = 0.54f - 0.46f * std::cos(2.0f * (float)M_PI * i / (n - 1));
        }
        return w;
    };

    auto aggregate_buffers = [&](
        const std::deque<BufferSeg> & buffers,
        double focus_start,
        double focus_end,
        int target_frames,
        int num_speakers
    ) {
        std::vector<float> sum(target_frames * num_speakers, 0.0f);
        std::vector<float> sum_w(target_frames, 0.0f);

        for (const auto & buf : buffers) {
            std::vector<float> hw = hamming(buf.num_frames);
            int idx0 = (int)std::floor((focus_start - buf.start) / buf.resolution);
            for (int t = 0; t < target_frames; ++t) {
                int src = idx0 + t;
                if (src < 0 || src >= buf.num_frames) {
                    continue;
                }
                float w = hw[src];
                sum_w[t] += w;
                const float * row = &buf.data[src * buf.num_speakers];
                float * out_row = &sum[t * num_speakers];
                for (int s = 0; s < num_speakers; ++s) {
                    out_row[s] += w * row[s];
                }
            }
        }

        for (int t = 0; t < target_frames; ++t) {
            float w = sum_w[t];
            if (w <= 0.0f) continue;
            float * out_row = &sum[t * num_speakers];
            for (int s = 0; s < num_speakers; ++s) {
                out_row[s] /= w;
            }
        }

        return sum;
    };

    std::deque<BufferSeg> pred_buffer;
    int num_overlapping = std::max(1, (int)std::round(latency / step));

    size_t pos = 0;
    int chunk_index = 0;
    while (pos < wav.data.size()) {
        std::vector<float> chunk(chunk_samples, 0.0f);
        size_t copy_len = std::min((size_t)chunk_samples, wav.data.size() - pos);
        memcpy(chunk.data(), &wav.data[pos], copy_len * sizeof(float));

        int num_frames = 0;
        std::vector<std::vector<float>> seg = run_segmentation(seg_model, seg_cache, chunk, sample_rate, num_frames);

        std::vector<std::vector<float>> embeddings(NUM_LOCAL_SPEAKERS, std::vector<float>(EMBEDDING_DIM, NAN));
        bool compute_embeddings = (embed_every <= 1) || (chunk_index % embed_every == 0);
        if (compute_embeddings) {
            const int min_embed_samples = (int)std::round(0.3 * sample_rate);
            const int min_num_frames = (int)std::ceil((double)num_frames * min_embed_samples / std::max(1, chunk_samples));
            std::vector<int> frame_sum(num_frames, 0);
            for (int t = 0; t < num_frames; ++t) {
                int sum = 0;
                for (int s = 0; s < NUM_LOCAL_SPEAKERS; ++s) {
                    if (seg[t][s] >= tau_embed) sum++;
                }
                frame_sum[t] = sum;
            }

            for (int s = 0; s < NUM_LOCAL_SPEAKERS; ++s) {
                float max_val = 0.0f;
                for (int t = 0; t < num_frames; ++t) {
                    max_val = std::max(max_val, seg[t][s]);
                }
                if (max_val < tau_embed) {
                    continue;
                }

                int clean_frames = 0;
                int active_frames = 0;
                for (int t = 0; t < num_frames; ++t) {
                    if (seg[t][s] >= tau_embed) {
                        active_frames++;
                        if (frame_sum[t] == 1) clean_frames++;
                    }
                }

                bool use_clean = clean_frames >= min_num_frames;

                std::vector<float> masked;
                masked.reserve(chunk.size());
                for (int t = 0; t < num_frames; ++t) {
                    if (seg[t][s] < tau_embed) continue;
                    if (use_clean && frame_sum[t] != 1) continue;
                    int s0 = t * frame_samples;
                    int s1 = std::min(s0 + frame_samples, (int)chunk.size());
                    if (s1 <= s0) continue;
                    for (int i = s0; i < s1; ++i) {
                        masked.push_back(chunk[i]);
                    }
                }
                if ((int)masked.size() < sample_rate / 2) {
                    continue;
                }
                std::vector<float> fbank = compute_fbank(masked, sample_rate, 80);
                int T_emb = (int)fbank.size() / 80;
                if (T_emb < 10) continue;
                embeddings[s] = get_embedding(emb_model, fbank, T_emb, 80);
                last_embeddings[s] = embeddings[s];
            }
        } else {
            embeddings = last_embeddings;
        }

        std::vector<int> mapping = clustering.identify(seg, embeddings);
        std::vector<std::vector<float>> global_seg = permute_segmentation(seg, mapping, max_speakers, true);

        const double chunk_start_time = (double)pos / sample_rate;
        const double seg_resolution = duration / num_frames;

        BufferSeg buf;
        buf.start = chunk_start_time;
        buf.resolution = seg_resolution;
        buf.num_frames = num_frames;
        buf.num_speakers = max_speakers;
        buf.data.resize(num_frames * max_speakers, 0.0f);
        for (int t = 0; t < num_frames; ++t) {
            memcpy(&buf.data[t * max_speakers], global_seg[t].data(), max_speakers * sizeof(float));
        }
        pred_buffer.push_back(std::move(buf));

        double focus_start = chunk_start_time + duration - latency;
        double focus_end = focus_start + step;

        if (pred_buffer.size() == 1 && std::abs(chunk_start_time) < 1e-6) {
            focus_start = 0.0;
        }

        int target_frames = (int)std::round((focus_end - focus_start) / seg_resolution);
        if (target_frames <= 0) {
            pos += step_samples;
            chunk_index++;
            continue;
        }

        std::vector<float> agg = aggregate_buffers(pred_buffer, focus_start, focus_end, target_frames, max_speakers);

        for (int t = 0; t < target_frames; ++t) {
            double frame_time = focus_start + (t + 0.5) * seg_resolution;
            const float * row = &agg[t * max_speakers];
            for (int s = 0; s < max_speakers; ++s) {
                bool is_active = row[s] >= tau_active;
                if (is_active && !active_global[s]) {
                    active_global[s] = true;
                    active_start[s] = frame_time;
                } else if (!is_active && active_global[s]) {
                    printf("start=%.1fs stop=%.1fs speaker_SPEAKER_%02d\n", active_start[s], frame_time, s);
                    active_global[s] = false;
                }
            }
        }

        if ((int)pred_buffer.size() > num_overlapping) {
            pred_buffer.pop_front();
        }

        pos += step_samples;
        chunk_index++;
        if (step_samples <= 0) break;
    }

    double end_time = (double)wav.data.size() / sample_rate;
    for (int s = 0; s < max_speakers; ++s) {
        if (active_global[s]) {
            printf("start=%.1fs stop=%.1fs speaker_SPEAKER_%02d\n", active_start[s], end_time, s);
        }
    }

    free_global_backend();

    return 0;
}
