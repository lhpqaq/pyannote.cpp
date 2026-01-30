#pragma once

#include "ggml.h"
#include <vector>
#include <string>
#include <cmath>

// Enable verbose debug output
#ifndef VBX_DEBUG
#define VBX_DEBUG 0
#endif

struct vbx_model {
    int D_emb; // 256
    int D_lda; // 128
    
    std::vector<float> mean1;      // (D_emb)
    std::vector<float> mean2;      // (D_lda)
    std::vector<float> lda;        // (D_emb * D_lda) - Row major (numpy C order)
    
    std::vector<float> plda_mu;    // (D_lda)
    std::vector<float> plda_tr;    // (D_lda * D_lda)
    std::vector<float> plda_psi;   // (D_lda)
};

// VBx clustering parameters
struct vbx_params {
    float Fa = 0.3f;         // Speaker factor weight
    float Fb = 6.0f;         // Prior regularization (lower = more speakers)
    float threshold = 0.6f;  // AHC init threshold (normalized Euclidean)
    int max_iters = 20;      // VB loop iterations
    float init_smoothing = 7.0f;
};

// Initialize VBx model from tensors in the GGML context
// Looks for tensors: vbx.mean1, vbx.mean2, vbx.lda, vbx.plda_mu, vbx.plda_tr, vbx.plda_psi
bool init_vbx_model_from_gguf(struct ggml_context * ctx, vbx_model & model);

// Preprocess and transform embeddings to PLDA space (Single)
std::vector<float> vbx_transform_embedding(const vbx_model & model, const std::vector<float> & emb, bool debug = false);

// Batched preprocessing and transformation (Optimized with ggml)
// Input: embeddings (N x D_emb)
// Output: transformed embeddings (N x D_lda)
std::vector<std::vector<float>> vbx_transform_embeddings_batch(
    const vbx_model & model, 
    const std::vector<std::vector<float>> & embeddings, 
    bool debug = false);

// Print VBx model matrices summary (for debugging)
void vbx_print_model_summary(const vbx_model & model);

// Run VBx Clustering
// inputs: 
//   vbx_embeddings (N x D_lda) - already transformed
//   raw_embeddings (N x D_emb) - for AHC initialization
// returns: vector of speaker labels
std::vector<int> cluster_vbx(
    const vbx_model & model,
    const std::vector<std::vector<float>> & vbx_embeddings,
    const std::vector<std::vector<float>> & raw_embeddings,
    const vbx_params & params = vbx_params()
);
