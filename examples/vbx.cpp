#include "vbx.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <cstring>

// --- Helper Math Functions ---

static float dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for(int i=0; i<n; i++) sum += a[i] * b[i];
    return sum;
}

static void mat_vec_mul(const std::vector<float>& M, const std::vector<float>& v, std::vector<float>& out, int rows, int cols) {
    // M is rows x cols
    out.resize(rows);
    for(int i=0; i<rows; i++) {
        out[i] = dot_product(&M[i*cols], v.data(), cols);
    }
}

static void mat_vec_mul_trans(const std::vector<float>& M, const std::vector<float>& v, std::vector<float>& out, int rows, int cols) {
    // M is rows x cols, compute M^T * v
    out.assign(cols, 0.0f);
    for(int i=0; i<rows; i++) {
        float val = v[i];
        for(int j=0; j<cols; j++) {
            out[j] += M[i*cols + j] * val;
        }
    }
}

static std::vector<float> normalize(const std::vector<float>& v) {
    float norm = 0.0f;
    for(float x : v) norm += x*x;
    norm = std::sqrt(norm + 1e-12f);
    std::vector<float> res = v;
    for(float &x : res) x /= norm;
    return res;
}

static float logsumexp(const std::vector<float>& vals) {
    if (vals.empty()) return -std::numeric_limits<float>::infinity();
    float max_val = vals[0];
    for(float v : vals) if(v > max_val) max_val = v;
    
    float sum = 0.0f;
    for(float v : vals) sum += std::exp(v - max_val);
    return max_val + std::log(sum);
}

// --- Debug Helpers ---

static void print_vec(const char* name, const std::vector<float>& v, int n = 5) {
    printf("  %s[:5]: [", name);
    for (int i = 0; i < std::min(n, (int)v.size()); i++) {
        printf("%.8f%s", v[i], i < n - 1 ? ", " : "");
    }
    printf("]\n");
}

static float vec_norm(const std::vector<float>& v) {
    float sum = 0.0f;
    for (float x : v) sum += x * x;
    return std::sqrt(sum);
}

void vbx_print_model_summary(const vbx_model & model) {
    printf("\n=== VBx Model Summary ===\n");
    printf("D_emb=%d, D_lda=%d\n", model.D_emb, model.D_lda);
    print_vec("mean1", model.mean1);
    print_vec("mean2", model.mean2);
    printf("  lda.flatten()[:5]: [");
    for (int i = 0; i < 5 && i < (int)model.lda.size(); i++)
        printf("%.8f%s", model.lda[i], i < 4 ? ", " : "");
    printf("]\n");
    print_vec("plda_mu", model.plda_mu);
    printf("  plda_tr.flatten()[:5]: [");
    for (int i = 0; i < 5 && i < (int)model.plda_tr.size(); i++)
        printf("%.8f%s", model.plda_tr[i], i < 4 ? ", " : "");
    printf("]\n");
    print_vec("plda_psi", model.plda_psi);
    printf("=========================\n\n");
}

// --- VBx Model Loading ---

static bool copy_tensor_data(struct ggml_context * ctx, const char * name, std::vector<float> & out, int expected_size = -1) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "VBx: Tensor %s not found\n", name);
        return false;
    }
    
    int size = ggml_nelements(t);
    if (expected_size != -1 && size != expected_size) {
         fprintf(stderr, "VBx: Tensor %s size mismatch: expected %d, got %d\n", name, expected_size, size);
         // return false; // Allow mismatch if it's just dimension logic, but better strict.
    }
    
    out.resize(size);
    memcpy(out.data(), t->data, size * sizeof(float));
    return true;
}

bool init_vbx_model_from_gguf(struct ggml_context * ctx, vbx_model & model) {
    struct ggml_tensor * t_mean1 = ggml_get_tensor(ctx, "vbx.mean1");
    if (!t_mean1) {
        printf("VBx: vbx.mean1 not found in context. Searching for VBx tensors:\n");
        struct ggml_tensor * t = ggml_get_first_tensor(ctx);
        int count = 0;
        int found = 0;
        while(t) {
            count++;
            if (strstr(t->name, "vbx")) {
                printf("  FOUND: %s\n", t->name);
                found++;
            }
            t = ggml_get_next_tensor(ctx, t);
        }
        printf("Scanned %d tensors, found %d VBx tensors.\n", count, found);
        return false; // Not a VBx model
    }
    
    int D_emb = t_mean1->ne[0];
    model.D_emb = D_emb;
    
    struct ggml_tensor * t_lda = ggml_get_tensor(ctx, "vbx.lda");
    if (!t_lda) return false;
    int D_lda = t_lda->ne[0]; // Wait, lda is (256, 128) usually. ne[0] is dim 0 (128?), ne[1] is 256?
    // In GGUF, dimensions are stored reversed?
    // Python np array (256, 128).
    // GGUF tensor ne[0]=128, ne[1]=256.
    // So D_lda is ne[0].
    model.D_lda = D_lda;
    
    printf("VBx: Found model with D_emb=%d, D_lda=%d\n", D_emb, D_lda);
    
    if(!copy_tensor_data(ctx, "vbx.mean1", model.mean1)) return false;
    if(!copy_tensor_data(ctx, "vbx.mean2", model.mean2)) return false;
    if(!copy_tensor_data(ctx, "vbx.lda", model.lda)) return false;
    if(!copy_tensor_data(ctx, "vbx.plda_mu", model.plda_mu)) return false;
    if(!copy_tensor_data(ctx, "vbx.plda_tr", model.plda_tr)) return false;
    if(!copy_tensor_data(ctx, "vbx.plda_psi", model.plda_psi)) return false;
    
    return true;
}

// --- Preprocessing ---

std::vector<float> vbx_transform_embedding(const vbx_model & model, const std::vector<float> & emb_in, bool debug) {
    int D = model.D_emb;
    int L = model.D_lda;
    
    if (debug) {
        printf("\n=== vbx_transform_embedding DEBUG ===\n");
        print_vec("input", emb_in);
        printf("  input norm: %.8f\n", vec_norm(emb_in));
    }
    
    // 1. Center (x - mean1)
    std::vector<float> x(D);
    for(int i=0; i<D; i++) x[i] = emb_in[i] - model.mean1[i];
    
    if (debug) {
        printf("\nStep 1 (x - mean1):\n");
        print_vec("centered", x);
        printf("  norm: %.8f\n", vec_norm(x));
    }
    
    // 2. Norm
    // Python: xvec_tf = sqrt(L) * l2_norm( lda.T.dot( sqrt(D) * l2_norm(x - mean1).T ).T - mean2 )
    x = normalize(x);
    
    if (debug) {
        printf("\nStep 2 (L2 norm):\n");
        print_vec("normed", x);
        printf("  norm: %.8f (should be 1.0)\n", vec_norm(x));
    }
    
    float scale1 = std::sqrt((float)D);
    for(int i=0; i<D; i++) x[i] *= scale1;
    
    if (debug) {
        printf("\nStep 3 (scale by sqrt(%d) = %.4f):\n", D, scale1);
        print_vec("scaled", x);
        printf("  norm: %.8f\n", vec_norm(x));
    }
    
    // 3. LDA (lda.T @ x)
    // lda is (256, 128) in row-major (numpy C-order)
    // We want lda.T.dot(x) = sum_i lda[i,j] * x[i] for each j
    std::vector<float> x_lda(L);
    mat_vec_mul_trans(model.lda, x, x_lda, D, L);
    
    if (debug) {
        printf("\nStep 4 (LDA: lda.T @ x):\n");
        print_vec("lda_out", x_lda);
        printf("  norm: %.8f\n", vec_norm(x_lda));
    }
    
    // 4. Center 2 (x - mean2)
    for(int i=0; i<L; i++) x_lda[i] -= model.mean2[i];
    
    if (debug) {
        printf("\nStep 5 (x - mean2):\n");
        print_vec("centered2", x_lda);
        printf("  norm: %.8f\n", vec_norm(x_lda));
    }
    
    // 5. Norm 2
    x_lda = normalize(x_lda);
    
    if (debug) {
        printf("\nStep 6 (L2 norm 2):\n");
        print_vec("normed2", x_lda);
        printf("  norm: %.8f (should be 1.0)\n", vec_norm(x_lda));
    }
    
    float scale2 = std::sqrt((float)L);
    for(int i=0; i<L; i++) x_lda[i] *= scale2;
    
    if (debug) {
        printf("\nStep 7 (scale by sqrt(%d) = %.4f):\n", L, scale2);
        print_vec("scaled2", x_lda);
        printf("  norm: %.8f\n", vec_norm(x_lda));
    }
    
    // 6. PLDA Transform
    // plda_tf = lambda x0: (x0 - plda_mu).dot(plda_tr.T)
    for(int i=0; i<L; i++) x_lda[i] -= model.plda_mu[i];
    
    if (debug) {
        printf("\nStep 8 (x - plda_mu):\n");
        print_vec("plda_centered", x_lda);
        printf("  norm: %.8f\n", vec_norm(x_lda));
    }
    
    // x @ plda_tr.T
    // Python: (row_vec) @ plda_tr.T = row_vec @ plda_tr.T
    // = for each output j: sum_i x[i] * plda_tr.T[i,j] = sum_i x[i] * plda_tr[j,i]
    // plda_tr is (128, 128) row-major.
    // result[j] = sum_i x[i] * plda_tr[j*128 + i]
    // This is mat_vec_mul with (rows=L, cols=L)
    std::vector<float> final_emb(L);
    mat_vec_mul(model.plda_tr, x_lda, final_emb, L, L);
    
    if (debug) {
        printf("\nStep 9 (x @ plda_tr.T) - FINAL:\n");
        print_vec("final", final_emb);
        printf("  norm: %.8f\n", vec_norm(final_emb));
        printf("=====================================\n\n");
    }
    
    return final_emb;
}

// --- AHC Initialization for VBx ---
// We need basic cosine clustering but using the TRANSFORMED features?
// Python says: "AHC ... train_embeddings_normed = train_embeddings / norm"
// "linkage(..., method='centroid', metric='euclidean')"
// Centroid linkage + Euclidean is standard.
// For initialization, we can reuse my existing AHC code but make sure to use Euclidean distance equivalent.
// Normalized Euclidean distance is related to Cosine.
// 0.6 VBx threshold corresponds to some Cosine threshold.

static std::vector<int> simple_ahc(const std::vector<std::vector<float>> & embeddings, float threshold) {
    int n = embeddings.size();
    if(n==0) return {};
    if(n==1) return {0};
    
    int dim = embeddings[0].size();
    
    // Normalize embeddings (Python uses unit-normalized embeddings + euclidean metric)
    std::vector<std::vector<float>> normed = embeddings;
    for (auto & v : normed) {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm + 1e-12f);
        for (float & x : v) x /= norm;
    }
    
    // Initialize: each point is its own cluster
    // Use vector to track active clusters and their centroids
    std::vector<bool> active(n, true);
    std::vector<std::vector<float>> centroids = normed;  // Cache centroids
    std::vector<int> cluster_size(n, 1);
    std::vector<int> parent(n);  // For union-find style tracking
    for (int i = 0; i < n; i++) parent[i] = i;
    
    // Precompute initial pairwise distances (upper triangular matrix stored as flat)
    // dist[i * n + j] for j > i
    // Precompute initial pairwise distances using ggml (Gram matrix)
    // dist[i, j] = sqrt(2 * (1 - cos_sim(i, j))) since vectors are normalized
    std::vector<float> dist_matrix(n * n, 1e20f);
    
    {
        // Setup ggml for G = X @ X^T
        size_t buf_size = 1024 * 1024 * 8; // 8MB
        struct ggml_init_params params = { buf_size, NULL, true };
        struct ggml_context * ctx = ggml_init(params);
        ggml_backend_t backend = ggml_backend_cpu_init();
        
        // Tensor X: (dim, n)
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n);
        
        // Graph
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        // G = X @ X^T. ggml_mul_mat(X, X) computes X^T @ X ?? 
        // Wait, earlier logic: ggml_mul_mat(A, B) -> A^T @ B.
        // X (dim, n). ne[0]=dim.
        // ggml_mul_mat(x, x): X^T @ X -> (n, dim) @ (dim, n) -> (n, n).
        // Result: (n, n).
        // G[i, j] = sum_k x[k, i] * x[k, j]. 
        // This is dot product of vector i and vector j.
        // Yes! Correct.
        struct ggml_tensor * G = ggml_mul_mat(ctx, x, x);
        
        ggml_build_forward_expand(gf, G);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        
        // Set Data
        for(int i=0; i<n; i++) {
             ggml_backend_tensor_set(x, normed[i].data(), i * dim * sizeof(float), dim * sizeof(float));
        }
        
        ggml_backend_graph_compute(backend, gf);
        
        // Read G
        std::vector<float> G_data(n * n);
        ggml_backend_tensor_get(G, G_data.data(), 0, ggml_nbytes(G));
        
        // Compute Distances
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                // G is column-major (n, n). G[j, i] = G_data[i*n + j] ?
                // G[col=j, row=i] -> G_data[i*n + j]? No.
                // G ne[0]=n. (contiguous dim).
                // G[x, y] -> data[y*n + x].
                // G[i, j] (row i, col j of mathematical matrix) ??
                // G is result of X^T @ X.
                // Element (i, j) of resulting matrix involves row i of X^T and col j of X.
                // Row i of X^T is Col i of X (vector i).
                // Col j of X is vector j.
                // So Result(i, j) = dot(vec i, vec j).
                // In tensor memory: Result[j, i]?
                // ggml_mul_mat output is (n, n).
                // Usually output is col-major. 
                // data[k] corresponds to (row=k%n, col=k/n).
                // So element at (i, j) is at i + j*n.
                // We want dot(vec i, vec j).
                // Since dot is symmetric, G[i,j] == G[j,i].
                // So index (i, j) or (j, i) doesn't matter for value.
                float dot = G_data[j * n + i]; // or i*n+j
                
                // Dist = sqrt(2 * (1 - dot))
                // Clamp dot to 1.0
                if (dot > 1.0f) dot = 1.0f;
                float dsq = 2.0f * (1.0f - dot);
                if (dsq < 0.0f) dsq = 0.0f;
                float d = std::sqrt(dsq);
                
                dist_matrix[i * n + j] = d;
                dist_matrix[j * n + i] = d;
            }
        }
        
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    
    int num_active = n;
    int merge_count = 0;
    
    while (num_active > 1) {
        // Find minimum distance pair among active clusters
        float best_dist = 1e20f;
        int best_i = -1, best_j = -1;
        
        for (int i = 0; i < n; i++) {
            if (!active[i]) continue;
            for (int j = i + 1; j < n; j++) {
                if (!active[j]) continue;
                if (dist_matrix[i * n + j] < best_dist) {
                    best_dist = dist_matrix[i * n + j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        
        if (best_i < 0 || best_dist > threshold) break;
        
        // Merge cluster j into cluster i
        // Update centroid of i (weighted average)
        int size_i = cluster_size[best_i];
        int size_j = cluster_size[best_j];
        int new_size = size_i + size_j;
        
        for (int k = 0; k < dim; k++) {
            centroids[best_i][k] = (centroids[best_i][k] * size_i + centroids[best_j][k] * size_j) / new_size;
        }
        cluster_size[best_i] = new_size;
        
        // Mark j as inactive
        active[best_j] = false;
        parent[best_j] = best_i;
        num_active--;
        
        // Update distances from best_i to all other active clusters
        for (int k = 0; k < n; k++) {
            if (!active[k] || k == best_i) continue;
            float d = 0.0f;
            for (int dim_idx = 0; dim_idx < dim; dim_idx++) {
                float diff = centroids[best_i][dim_idx] - centroids[k][dim_idx];
                d += diff * diff;
            }
            d = std::sqrt(d);
            dist_matrix[best_i * n + k] = d;
            dist_matrix[k * n + best_i] = d;
        }
        
        merge_count++;
    }
    
    // Build final labels using path compression
    std::vector<int> labels(n);
    std::vector<int> root_to_label(n, -1);  // Map root index to label
    int next_label = 0;
    
    for (int i = 0; i < n; i++) {
        // Find root with path compression
        int root = i;
        while (parent[root] != root) root = parent[root];
        // Path compression
        int curr = i;
        while (parent[curr] != root) {
            int next = parent[curr];
            parent[curr] = root;
            curr = next;
        }
        
        if (root_to_label[root] == -1) {
            root_to_label[root] = next_label++;
        }
        labels[i] = root_to_label[root];
    }
    
    return labels;
}

// --- VBx Core ---

#include <map>

std::vector<int> cluster_vbx(
    const vbx_model & model,
    const std::vector<std::vector<float>> & vbx_embeddings,
    const std::vector<std::vector<float>> & raw_embeddings,
    const vbx_params & params
) {
    if (vbx_embeddings.empty()) return {};
    
    // Extract params
    float threshold = params.threshold;
    int maxIters = params.max_iters;
    float Fa = params.Fa;
    float Fb = params.Fb;
    float init_smoothing = params.init_smoothing;
    
    printf("VBx: Parameters - Fa=%.3f, Fb=%.3f, threshold=%.3f, max_iters=%d\n", 
           Fa, Fb, threshold, maxIters);
    
    // 1. Init with AHC on RAW embeddings (Python: centroid linkage + euclidean on unit-norm)
    std::vector<int> init_labels = simple_ahc(raw_embeddings, threshold);
    
    int N = vbx_embeddings.size();
    int K_init = 0;
    for(int l : init_labels) K_init = std::max(K_init, l + 1);
    printf("VBx: AHC init found %d clusters from %d embeddings\n", K_init, N);
    
    if (K_init <= 1) {
        printf("VBx: Warning - Only %d cluster(s) from AHC init. Consider lower threshold.\n", K_init);
        // Just return AHC labels directly
        return init_labels;
    }
    
    // DEBUG: Skip VBx iteration and return AHC result
    // This helps verify if AHC itself is working correctly
    #if 0
    printf("VBx: DEBUG - Skipping VBx iteration, returning AHC init labels\n");
    return init_labels;
    #endif
    
    // 2. Prepare for VBx
    int D = model.D_lda;
    std::vector<std::vector<float>> gamma(N, std::vector<float>(K_init, 0.0f));
    for(int i=0; i<N; i++) gamma[i][init_labels[i]] = 1.0f;
    if (init_smoothing >= 0.0f) {
        for (int i = 0; i < N; ++i) {
            float max_v = -1e20f;
            for (int k = 0; k < K_init; ++k) {
                float v = gamma[i][k] * init_smoothing;
                if (v > max_v) max_v = v;
            }
            float sum = 0.0f;
            for (int k = 0; k < K_init; ++k) {
                gamma[i][k] = std::exp(gamma[i][k] * init_smoothing - max_v);
                sum += gamma[i][k];
            }
            float inv = 1.0f / (sum + 1e-8f);
            for (int k = 0; k < K_init; ++k) gamma[i][k] *= inv;
        }
    }

    // pi initialized uniformly (python uses int -> uniform)
    std::vector<float> pi(K_init, 1.0f / K_init);
    
    std::vector<float> G(N);
    for(int i=0; i<N; i++) {
        float sum_sq = 0.0f;
        for(float x : vbx_embeddings[i]) sum_sq += x*x;
        G[i] = -0.5f * (sum_sq + D * std::log(2 * M_PI));
    }
    
    std::vector<float> V(D);
    for(int i=0; i<D; i++) V[i] = std::sqrt(model.plda_psi[i]);
    
    std::vector<std::vector<float>> rho(N, std::vector<float>(D));
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) rho[i][j] = vbx_embeddings[i][j] * V[j];
    }
    
    // Loop
    float prev_elbo = -1e20f;
    const float epsilon = 1e-4f;
    for(int iter=0; iter<maxIters; iter++) {
        std::vector<float> gamma_sum(K_init, 0.0f);
        for(int i=0; i<N; i++) {
            for(int k=0; k<K_init; k++) gamma_sum[k] += gamma[i][k];
        }
        
        std::vector<std::vector<float>> invL(K_init, std::vector<float>(D));
        for(int k=0; k<K_init; k++) {
            for(int d=0; d<D; d++) {
                invL[k][d] = 1.0f / (1.0f + (Fa/Fb) * gamma_sum[k] * model.plda_psi[d]);
            }
        }
        
        std::vector<std::vector<float>> alpha(K_init, std::vector<float>(D, 0.0f));
        for(int k=0; k<K_init; k++) {
            for(int d=0; d<D; d++) {
                float sum = 0.0f;
                for(int i=0; i<N; i++) sum += gamma[i][k] * rho[i][d];
                alpha[k][d] = (Fa/Fb) * invL[k][d] * sum;
            }
        }
        
        std::vector<std::vector<float>> log_p(N, std::vector<float>(K_init));
        for(int k=0; k<K_init; k++) {
            float term2 = 0.0f;
            for(int d=0; d<D; d++) {
                term2 += (invL[k][d] + alpha[k][d]*alpha[k][d]) * model.plda_psi[d];
            }
            term2 *= 0.5f;
            
            for(int i=0; i<N; i++) {
                float term1 = 0.0f;
                for(int d=0; d<D; d++) term1 += rho[i][d] * alpha[k][d];
                log_p[i][k] = Fa * (term1 - term2 + G[i]);
            }
        }
        
        float eps = 1e-8f;
        // compute ELBO components like python
        float log_pX = 0.0f;
        for(int i=0; i<N; i++) {
            std::vector<float> log_probs(K_init);
            for(int k=0; k<K_init; k++) {
                log_probs[k] = log_p[i][k] + std::log(pi[k] + eps);
            }
            float max_lp = -1e20f;
            for(float v : log_probs) if(v > max_lp) max_lp = v;

            float sum_prob = 0.0f;
            for(int k=0; k<K_init; k++) {
                gamma[i][k] = std::exp(log_probs[k] - max_lp);
                sum_prob += gamma[i][k];
            }
            for(int k=0; k<K_init; k++) gamma[i][k] /= (sum_prob + eps);

            log_pX += (max_lp + std::log(sum_prob + eps));
        }
        
        float sum_pi = 0.0f;
        for(int k=0; k<K_init; k++) {
            pi[k] = 0.0f;
            for(int i=0; i<N; i++) pi[k] += gamma[i][k];
            sum_pi += pi[k];
        }
        for(int k=0; k<K_init; k++) pi[k] /= (sum_pi + eps);

        float elbo_reg = 0.0f;
        for(int k=0; k<K_init; k++) {
            for(int d=0; d<D; d++) {
                elbo_reg += std::log(invL[k][d] + eps) - invL[k][d] - alpha[k][d]*alpha[k][d] + 1.0f;
            }
        }
        float elbo = log_pX + Fb * 0.5f * elbo_reg;
        
        // Count active speakers (pi > 0.01)
        int active_speakers = 0;
        for(int k=0; k<K_init; k++) {
            if (pi[k] > 0.01f) active_speakers++;
        }
        if (VBX_DEBUG) {
            printf("VBx: Iter %2d | ELBO=%.2f (delta=%.4f) | log_pX=%.2f | active_spk=%d\n", 
               iter, elbo, elbo - prev_elbo, log_pX, active_speakers);
        }
        
        if (iter > 0 && (elbo - prev_elbo) < epsilon) {
            if (elbo - prev_elbo < 0) {
                printf("VBx: Warning - ELBO decreased! This may indicate numerical issues.\n");
            } else {
                printf("VBx: Converged at iteration %d\n", iter);
            }
            break;
        }
        prev_elbo = elbo;
    }
    
    // Final assignment (drop speakers with tiny priors like python)
    std::vector<int> keep;
    for (int k = 0; k < K_init; ++k) {
        if (pi[k] > 1e-7f) keep.push_back(k);
    }
    if (keep.empty()) {
        for (int k = 0; k < K_init; ++k) keep.push_back(k);
    }

    std::vector<int> final_labels(N);
    std::map<int, int> counts;
    for(int i=0; i<N; i++) {
        int best_k = keep[0];
        float best_g = gamma[i][best_k];
        for (int kk = 1; kk < (int)keep.size(); ++kk) {
            int k = keep[kk];
            if (gamma[i][k] > best_g) {
                best_g = gamma[i][k];
                best_k = k;
            }
        }
        final_labels[i] = best_k;
        counts[best_k]++;
    }

    // remap to consecutive labels
    std::map<int, int> remap;
    int next_id = 0;
    for (int & lbl : final_labels) {
        if (remap.find(lbl) == remap.end()) remap[lbl] = next_id++;
        lbl = remap[lbl];
    }
    if (VBX_DEBUG)
    {
        printf("VBx: Final labels distribution: ");
        for(auto const& [k, v] : counts) printf("%d:%d ", k, v);
        printf("\n");
    }
        
    return final_labels;
}

// --- Optimization with GGML ---

std::vector<std::vector<float>> vbx_transform_embeddings_batch(
    const vbx_model & model, 
    const std::vector<std::vector<float>> & embeddings, 
    bool debug) {
    
    if (embeddings.empty()) return {};
    int N = embeddings.size();
    int D_emb = model.D_emb; // 256
    int D_lda = model.D_lda; // 128
    
    // Setup generic CPU backend
    size_t buf_size = 1024 * 1024 * 32; // 32MB buffer
    struct ggml_init_params params = { buf_size, NULL, true }; // no_alloc=true for backend alloc
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init(); // CPU for VBx ops
    
    // 1. Create Input/Weight Tensors
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_emb, N);
    struct ggml_tensor * mean1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_emb, 1);
    struct ggml_tensor * lda_T = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_emb, D_lda); // (256, 128)
    struct ggml_tensor * mean2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_lda, 1);
    struct ggml_tensor * plda_tr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_lda, D_lda);
    struct ggml_tensor * plda_mu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_lda, 1);
    
    // 2. Build Graph (defines operations and output tensors)
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    
    // sub mean1 (broadcast)
    struct ggml_tensor * cur = ggml_sub(ctx, x, mean1);
    
    // RMS Norm (Norm + Scale sqrt(D))
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    
    // LDA: lda.T @ x
    // lda_T (256, 128). x (256, N).
    // ggml_mul_mat computes A^T @ B = lda_T^T @ x = (128, 256) @ (256, N) -> (128, N)
    cur = ggml_mul_mat(ctx, lda_T, cur); 
    
    // sub mean2
    struct ggml_tensor * mean2_r = ggml_reshape_2d(ctx, mean2, D_lda, 1);
    cur = ggml_sub(ctx, cur, mean2_r);
    
    // RMS Norm 2
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    
    // sub plda_mu
    struct ggml_tensor * plda_mu_r = ggml_reshape_2d(ctx, plda_mu, D_lda, 1);
    cur = ggml_sub(ctx, cur, plda_mu_r);
    
    // PLDA: plda_tr @ x (equivalent to x @ plda.T)
    cur = ggml_mul_mat(ctx, plda_tr, cur);
    
    ggml_build_forward_expand(gf, cur);
    
    // 3. Allocate Tensors (Inputs + Intermediate outputs)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    // 4. Set Data (Now that memory is allocated)
    for(int n=0; n<N; n++) {
         ggml_backend_tensor_set(x, embeddings[n].data(), n * D_emb * sizeof(float), D_emb * sizeof(float));
    }
    ggml_backend_tensor_set(mean1, model.mean1.data(), 0, ggml_nbytes(mean1));
    {
        std::vector<float> lda_t_data(D_emb * D_lda);
        for(int j=0; j<D_lda; j++) {
            for(int i=0; i<D_emb; i++) {
                lda_t_data[j*D_emb + i] = model.lda[i*D_lda + j];
            }
        }
        ggml_backend_tensor_set(lda_T, lda_t_data.data(), 0, ggml_nbytes(lda_T));
    }
    ggml_backend_tensor_set(mean2, model.mean2.data(), 0, ggml_nbytes(mean2));
    ggml_backend_tensor_set(plda_tr, model.plda_tr.data(), 0, ggml_nbytes(plda_tr));
    ggml_backend_tensor_set(plda_mu, model.plda_mu.data(), 0, ggml_nbytes(plda_mu));
    
    // 5. Compute
    ggml_backend_graph_compute(backend, gf);
    
    // 6. Extract Results
    std::vector<std::vector<float>> result(N, std::vector<float>(D_lda));
    std::vector<float> res_flat(N * D_lda);
    ggml_backend_tensor_get(cur, res_flat.data(), 0, ggml_nbytes(cur));
    
    for(int n=0; n<N; n++) {
        for(int d=0; d<D_lda; d++) {
            result[n][d] = res_flat[n*D_lda + d];
        }
    }
    
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    
    return result;
}
