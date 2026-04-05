#include "pool1d.cuh"

template <typename Ti, typename To>
static __global__ void pool1d_nchw_kernel(
        const int iw, const int ow,
        const int kw, const int sw,
        const int pw, const int parallel_elements,
        const Ti * src, To * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int nc = idx / ow;
    const int cur_ow = idx % ow;

    const Ti * i_ptr = src + nc * iw;
    To * o_ptr = dst + nc * ow;

    const int start_w = cur_ow * sw - pw;
    const int bw = max(0, start_w);
    const int ew = min(iw, start_w + kw);

    To res = 0;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0; break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: assert(false);
    }

    const To scale = 1.0f / kw;

    for (int j = bw; j < ew; ++j) {
#if __CUDA_ARCH__ >= 350
        const Ti cur = __ldg(i_ptr + j);
#else
        const Ti cur = i_ptr[j];
#endif
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur * scale; break;
            case GGML_OP_POOL_MAX: res = max(res, (To) cur); break;
            default: assert(false);
        }
    }

    o_ptr[cur_ow] = res;
}

static void pool1d_nchw_kernel_f32_f32_cuda(
        const int iw, const int ow,
        const int kw, const int sw,
        const int pw, const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {
    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);

    pool1d_nchw_kernel<<<block_nums, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(
        iw, ow,
        kw, sw,
        pw, parallel_elements,
        src, dst, op);
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const enum ggml_op_pool op = static_cast<enum ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t N = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = (int) (N * OW);

    pool1d_nchw_kernel_f32_f32_cuda(
        (int) IW,
        (int) OW,
        k0,
        s0,
        p0,
        parallel_elements,
        src0_d,
        dst_d,
        op,
        stream);
}
