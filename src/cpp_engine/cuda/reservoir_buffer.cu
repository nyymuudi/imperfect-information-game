/**
 * cuda/reservoir_buffer.cu
 *
 * GPU-accelerated Vitter Algorithm R reservoir sampling.
 *
 * Strategy: maintain the buffer in pinned (page-locked) host memory for
 * zero-copy access, and run the index-selection kernel on GPU.
 * This avoids the D2H/H2D overhead of copying the entire buffer.
 *
 * Kernel: given N new samples and current reservoir state (size M, capacity C,
 * total_seen T), compute which reservoir slots to overwrite for each incoming
 * sample. This is embarrassingly parallel: each new sample independently
 * draws a uniform integer in [0, T+i) and checks if it falls in [0, C).
 */

#ifdef CFR_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>

// ── Kernel: compute replacement indices ──────────────────────────────────────
// For each of N incoming samples, output the slot to write (-1 = reject).
// Uses cuRAND for per-thread random state.

__global__ void reservoir_indices_kernel(
    int64_t* __restrict__ out_slots,   // [N] output slot indices (-1 = reject)
    const int64_t  capacity,
    const int64_t  total_seen_before,  // T before this batch
    const int64_t  current_size,       // current fill level (min(T, C))
    const int64_t  n_samples,          // N = number of incoming samples
    const uint64_t seed)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    // Init cuRAND state per thread
    curandState_t state;
    curand_init(seed + idx, 0, 0, &state);

    const int64_t t = total_seen_before + idx;  // total seen after this sample

    int64_t slot;
    if (t < capacity) {
        // Buffer not yet full — always insert at position t
        slot = t;
    } else {
        // Draw uniform integer in [0, t+1)
        // curand_uniform gives (0,1], so: floor(curand * (t+1))
        double r = curand_uniform_double(&state);
        slot = static_cast<int64_t>(r * static_cast<double>(t + 1));
        if (slot >= capacity) slot = -1;   // reject
    }
    out_slots[idx] = slot;
}

// ── Host-side launcher ────────────────────────────────────────────────────────

extern "C" {

/**
 * Launch reservoir sampling kernel.
 *
 * @param out_slots   Device pointer [n_samples] — output slot indices
 * @param capacity    Reservoir capacity C
 * @param total_seen  Total samples seen before this batch
 * @param cur_size    Current number of elements in reservoir
 * @param n_samples   Batch size
 * @param seed        Random seed (vary per call for independence)
 */
void cuda_reservoir_indices(
    int64_t* out_slots,
    int64_t  capacity,
    int64_t  total_seen,
    int64_t  cur_size,
    int64_t  n_samples,
    uint64_t seed)
{
    const int block_size = 256;
    const int grid_size  = (n_samples + block_size - 1) / block_size;

    reservoir_indices_kernel<<<grid_size, block_size>>>(
        out_slots, capacity, total_seen, cur_size, n_samples, seed);

    cudaDeviceSynchronize();
}

/**
 * Check last CUDA error and print diagnostics.
 * Returns 0 on success, nonzero on error.
 */
int cuda_check_last_error(const char* context) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA error] %s: %s\n", context, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

/**
 * Allocate device memory and return pointer cast to size_t (for pybind11 int).
 */
void* cuda_alloc(size_t bytes) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void cuda_free(void* ptr) {
    cudaFree(ptr);
}

void cuda_memcpy_to_host(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

} // extern "C"

// ── Warp-level parallel regret sum (bonus kernel) ────────────────────────────
// Given a batch of (info_set_hash, action, regret) triples from one traversal,
// accumulate regrets per (info_set, action) pair using atomic adds.
// Useful when batching hundreds of traversals simultaneously.

__global__ void accumulate_regrets_kernel(
    float*         __restrict__ regret_table,  // [num_info_sets * num_actions]
    const uint32_t* __restrict__ iset_indices, // [N] hash-based info set index
    const int8_t*  __restrict__ actions,       // [N]
    const float*   __restrict__ regrets,       // [N]
    const int      num_actions,
    const int      n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t iset = iset_indices[i];
    int      act  = actions[i];
    float    reg  = regrets[i];

    // Atomic add into flat regret table
    atomicAdd(&regret_table[iset * num_actions + act], reg);
}

extern "C" {
void cuda_accumulate_regrets(
    float*          regret_table,
    const uint32_t* iset_indices,
    const int8_t*   actions,
    const float*    regrets,
    int             num_actions,
    int             n)
{
    const int block_size = 128;
    const int grid_size  = (n + block_size - 1) / block_size;
    accumulate_regrets_kernel<<<grid_size, block_size>>>(
        regret_table, iset_indices, actions, regrets, num_actions, n);
    cudaDeviceSynchronize();
}
} // extern "C"

#endif // CFR_CUDA_AVAILABLE
