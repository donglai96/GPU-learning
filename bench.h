#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Minimal CUDA error check, independent of the caller's own macro.
#define BENCH_CUDA_CHECK(call)                                             \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr,                                                    \
                "CUDA error at %s:%d: %s\n",                               \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// Time a GPU launch with CUDA events.
//   warmup : number of untimed iterations (let the GPU reach steady state)
//   iters  : number of timed iterations
//   launch : a callable (e.g. a lambda) that issues the kernel launch
// Returns the average elapsed time per iteration, in milliseconds.
template <typename F>
float bench(int warmup, int iters, F launch)
{
    cudaEvent_t start, stop;
    BENCH_CUDA_CHECK(cudaEventCreate(&start));
    BENCH_CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup: run but do not time.
    for (int i = 0; i < warmup; ++i) launch();
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Timed region.
    BENCH_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    BENCH_CUDA_CHECK(cudaEventRecord(stop));
    BENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    BENCH_CUDA_CHECK(cudaEventDestroy(start));
    BENCH_CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}
