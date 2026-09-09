#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include "bench.h"

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr,                                                    \
                "CUDA error at %s:%d: %s\n",                               \
                __FILE__,                                                  \
                __LINE__,                                                  \
                cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// 2D matrix add: C = A + B
// Matrices are stored in row-major order, so element (row, col) is at
// index row * W + col.
__global__
void matAdd(const float* A, const float* B, float* C, int W, int H)
{
    // TODO: write the kernel yourself
    // hints:
    //   int col = blockIdx.x * blockDim.x + threadIdx.x;
    //   int row = blockIdx.y * blockDim.y + threadIdx.y;
    //   guard with col < W && row < H, then use idx = row * W + col
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * W + col;
    if (col < W && row < H){
    C[idx] = A[idx] + B[idx];
}
}

int main()
{
    const int W = 4096;
    const int H = 2049;
    const int N = W * H;
    const size_t bytes = (size_t)N * sizeof(float);
    srand(42);

    // Allocate host arrays.
    float* A_h = new float[N];
    float* B_h = new float[N];
    float* C_h = new float[N];

    // Initialize host input arrays.
    for (int i = 0; i < N; ++i) {
        A_h[i] = rand()/(float)RAND_MAX;
        B_h[i] = rand()/(float)RAND_MAX;
    }

    // Device pointers.
    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;

    // Allocate device memory.
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    // Copy input data from host to device.
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    // Configure a 2D kernel launch.
    // block.x maps to columns (width), block.y maps to rows (height).
    const dim3 threadsPerBlock(16, 16);
    const dim3 blocks((W + 16-1)/16,(H + 16 -1)/16);


    printf(
        "Launching kernel with grid (%d, %d) and block (%d, %d)\n",
        blocks.x, blocks.y,
        threadsPerBlock.x, threadsPerBlock.y
    );

    // Launch the kernel.
    matAdd<<<blocks, threadsPerBlock>>>(A_d, B_d, C_d, W, H);

    // Check whether the kernel launch configuration was valid.
    CUDA_CHECK(cudaGetLastError());

    // Wait for the kernel to finish and detect execution errors.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result from device to host.
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i) {
        float ref = A_h[i] + B_h[i];
        if (fabsf(C_h[i] - ref) > 1e-5f*fabsf(ref) + 1e-6f) {
            if (bad < 5) printf("mismatch at %d %f %f\n", i, C_h[i], ref);
            ++bad;
        }
    }
    printf(bad ? "%d errors\n" : "no errors\n", bad);

    // Benchmark the kernel.
    const int warmup = 10;
    const int iters  = 100;
    float ms = bench(warmup, iters, [&]{
        matAdd<<<blocks, threadsPerBlock>>>(A_d, B_d, C_d, W, H);
    });
    CUDA_CHECK(cudaGetLastError());

    // Bytes moved per launch: read A, read B, write C = 3 arrays.
    double gbytes = 3.0 * bytes / 1e9;
    printf("matadd: %.4f ms/iter, %.1f GB/s\n", ms, gbytes / (ms / 1e3));

    // Free device memory.
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    // Free host memory.
    delete[] A_h;
    delete[] B_h;
    delete[] C_h;

    return EXIT_SUCCESS;
}
