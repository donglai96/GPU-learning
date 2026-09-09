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

// SAXPY: y = a * x + y
__global__
void saxpy(int N, float a, const float* x, float* y)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < N){
	y[i] = a * x[i] + y[i];
	}

}

int main()
{
    const int N = (1<<24) + 37;
    const size_t bytes = N * sizeof(float);
    const float a = 2.0f;
    srand(42);

    // Allocate host arrays.
    float* x_h = new float[N];
    float* y_h = new float[N];

    // Keep a copy of the original y so we can verify the result.
    float* y_ref = new float[N];

    // Initialize host input arrays.
    for (int i = 0; i < N; ++i) {
        x_h[i]  = rand()/(float)RAND_MAX;
        y_h[i]  = rand()/(float)RAND_MAX;
        y_ref[i] = y_h[i];
    }

    // Device pointers.
    float* x_d = nullptr;
    float* y_d = nullptr;

    // Allocate device memory.
    CUDA_CHECK(cudaMalloc(&x_d, bytes));
    CUDA_CHECK(cudaMalloc(&y_d, bytes));

    // Copy input data from host to device.
    CUDA_CHECK(cudaMemcpy(x_d, x_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d, y_h, bytes, cudaMemcpyHostToDevice));

    // Configure the kernel launch.
    const int threadsPerBlock = 256;
    const int blocks =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    printf(
        "Launching kernel with %d blocks and %d threads per block\n",
        blocks,
        threadsPerBlock
    );

    // Launch the kernel.
    saxpy<<<blocks, threadsPerBlock>>>(N, a, x_d, y_d);

    // Check whether the kernel launch configuration was valid.
    CUDA_CHECK(cudaGetLastError());

    // Wait for the kernel to finish and detect execution errors.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result from device to host.
    CUDA_CHECK(cudaMemcpy(y_h, y_d, bytes, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i) {
        float ref = a * x_h[i] + y_ref[i];
        if (fabsf(y_h[i] - ref) > 1e-5f*fabsf(ref) + 1e-6f) {
            if (bad < 5) printf("mismatch at %d %f %f\n", i, y_h[i], ref);
            ++bad;
        }
    }
    printf(bad ? "%d errors\n" : "no errors\n", bad);

    // Benchmark the kernel. Note saxpy is in-place, so y_d keeps changing
    // across iterations; that is fine because we only care about timing here.
    const int warmup = 10;
    const int iters  = 100;
    float ms = bench(warmup, iters, [&]{
        saxpy<<<blocks, threadsPerBlock>>>(N, a, x_d, y_d);
    });
    CUDA_CHECK(cudaGetLastError());

    // Bytes moved per launch: read x, read y, write y = 3 arrays.
    double gbytes = 3.0 * bytes / 1e9;
    printf("saxpy: %.4f ms/iter, %.1f GB/s\n", ms, gbytes / (ms / 1e3));

    // Free device memory.
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));

    // Free host memory.
    delete[] x_h;
    delete[] y_h;
    delete[] y_ref;

    return EXIT_SUCCESS;
}
