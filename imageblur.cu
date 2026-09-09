#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

// Radius of the blur window. The window is (2*BLUR_SIZE+1) x (2*BLUR_SIZE+1).
#define BLUR_SIZE 3

// Grayscale image blur: each output pixel is the average of the pixels in a
// square window centered on it. Pixels stored row-major, one byte per pixel.
// out[row*w + col] = average of in[ (row+dy)*w + (col+dx) ] for dx,dy in the
// window, skipping neighbors that fall outside the image.
__global__
void blurKernel(unsigned char* out, const unsigned char* in, int w, int h)
{
    // TODO: write the kernel yourself
    // hints:
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
       if (col < w && row < h) {
           int sum = 0, count = 0;
    //     
    //     loop over  dy in [-BLUR_SIZE, BLUR_SIZE], dx in [-BLUR_SIZE, BLUR_SIZE]:
    	for(int dx = -BLUR_SIZE; dx < (BLUR_SIZE + 1); ++dx ){
           for (int dy= -BLUR_SIZE; dy < (BLUR_SIZE + 1); ++dy){    
	   	int r = row + dy, c = col + dx;
              	if (r >= 0 && r < h && c >= 0 && c < w) { sum += in[r*w+c]; ++count; }
       		}
	}

         	out[row*w + col] = (unsigned char)(sum / count);
       }
}

// CPU reference so we can verify the GPU result.
void blurCPU(unsigned char* out, const unsigned char* in, int w, int h)
{
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int sum = 0, count = 0;
            for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; ++dy) {
                for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; ++dx) {
                    int r = row + dy, c = col + dx;
                    if (r >= 0 && r < h && c >= 0 && c < w) {
                        sum += in[r*w + c];
                        ++count;
                    }
                }
            }
            out[row*w + col] = (unsigned char)(sum / count);
        }
    }
}

int main()
{
    const int W = 4096;
    const int H = 2049;
    const int N = W * H;
    const size_t bytes = (size_t)N * sizeof(unsigned char);
    srand(42);

    // Allocate host arrays.
    unsigned char* in_h  = new unsigned char[N];
    unsigned char* out_h = new unsigned char[N];   // GPU result
    unsigned char* ref_h = new unsigned char[N];   // CPU reference

    // Initialize the input image with random pixel values [0, 255].
    for (int i = 0; i < N; ++i) {
        in_h[i] = (unsigned char)(rand() & 0xFF);
    }

    // Compute the reference on the CPU.
    blurCPU(ref_h, in_h, W, H);

    // Device pointers.
    unsigned char* in_d  = nullptr;
    unsigned char* out_d = nullptr;

    // Allocate device memory.
    CUDA_CHECK(cudaMalloc(&in_d,  bytes));
    CUDA_CHECK(cudaMalloc(&out_d, bytes));

    // Copy input image from host to device.
    CUDA_CHECK(cudaMemcpy(in_d, in_h, bytes, cudaMemcpyHostToDevice));

    // Configure a 2D kernel launch.
    const dim3 threadsPerBlock(16, 16);
    const dim3 blocks(
        (W + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (H + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    printf(
        "Launching kernel with grid (%d, %d) and block (%d, %d)\n",
        blocks.x, blocks.y,
        threadsPerBlock.x, threadsPerBlock.y
    );

    // Launch the kernel.
    blurKernel<<<blocks, threadsPerBlock>>>(out_d, in_d, W, H);

    // Check whether the kernel launch configuration was valid.
    CUDA_CHECK(cudaGetLastError());

    // Wait for the kernel to finish and detect execution errors.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result from device to host.
    CUDA_CHECK(cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost));

    // Verify against the CPU reference (exact match expected).
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        if (out_h[i] != ref_h[i]) {
            if (bad < 5)
                printf("mismatch at %d: gpu=%d cpu=%d\n",
                       i, (int)out_h[i], (int)ref_h[i]);
            ++bad;
        }
    }
    printf(bad ? "%d errors\n" : "no errors\n", bad);

    // Free device memory.
    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));

    // Free host memory.
    delete[] in_h;
    delete[] out_h;
    delete[] ref_h;

    return EXIT_SUCCESS;
}
