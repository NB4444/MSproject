#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void kernel(float* A, long const num_runs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    for(long i =0; i < num_runs; i++) a += 1.234f;
    A[index] = a;
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int num_streams = atoi(argv[4]);
    int total = num_sms * num_threads;

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) cudaStreamCreate(&(streams[i]));

    float* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(float));

    for (int i = 0; i < num_streams; i++) kernel<<<num_sms, num_threads, 0, streams[i]>>>(A, num_runs);

    cudaFree(A);
}