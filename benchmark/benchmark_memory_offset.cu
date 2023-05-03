#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void kernel(float* A, float* B, long const num_runs, int s) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * s;
    for(long i =0; i < num_runs; i++) A[index] = B[index];
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int total = num_sms * num_threads;
    int s = 32;

    float* A = NULL;
    cudaMalloc((void **) &A, total * s *  sizeof(float));

    float* B = NULL;
    cudaMalloc((void **) &B, total * s * sizeof(float));

    kernel<<<num_sms, num_threads>>>(A, B, num_runs, s);

    cudaFree(A);
}