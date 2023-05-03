#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void kernel(float* A, float* B, long const num_runs) {
    int index = blockIdx.x + blockDim.x * threadIdx.x;
    for(long i =0; i < num_runs; i++) A[index] = B[index];
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int total = num_sms * num_threads;

    float* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(float));

    float* B = NULL;
    cudaMalloc((void **) &B, total * sizeof(float));

    kernel<<<num_sms, num_threads>>>(A, B, num_runs);

    cudaFree(A);
}