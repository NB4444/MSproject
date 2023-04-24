#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void kernel(double* A, int total, long const num_runs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double a = 0;
    for(long i =0; i < num_runs; i++) a += 1.234;
    A[index] = a;
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int total = num_sms * num_threads;

    double* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(float));

    kernel<<<num_sms, num_threads>>>(A, total, num_runs);

    cudaFree(A);
}