#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void kernel(int* A, long const num_runs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int a = 1;
    for(long i =0; i < num_runs; i++) a += 2;
    A[index] = a;
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int total = num_sms * num_threads;

    int* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(int));

    kernel<<<num_sms, num_threads>>>(A, num_runs);

    cudaFree(A);
}