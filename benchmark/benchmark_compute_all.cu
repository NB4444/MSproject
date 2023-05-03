#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void kernel(float* A, double* B, int* C, long const num_runs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    double b = 0;
    int c = 0;
    for(long i =0; i < num_runs; i++) {
        a += 1.234f;
        b += 1.234;
        c += 2;
    }
    A[index] = a;
    B[index] = b;
    C[index] = c;
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int total = num_sms * num_threads;

    float* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(float));

    double* B = NULL;
    cudaMalloc((void **) &B, total * sizeof(double));

    int* C = NULL;
    cudaMalloc((void **) &B, total * sizeof(int));

    kernel<<<num_sms, num_threads>>>(A, B, C, num_runs/3);

    cudaFree(A);
}