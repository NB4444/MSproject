#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void kernel(float* A, float* B, long const num_runs, int const n_mem, int const m_compute) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    for(long i =0; i < num_runs; i++) {
        for (int n = 0; n < n_mem; n++) A[index] = B[index];
        for (int m = 0; m < m_compute; m++) a += 1.234f;
    }
    A[index] = a;
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int const n_mem = atol(argv[4]);
    int const m_compute = atol(argv[5]);
    int total = num_sms * num_threads;

    float* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(float));

    float* B = NULL;
    cudaMalloc((void **) &B, total * sizeof(float));

    kernel<<<num_sms, num_threads>>>(A, B, num_runs, n_mem, m_compute);

    cudaFree(A);
}