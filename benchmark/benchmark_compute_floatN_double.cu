#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void kernel(float* A, double* B, long const num_runs, int const num_floats) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    double b = 0;
    for(long i =0; i < num_runs; i++) {
        for(int j=0; j < num_floats; j++) {
            a += 1.234f;
        }
        b += 1.234;
    }
    A[index] = a;
    B[index] = b;
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int const num_floats = atoi(argv[4]);
    int total = num_sms * num_threads;

    float* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(float));

    double* B = NULL;
    cudaMalloc((void **) &B, total * sizeof(double));

    kernel<<<num_sms, num_threads>>>(A, B, num_runs, num_floats);

    cudaFree(A);
}