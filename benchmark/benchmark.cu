#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_RUNS 1000000000

__global__ void kernel(float* A, float* B, int total) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i =0; i < NUM_RUNS; i++) {
        A[index] += B[index];
    }
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int total = num_sms * num_threads;
    // float* a = (float *) calloc(total, sizeof(float));
    // float* b = (float *) malloc(total * sizeof(float));
    // for (int i = 0; i < total; i++) {
    //     b[i] = 1.234f;
    // }

    float* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(float));
    float* B = NULL;
    cudaMalloc((void **) &B, total * sizeof(float));

    // cudaMemcpy(A,  a, total * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(B,  b, total * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<num_sms, num_threads>>>(A, B, total);

    // cudaMemcpy(a,  A, total * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("%f\n", a[0]);

    cudaFree(A);
    cudaFree(B);
}