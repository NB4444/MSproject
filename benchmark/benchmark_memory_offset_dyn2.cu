#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("cuda error \n");
        exit(1);
    }
}

__global__ void kernel(float* A, float* B, long const num_runs, int s, int total) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    long size = (num_runs*s)+index;
    int array_size = total * s;
    for(long i = index; i < size; i+=s) {
        int new_index = i % array_size;
        A[new_index] = B[new_index];
    }
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int s = atoi(argv[4]);
    int total = num_sms * num_threads;


    float* A = NULL;
    checkCudaCall(cudaMalloc((void **) &A, total * s *  sizeof(float)));

    float* B = NULL;
    checkCudaCall(cudaMalloc((void **) &B, total * s * sizeof(float)));

    kernel<<<num_sms, num_threads>>>(A, B, num_runs, s, total);

    cudaFree(A);
}