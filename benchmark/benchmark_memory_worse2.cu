#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <numeric>
#include <algorithm>

__global__ void kernel(float* A, float* B, int* index, long const num_runs) {
    int index_num = blockIdx.x * blockDim.x + threadIdx.x;
    int run = num_runs + index_num;
    for(long i = index_num; i < run; i++) {
        int l = index[i];
        A[l] = B[l];
    }
}

int main(int argc, char **argv) {
    int num_sms = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int total = num_sms * num_threads;
    int size = total + num_runs;

    int *a = (int *) malloc(sizeof(int) * size);
    std::iota(a, a + size, 1);
    std::random_shuffle(a, a + size);
    // for(long i = 0; i < total; i++) {
    //     printf("%d\n", a[i]);
    // }
    int* index;
    cudaMalloc((void **) &index, total * sizeof(int));
    cudaMemcpy(index,  a, total * sizeof(int), cudaMemcpyHostToDevice);

    float* A = NULL;
    cudaMalloc((void **) &A, (total + num_runs) * sizeof(float));

    float* B = NULL;
    cudaMalloc((void **) &B, (total + num_runs) * sizeof(float));


    kernel<<<num_sms, num_threads>>>(A, B, index, num_runs);

    // cudaMemcpy(a,  index, total * sizeof(int), cudaMemcpyDeviceToHost);
    // for(long i = 0; i < total; i++) {
    //     printf("%d\n", a[i]);
    // }

    cudaFree(A);
}