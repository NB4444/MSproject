#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <iostream>


struct KernelData{
  int num_sms;
  int num_threads;
  long num_runs;
  cudaStream_t stream;
  float* A;
  int count;
};

pthread_barrier_t barr;

__global__ void kernel(float* A, long const num_runs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    for(long i =0; i < num_runs; i++) a += 1.234f;
    A[index] = a;
}

void *worker_thread(void *arg)
{
    KernelData* data = static_cast<KernelData*>(arg);
    kernel<<<data->num_sms, data->num_threads>>>(data->A, data->num_runs);
    pthread_barrier_wait(&barr);
}

int main(int argc, char **argv) {
    int const num_sms = atoi(argv[1]);
    int const num_threads = atoi(argv[2]);
    long const num_runs = atol(argv[3]);
    int const num_streams = atoi(argv[4]);
    int const total = num_sms * num_threads;

    float* A = NULL;
    cudaMalloc((void **) &A, total * sizeof(float));

    pthread_barrier_init(&barr, NULL, num_streams);

    pthread_t threads[num_streams];
    KernelData* data[num_streams];
    for (int i = 0; i < num_streams; i++) {
        data[i] = new KernelData();
        data[i]->num_sms = num_sms;
        data[i]->num_threads = num_threads;
        data[i]->num_runs = num_runs;
        data[i]->A = A;
        data[i]->count = i;
        pthread_create(&threads[i], NULL, worker_thread, data[i]);
    }

    for (int i = 0; i < num_streams; i++) {
        pthread_join(threads[i], NULL);
        delete data[i];
    }

    cudaFree(A);
}