#include <iostream>
#include "stdio.h"
#include <stdint.h>
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "sys/time.h"
#include "unistd.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void g2(float* A, long long n) {
  long long i = threadIdx.x;
  long long ji = 1024*threadIdx.x + blockIdx.x;
  float x = A[ji];
  long long j=1;
    do {
      x = A[i];
      __syncthreads();
      if((i&j)==0) {
        A[i] = (x + A[i+j])/2.0;
        A[i+j] = (x - A[i+j])/2.0;
      }
      __syncthreads();
      j *= 2;
    } while(j<n);
  //}
}

int main() {
  cudaEvent_t start, stop, startg, stopg;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&startg);
  cudaEventCreate(&stopg);
  float ms0 = 0.0, ms1 = 0.0;
  long long n = 4294967296; // address states
  float* A;
  A = (float*)malloc(n * sizeof(float));
  double size = n * sizeof(float);
  float* d_A;
  for(long long i=0; i<n; i++) {
    A[i] = rand()%255;
  }
  cudaMalloc((void **) &d_A, size);
  cudaEventRecord(startg);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaEventRecord(start);
  g2<<<n/1024, 1024>>>(d_A, n);
  cudaEventRecord(stop);
  cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stopg);
  cudaFree(d_A);
  free(A);
  cudaEventElapsedTime(&ms0, start, stop);
  cudaEventElapsedTime(&ms1, startg, stopg);
	printf("\n\nGPU warp_shuffle Computation time %lf ms\n\n", ms0);
  printf("\n\nGPU warp_shuffle Computation time with memcpy %lf ms\n\n", ms1);
}