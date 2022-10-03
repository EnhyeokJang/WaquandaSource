// (c) Copyright Enhyeok Jang, Yonsei University, Seoul, Korea.
#include <iostream>
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "sys/time.h"
#include "unistd.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"

__global__ void g2(float* A, int n) {
  int i = threadIdx.x + 1024*blockIdx.x;
  float x;
  int j=1;
  if(n<=32) {
    do {
      int i2 = (i&j)/j;
      A[i] = (__shfl_xor_sync(-1,A[i],j)+(1-2.0*i2)*(A[i]))/2.0;
      j *= 2;
    } while(j<n);
  }
  else if(n<=1024) {
    do {
      int i2 = (i&j)/j;
      A[i] = (__shfl_xor_sync(-1,A[i],j)+(1-2.0*i2)*(A[i]))/2.0;
      j *= 2;
    } while(j<32);
    do {
      x = A[i];
      __syncthreads();
      if((i&j)==0) {
        A[i] = (x + A[i+j])/2.0;
        A[i+j] = (x - A[i+j])/2.0;
      }
      __syncthreads();
      j *= 2;
    } while(j<=n);
  }
  else {
    do {
      x = A[i];
      __syncthreads();
      if((i&j)==0) {
        A[i] = (x + A[i+j])/2.0;
        A[i+j] = (x - A[i+j])/2.0;
      }
      __syncthreads();
      j *= 2;
    } while(j<=n);
  }
}

int main()
{
  int paq = 25;// address qubits
  const int a = paq;//# of address qubits
  int ia = 1;//# of LUT
  int pa = pow(2,paq);//# of address states
  int n = ia*pa;
  cudaEvent_t startc, stopc, startg, stopg, startgm, stopgm;
  cudaEventCreate(&startc);
  cudaEventCreate(&stopc);
  cudaEventCreate(&startg);
  cudaEventCreate(&stopg);
  cudaEventCreate(&startgm);
  cudaEventCreate(&stopgm);
  float ms0 = 0.0;
  float ms1 = 0.0;
  float ms2 = 0.0;
  int cnt = 0;
  float pi = 3.14159265359;
  float* A;
  //float* B, *B2;
  int gr = 1;
  int bl = n;
  if(n>1024) {
    gr = n/1024;
    bl = 1024;
  }
  A = (float*)malloc(n * sizeof(float));
  int p[a];
  for(int i=0; i<a; i++) {
    p[i] = 0;
  }
  for(int i=0; i<n; i++) {
    A[i] = rand()%255;
  }
  
  //main calculation
  double size = n * sizeof(float);
  float* d_A;
  cudaMalloc((void **) &d_A, size);
  dim3 dimGrid(gr, 1, 1);
  dim3 dimBlock(bl, 1, 1);
  cudaEventRecord(startgm);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaEventRecord(startg);
  for(int j=1; j<n; j*=2) {
    g2<<<dimGrid, dimBlock>>>(d_B, n);
  }
  cudaEventRecord(stopg);
  cudaEventSynchronize(stopg);
  cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stopgm);
  cudaEventSynchronize(stopgm);
  cudaFree(d_A);
  free(A);
  cudaEventElapsedTime(&ms0, startc, stopc);
  cudaEventDestroy(startc);
  cudaEventDestroy(stopc);
  cudaEventElapsedTime(&ms1, startg, stopg);
  cudaEventDestroy(startg);
  cudaEventDestroy(stopg);
  cudaEventElapsedTime(&ms2, startgm, stopgm);
  cudaEventDestroy(startgm);
  cudaEventDestroy(stopgm);
  printf("\n\n----------------------------------\n");
	printf("CPU Computation time %lf ms\n\n", ms0);
	printf("GPU Computation time %lf ms\n\n", ms1);
	printf("GPU Computation + Host-Device Transfer time %lf ms\n\n", ms2);
}