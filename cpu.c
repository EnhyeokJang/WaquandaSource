// (c) Copyright Enhyeok Jang, Yonsei University, Seoul, Korea.
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "sys/time.h"
#include "unistd.h"
#include "omp.h"

int main()
{
  struct timeval  tv;
	double begin, end;
  long long n = 8; // address qubit
  float* A, * B;

  gettimeofday(&tv, NULL);
	begin = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000 ;
  A = (float*)malloc(n * sizeof(float));
  B = (float*)malloc(n * sizeof(float));
  for(long long i=0; i<n; i++) {
    A[i] = rand()%255;
    B[i] = 0;
  }

  float x;
  for(long long j=1; j<=n-1; j*=2) {
    #pragma omp parallel for
    for (long long i = 0; i < n; i++) {
      float x;
      if((i&j)==0) {
      x = B[i];
      B[i] = (x + B[i+j])/2.0;
      B[i+j] = (x - B[i+j])/2.0;
      }
    }
  }
  free(A);
  free(B);
  gettimeofday(&tv, NULL);
	end = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000 ;
  printf("Execution time %f ms\n", (end - begin));
}