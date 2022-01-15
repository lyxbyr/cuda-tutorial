#include "stdio.h"
#include "assert.h"

#define BLOCKS 10
#define THREADS 10
#define ARRAY_SIZE 100
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);



__global__ void addKernel(int* d_a, int* d_b, int* d_result) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_result[idx] = d_a[idx] + d_b[idx];
}


void onDevice(int* h_a, int* h_b, int* h_result) {
  
  int *d_a, *d_b, *d_result;

  cudaMalloc((void**)&d_a, ARRAY_BYTES);
  cudaMalloc((void**)&d_b, ARRAY_BYTES);
  cudaMalloc((void**)&d_result, ARRAY_BYTES);

  cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice);

  addKernel<<<BLOCKS, THREADS >>>(d_a, d_b, d_result);

  cudaMemcpy(h_result, d_result, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}




void onHost() {

  int *h_a, *h_b, *h_result;
  h_a      = (int*)malloc(ARRAY_BYTES);
  h_b      = (int*)malloc(ARRAY_BYTES);
  h_result = (int*)malloc(ARRAY_BYTES);

  for (int i = 0; i < ARRAY_SIZE; ++i) {
    h_a[i]      = i;
    h_b[i]      = i + 1;
    h_result[i] = 0;
  }

  onDevice(h_a, h_b, h_result);

  for (int i = 0; i < ARRAY_SIZE; ++i) {
    assert(h_a[i] + h_b[i] == h_result[i]);
    printf("ha=%d, hb=%d, h_result=%d\n", h_a[i], h_b[i], h_result[i]);
  }

  printf("-: successful execution :-\n");
  free(h_a);
  free(h_b);
  free(h_result);
}


int main() {
  onHost();
  return 0;
}