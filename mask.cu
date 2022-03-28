#include "stdio.h"
#include "assert.h"
#include <stdint.h>
#define BLOCKS 10
#define THREADS 10
#define ARRAY_SIZE 544*544
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);



__global__ void addKernel(int* d_a, uint8_t* mask) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_a[idx] = 33;
}


void onDevice(uint8_t* h_mask, int* h_a) {
  
  uint8_t *d_mask;
  int* d_a;
  cudaMalloc((void**)&d_mask, ARRAY_BYTES);
  cudaMalloc((void**)&d_a, ARRAY_BYTES);

  cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
  

  addKernel<<<BLOCKS, THREADS >>>(d_a, d_mask);

  cudaMemcpy(h_mask, d_mask, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);


  cudaFree(d_mask);
  cudaFree(d_a);

 
  
}




void onHost() {

  //uint8_t h_mask[544*544];
  uint8_t* h_mask;
  h_mask      = (uint8_t*)malloc(ARRAY_BYTES);

  int *h_a;
  h_a      = (int*)malloc(ARRAY_BYTES);

 
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    h_mask[i] = 0;
    h_a[i] = 1;
  }

  onDevice(h_mask, h_a);

  for (int i = 0; i < ARRAY_SIZE; ++i) {
    printf("%d ",h_a[i]);
  }

 

  printf("-: successful execution :-\n");
  free(h_mask);
  

}


int main() {
  onHost();
  return 0;
}