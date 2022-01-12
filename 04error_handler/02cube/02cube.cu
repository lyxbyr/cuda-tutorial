#include <stdio.h>
#include <assert.h>
#include "error.h"

const int ARRAY_SIZE = 64;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


__global__ void cubeKernel(float* d_in, float* d_out) {

  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f * f * f;

}


void onDevice(float* h_in, float* h_out) {

  float* d_in;
  float* d_out;

  HANDLER_ERROR_ERR(cudaMalloc((void**)&d_in, ARRAY_BYTES));
  HANDLER_ERROR_ERR(cudaMalloc((void**)&d_out, ARRAY_BYTES));

  HANDLER_ERROR_ERR(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));

  cubeKernel<<<1, ARRAY_SIZE>>>(d_in, d_out);

  HANDLER_ERROR_ERR(cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  HANDLER_ERROR_ERR(cudaFree(d_in));
  HANDLER_ERROR_ERR(cudaFree(d_out));

}

void test(float* h_in, float* h_out) {

  for (int i = 0; i < ARRAY_SIZE; ++i) {
    assert(h_out[i] == h_in[i] * h_in[i] * h_in[i]);
    printf("%f", h_out[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");

  }
  printf("-: successful execution :-\n");
  
}


void onHost() {

  float* h_in;
  float* h_out;

  h_in = (float*)malloc(ARRAY_BYTES);
  h_out = (float*)malloc(ARRAY_BYTES);

  for (int i = 0; i < ARRAY_SIZE; ++i) {
    h_in[i] = float(i);
  }

  onDevice(h_in, h_out);

  test(h_in, h_out);

  free(h_in);
  free(h_out);
}







int main() {


  onHost();
  return 0;
}








