#include <stdio.h>
#include <assert.h>


__global__ void addKernel(int a, int b, int* c) {

  *c = a + b;
}

int main() {

  int h_c;
  int* d_c;
  const int c_BYTES = 1 * sizeof(int);

  cudaMalloc((void**)&d_c, c_BYTES);

  addKernel<<<1, 1>>>(2, 7, d_c);

  cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  assert(2 + 7 == h_c);
  printf("-: successful execution :-\n");

  cudaFree(d_c);
  return 0;
}