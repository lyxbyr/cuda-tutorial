#include <stdio.h>
#include <assert.h>
#include "Vector.h"

#define N 64
const int ARRAY_BYTES = N * sizeof(int);



__global__ void staticRverseKernel(Vector<int> d_a) {

  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = N - t - 1;
  s[t] = d_a.getElement(t);

  __syncthreads();
  d_a.setElement(t, s[tr]);
}

__global__ void dynamicRverseKernel(Vector<int> d_a) {

  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = N - t - 1;
  s[t] = d_a.getElement(t);

  __syncthreads();
  d_a.setElement(t, s[tr]);
}


void test(Vector<int> d_b, Vector<int> d_c) {

  for (int i = 0; i < N; ++i) {
    assert(d_b.getElement(i) == d_c.getElement(i));
    printf("%d ", d_c.getElement(i));
  }
  printf("\n");

}


void onDevice(Vector<int> d_a, Vector<int> d_b, Vector<int> d_c) {

  Vector<int> d_d;
  d_d.length = N;

  cudaMalloc((void**)&d_d.elements, ARRAY_BYTES);

  cudaMemcpy(d_d.elements, d_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice);

  staticRverseKernel<<<1, N>>>(d_d);

  cudaMemcpy(d_c.elements, d_d.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  test(d_b, d_c);

//====================================================================================================
  cudaMemcpy(d_d.elements, d_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice);

  dynamicRverseKernel<<<1, N, N * sizeof(int)>>>(d_d);

  cudaMemcpy(d_c.elements, d_d.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  test(d_b, d_c);

  cudaFree(d_d.elements);
}



void onHost() {

  Vector<int> h_a, h_b, h_c;
  h_a.length = N;
  h_b.length = N;
  h_c.length = N;

  h_a.elements = (int*)malloc(ARRAY_BYTES);
  h_b.elements = (int*)malloc(ARRAY_BYTES);
  h_c.elements = (int*)malloc(ARRAY_BYTES);

  for (int i = 0; i < N; ++i) {
    h_a.setElement(i, i);
    h_b.setElement(i, N - i -1);
    h_c.setElement(i, 0);
  }

  onDevice(h_a, h_b, h_c);

  printf("-: successful execution :-\n");

  free(h_a.elements);
  free(h_b.elements);
  free(h_c.elements);
}





  











int main() {

  onHost();
  return 0;
}