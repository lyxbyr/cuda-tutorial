#include <stdio.h>

__global__ void kernel() {

}

int main() {

  kernel<<<1, 1>>>();
  printf("Hello Cuda\n");

}