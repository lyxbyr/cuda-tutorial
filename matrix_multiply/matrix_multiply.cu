#include <stdio.h>

/*
存在的问题
  1、矩阵长度限制
      仅有一个block
  2、global memory 读写访问频繁
*/

__global__ void matrixMultiplyKernel(float* Md, float* Nd, float* Pd, int Width) {

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float pValue = 0;
  for (int k = 0; k < Width; ++k) {
    float Md_element = Md[ty * Md.width + k];
    float Nd_element = Nd[k * Nd.width + tx];
    pValue += Md_element * Nd_element; 
  }
  Pd[ty * width + tx] = pValue;
}






int main() {
  

  return 0;
}