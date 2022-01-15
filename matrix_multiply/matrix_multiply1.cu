#include <stdio.h>

/*
存在的问题
  1、矩阵长度限制
      仅有一个block
  2、global memory 读写访问频繁
*/


/*
1、去除长度的限制
  (1)将Pd矩阵拆成tile小块
  (2)把一个tile布置到一个block
  (3)通过threadIdx和blockIdx索引

  eg:
  矩阵: 4 x 4
  TILE_WIDTH = 2
  Block尺寸: 2x2
*/

#define WIDTH 4
#define HEIGHT 4
#define TILE_WIDTH 2


__global__ void matrixMultiplyKernel(float* Md, float* Nd, float* Pd, int Width) {

  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  float pValue = 0;
  for (int k = 0; k < Width; ++k) {
    pValue += Md[row_index * width + k] * Nd[k * width + col_index]]; 
  }
  Pd[row_index * width + col_index] = pValue;
}



void onDevice() {

  dim3 dimGrid(WIDTH / TILE_WIDTH, HEIGHT / TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  matrixMultiplyKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, TILE_WIDTH);

}


int main() {
  

  return 0;
}