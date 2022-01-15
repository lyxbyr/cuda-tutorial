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

2、每个输入元素被Width个线程读取
  使用shared memory来减少global memory带宽需求
   
  (1)把kernel拆分成多个阶段
    -每个阶段用Md和Nd的子集累加Pd
    -每个阶段有很好的数据局部性
  (2)每个线程
    -读入瓦片内Md和Nd的一个元素存入shared memory
    -在shared memory里进行累加
       

*/




#define WIDTH 4
#define HEIGHT 4
#define TILE_WIDTH 2


__global__ void matrixMultiplyKernel(float* Md, float* Nd, float* Pd, int Width) {

  __shared__float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row_index = by * TILE_WIDTH + ty;
  int col_index = bx * TILE_WIDTH + tx;

  float pValue = 0;
  for (int m = 0; m < Width / TILE_WIDTH; ++m) {
    Mds[ty][tx] = Md[row_index * width + (m * TILE_WIDTH) + tx];
    Nds[ty][tx] = Nd[col_index + (m * TILE_WIDTH + ty) * Width];
    __syncthreads();
  
    for (int k = 0; k < TILE_WIDTH; ++k) {
      pValue += Mds[ty][k] * Nds[k][tx];
      __syncthreads();
    }
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