# 函数声明
## global和device函数 
* 1、尽量少用递归
* 2、不要用静态变量
* 3、少用malloc
* 4、小心通过指针实现的函数调用


## 向量数据类型
* 通过函数```make_<type name> ``` 构造
```
int2 i2 = make_int2(1, 2);
float4 f4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
```
* 通过.x , .y, .z, and, .w 访问
```
int2 i2 = make_int2(1, 2);
int x = i2.x;
int y = i2.y;
```
## 部分函数列表
* sqrt, rsqrt
* exp, log
* sin, cos, tan, sincos
* asinm acos, atan2
* trunc, ceil, floor

## Intrinsic function内建函数
* 仅面向Device设备端
* 更快，但精度降低
* 以__为前缀，eg:
```
__exp, __log, __sin, __pow, ...
```

# 线程同步
## 块内线程可以同步
* 调用```__syncthreads```创建一个barrier栅栏
* 每个线程在调用点等待块内存所有线程执行到这个地方, 然后所有线程继续执行后续指令
```
Mds[i] = Md[i]
__syncthreads();
func(Mds[i], Mds[i + 1]);
```

# 内存模型
## 局部存储器Local Memory
* 存储于global memory
* 作用域是每个thread
* 用于存储自动变量数组
* 通过常量索引访问

## 共享存储器Shared Memory
* 每个块
* 快速，片上，可读写
* 全速随机访问

## 全局存储器Global Memory
* 长延时(100个周期)
* 片外，可读写
* 随机访问影响性能
* Host主机端可读写

## 常量存储器Constant Memory
* 短延时，高带宽，当所有线程访问同一位置时只读
* 存储于global memory但是有缓存
* Host主机端可读写
* 容量:64kB



|    变量声明　    | 存储器      |   作用域    |  生命期    |
| :-----         | ----:      | :----:     | :----:    |     
|必须是单独的自动变量而不能是数组  | register   |  thread   | kernel |
|自动变量数组                   | local      |thread    | kernel |
|__shared__ int sharedVar     | shared     |  block   | kernel |
|__device__ int globalVar     | gobal      |  grid    | application |
|__constant__ int constantVar | constant   |  grid    | application |

## Global and Constant 变量
### Host可以通过一下函数进行访问
* cudaGetsymbolAddress()
* cudaGeesymbolSize()
* cudaMemcpyToSymbol()
* cudaMemcpyFromSymbol()
### Constants变量必须在函数外声明


