# CPU-GPU数据传输最小化
## Host<->device 数据传输带宽远低于global memory
## 减少传输
* 中间数据直接在GPU分配，操作，释放
* 有时更适合在GPU进行重复计算－GPU有足够多的运算单元，总线传输的开销比重复计算的开销要大
* 如果没有减少数据传输的话，将CPU代码移植到GPU可能无法提升性能
## 组团传输
* 大块传输好于小块，如果数据小于80KB,性能将受延迟支配
## 内存传输与计算时间重叠
* 双缓存

## Shared Memory
* 比global memory快上百倍
* 可以通过缓存数据减少global memory访存次数
* 线程可以通过shared memory协作
* 用来避免不满足合并条件的访存-读入shared memory重排顺序，从而支持合并寻址

## conclusion
* 有效利用并行性
* 尽可能合并内存访问
* 利用shared memory
* 开发其他存储空间－Texture, Constant
* 减少bank冲突


# 存储器访问优化

## pinnend memory

- 强制让操作系统在物理内存中完成内存申请和释放，这一部分内存不用参与页交换，因此速度比普通的可分页内存快
- 声明这些物理内存只会分配给对应的GPU设备使用，占用了操作系统的可用内存，故可能会影响CPU运行需要的物理内存
- 所以需要合理规划cpu和GPU各自使用的内存，使整个系统达到最优