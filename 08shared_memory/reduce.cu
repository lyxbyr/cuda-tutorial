


__global__ void reduce_global(read *d_x, read *d_y) {

   const int tid = threadIdx.x;
   real *x = d_x + blockDim.x * blockIdx.x;

   for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
      if (tid < offset) {
        x[tid] += x[tid + offset];
      }
      __syncthreads();  
   }

   if (tid == 0) {
    d_y[blockIdx.x] = x[0];
   }
}


__global__ void reduce_shared(read *d_x, read *d_y) {

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int n = bid * blockDim.x + threadIdx.x;

  __shared__ real s_y[128];
  s_y[tid] = (n < N) ? d_x[n] : 0;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
     if (tid < offset) {
       x[tid] += x[tid + offset];
     }
     __syncthreads();  
  }

  if (tid == 0) {
   d_y[bid] = x[0];
  }
}


__global__ void reduce_global(read *d_x, read *d_y) {

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int n = bid * blockDim.x + tid;
  
  extern __shared__ real s_y[];
  s_y[tid] = n < N ? d_x[n] : 0;
  __syncthreads();
  
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
     if (tid < offset) {
       x[tid] += x[tid + offset];
     }
     __syncthreads();  
  }

  if (tid == 0) {
   d_y[bid] = x[0];
  }
}