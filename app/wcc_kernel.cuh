#ifndef __WCC_KERNEL_CUH__
#define __WCC_KERNEL_CUH__
#include "gpu_tools.cuh"
template<typename value_t>
__device__ __forceinline__ bool cond(value_t src_v,value_t dst_v)
{
   return  dst_v > src_v ;
}

template<typename value_t>
__device__ __forceinline__ bool update(value_t src_v,value_t &dst_v)
{
    return writeMin<value_t>(src_v,dst_v);
}

template<typename value_t>
__device__ __forceinline__ bool active(value_t new_value, value_t old_value)
{
  return !(new_value == old_value);
}

#endif
