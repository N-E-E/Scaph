#ifndef __SSSP_KERNEL_CUH__
#define __SSSP_KERNEL_CUH__

template<typename value_t>
__device__ __forceinline__ bool cond(value_t src_v,value_t dst_v,value_t e)
{
   return  dst_v > src_v + e;
}

template<typename value_t>
__device__ __forceinline__ void update(value_t src_v,value_t &dst_v,value_t e)
{
   dst_v = src_v + e;
}

template<typename value_t>
__device__ __forceinline__ bool active(value_t new_value, value_t old_value)
{
  return !(new_value == old_value);
}

template<typename value_t>
__device__ __forceinline__ bool fconv(value_t new_value, value_t old_value,value_t defalut_value)
{
  return (new_value == old_value)&&(old_value != defalut_value);
}

#endif
