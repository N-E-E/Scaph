#ifndef GPUTOOLS_HEADER
#define GPUTOOLS_HEADER

template<typename value_t>
__device__ __forceinline__ value_t warpReduce(value_t value)
{
	const int wp_size = 32;
	for(int offset = wp_size/2; offset > 0; offset /= 2){
		value_t temp = __shfl_down(value,offset);
		value = temp < value ? temp : value;
	}
	return value;
}

template<typename value_t>
__device__ __forceinline__ value_t warpFrontReduce(value_t value,int front)
{
	for(int offset = front/2; offset > 0; offset /= 2){
		value_t temp = __shfl_down(value,offset);
		value = temp < value ? temp : value;
	}
	return value;
}

template<typename value_t>
__device__ __forceinline__ value_t blockReduce(value_t value)
{
	const int wp_size = 32;
	__shared__ value_t result[8];
	int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    value = warpReduce(value);     

    if (lane==0) result[wid]=value; 

     __syncthreads();             

     value = (threadIdx.x < blockDim.x / wp_size) ? result[lane] : 0;

     if (wid==0) value = warpFrontReduce(value,8);

     return value;
}

__device__ __forceinline__ void __sync_warp(int predicate)
{
    while((!__all(predicate)))
    {
        ;    
    }
}

template<typename value_t>
__device__ __forceinline__ bool writeMin(value_t src,value_t &dst)
{
    value_t assumed = dst, old;
    while(assumed > src){
      old = atomicCAS(&dst,assumed,src);
      if(old == assumed){
         return true;
      }
      assumed = old;
    }    return false;
}

template<typename value_t>
__device__ __forceinline__ bool writeMax(value_t src,value_t &dst)
{
    value_t assumed = dst, old;
    while(assumed < src){
      old = atomicCAS(&dst,assumed,src);
      if(old == assumed){
         return true;
      }
      assumed = old;
    }    
    return false;
}

#endif
