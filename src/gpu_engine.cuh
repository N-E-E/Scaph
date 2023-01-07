#include <iostream>
#include <stdio.h>

#include "gpu_tools.cuh"
#include "page.h"
#include "../third-tool-lib/cub/cub/util_ptx.cuh"

#ifdef APPBFS
#include "bfs_kernel.cuh"
#endif

#ifdef APPWCC
#include "wcc_kernel.cuh"
#endif

#ifdef APPPR
#include "pr_kernel.cuh"
#endif

#ifdef APPSSSP
#include "sssp_kernel.cuh"
#define WEIGHT
#endif

__device__ int64_t d_total_nodes;
__device__ int64_t d_total_datas;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", \
        cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define H_ERR( err ) \
  (HandleError( err, __FILE__, __LINE__ ))

template<typename vertex_t,typename index_t,typename value_t>
__global__ void  gpu_push_csr(value_t *value, bool *stat, vertex_t *page, vertex_t nodenum, int pageid, Page<vertex_t> *pglist)
{
    const vertex_t TBSIZE = blockDim.x;
    const vertex_t work_size = nodenum/gridDim.x;
    const vertex_t bound  = nodenum - gridDim.x*(nodenum/gridDim.x);
    const vertex_t bnode  = blockIdx.x < bound ? blockIdx.x*(work_size + 1) : bound*(work_size + 1) + (blockIdx.x - bound)*work_size;
    const vertex_t enode  = blockIdx.x < bound ? bnode + (work_size + 1) : bnode + work_size;
    const vertex_t venode = bnode + (enode - bnode + gridDim.x)/gridDim.x*gridDim.x;
    const vertex_t WPSIZE = 32;
    __shared__ vertex_t tbowner;
    __shared__ vertex_t tbpos;
    __shared__ vertex_t tblen;
    __shared__ value_t  tbdata;
    __shared__ value_t  twvalue[8];

    Node<index_t,vertex_t>     *vlist = (Node<index_t,vertex_t> *)(page);
    Edge<vertex_t,value_t>     *nlist = (Edge<vertex_t,value_t> *)(page + nodenum*sizeof(Node<index_t,vertex_t>)/sizeof(vertex_t));

    int tid = threadIdx.x + bnode;

    while(tid < venode){
        unsigned int len  = 0;
        unsigned int pos  = 0;
        unsigned int node = 0;
        if(tid < enode){
            node = vlist[tid].vtx;
            pos  = vlist[tid].idx;
            len  = vlist[tid].len;
            if(len == 0){
               stat[node] = false;
            }
            if(node == 99732){
               node = 99732;
            }
            len  = stat[node] ? len : 0;  //exist hot
        }
        if(threadIdx.x == 0){
            tbowner = TBSIZE + 1;
        }
        __syncthreads();
        //process CTA node
        while(true){
            if(len >= TBSIZE){
                tbowner = threadIdx.x;
            }
            __syncthreads();
            if(tbowner == TBSIZE + 1){
                break;
            }
            if(tbowner == threadIdx.x){
                tbpos    = pos;
                tblen    = len;
                stat[node] = false;
                tbdata   = value[node];
                pos = 0;
                len = 0;
            }
            __syncthreads();
            unsigned int tpos = tbpos;
            unsigned int tlen = tblen;
            value_t scatter = tbdata;
            if(threadIdx.x == 0){
               tbowner = TBSIZE + 1;
            }
            for(unsigned int ii = threadIdx.x; ii < tlen; ii += TBSIZE){
                Edge<vertex_t,value_t> ng = nlist[tpos+ii];
#ifdef WEIGHT
                if(cond<value_t>(scatter,value[ng.ngr],ng.wgh) && update<value_t>(scatter,value[ng.ngr],ng.wgh)){
                   stat[ng.ngr] = true;
                }
#else
                if(cond<value_t>(scatter,value[ng.ngr]) && update<value_t>(scatter,value[ng.ngr])){
                   stat[ng.ngr] = true;
                }
#endif
            }
            __syncthreads();
        }
        //process warp node
        unsigned int warp_id = threadIdx.x/WPSIZE;
        unsigned int lane_id = cub::LaneId();
        while(__any(len >= WPSIZE))
        {
           int mask = __ballot(len >= WPSIZE ? 1 : 0);
           int leader = __ffs(mask) - 1;
           unsigned int tpos = cub::ShuffleIndex(pos, leader);
           unsigned int tlen = cub::ShuffleIndex(len, leader);
           if(leader == lane_id){
              stat[node] = false;
              twvalue[warp_id] = value[node];
              len = 0;
              pos = 0;
           }
           __sync_warp(1);
           value_t scatter = twvalue[warp_id];
           for(unsigned int ii = lane_id; ii < tlen; ii += WPSIZE){
                Edge<vertex_t,value_t> ng = nlist[tpos+ii];
#ifdef WEIGHT
                if(cond<value_t>(scatter,value[ng.ngr],ng.wgh) && update<value_t>(scatter,value[ng.ngr],ng.wgh)){
                   stat[ng.ngr] = true;
                }
#else
                if(cond<value_t>(scatter,value[ng.ngr]) && update<value_t>(scatter,value[ng.ngr])){
                   stat[ng.ngr] = true;
                }
#endif
           }
        }
        if(len > 0){
           stat[node] = false;
           value_t scatter = value[node];
           for(unsigned int ii = 0; ii < len; ii++){
                Edge<vertex_t,value_t> ng = nlist[pos+ii];
#ifdef WEIGHT
                if(cond<value_t>(scatter,value[ng.ngr],ng.wgh) && update<value_t>(scatter,value[ng.ngr],ng.wgh)){
                   stat[ng.ngr] = true;
                }
#else
                if(cond<value_t>(scatter,value[ng.ngr]) && update<value_t>(scatter,value[ng.ngr])){
                   stat[ng.ngr] = true;
                }
#endif
           }
        }
        tid += blockDim.x;
        __syncthreads();
    }
}

template<typename vertex_t,typename index_t,typename value_t>
__global__ void  gpu_push_slot(value_t *value, bool *stat, vertex_t *page,vertex_t nodenum,vertex_t pagesize, int pageid, Page<vertex_t> *pglist)
{
    const vertex_t TBSIZE = blockDim.x;
    const vertex_t work_size = nodenum/gridDim.x;
    const vertex_t bound  = nodenum - gridDim.x*(nodenum/gridDim.x);
    const vertex_t bnode  = blockIdx.x < bound ? blockIdx.x*(work_size + 1) : bound*(work_size + 1) + (blockIdx.x - bound)*work_size;
    const vertex_t enode  = blockIdx.x < bound ? bnode + (work_size + 1) : bnode + work_size;
    const vertex_t venode = bnode + (enode - bnode + gridDim.x)/gridDim.x*gridDim.x;
    const vertex_t WPSIZE = 32;
    __shared__ vertex_t tbowner;
    __shared__ vertex_t tbpos;
    __shared__ vertex_t tblen;
    __shared__ value_t  tbdata;
    __shared__ value_t  twvalue[8];

    Node<index_t,vertex_t>     *vlist = (Node<index_t,vertex_t> *)(page + pagesize - sizeof(Node<index_t,vertex_t>)/sizeof(vertex_t));
    Edge<vertex_t,value_t>     *nlist = (Edge<vertex_t,value_t> *)(page);

    int tid = threadIdx.x + bnode;

    while(tid < venode){
        unsigned int len  = 0;
        unsigned int pos  = 0;
        unsigned int node = 0;
        if(tid < enode){
            node = vlist[-tid].vtx;
            pos  = vlist[-tid].idx;
            len  = vlist[-tid].len;
            if(len == 0){
               stat[node] = false;
            }
            len  = stat[node] ? len : 0;  //exist hot
        }
        if(threadIdx.x == 0){
            tbowner = TBSIZE + 1;
        }
        __syncthreads();
        //process CTA node
        while(true){
            if(len >= TBSIZE){
                tbowner = threadIdx.x;
            }
            __syncthreads();
            if(tbowner == TBSIZE + 1){
                break;
            }
            if(tbowner == threadIdx.x){
                tbpos    = pos;
                tblen    = len;
                stat[node] = false;
                tbdata   = value[node];
                pos = 0;
                len = 0;
            }
            __syncthreads();
            unsigned int tpos = tbpos;
            unsigned int tlen = tblen;
            value_t scatter = tbdata;
            if(threadIdx.x == 0){
               tbowner = TBSIZE + 1;
            }
            for(unsigned int ii = threadIdx.x; ii < tlen; ii += TBSIZE){
                Edge<vertex_t,value_t> ng = nlist[tpos+ii];
#ifdef WEIGHT
                if(cond<value_t>(scatter,value[ng.ngr],ng.wgh) && update<value_t>(scatter,value[ng.ngr],ng.wgh)){
                   stat[ng.ngr] = true;
                }
#else
                if(cond<value_t>(scatter,value[ng.ngr]) && update<value_t>(scatter,value[ng.ngr])){
                   stat[ng.ngr] = true;
                }
#endif
            }
            __syncthreads();
        }
        //process warp node
        unsigned int warp_id = threadIdx.x/WPSIZE;
        unsigned int lane_id = cub::LaneId();
        while(__any(len >= WPSIZE))
        {
           int mask = __ballot(len >= WPSIZE ? 1 : 0);
           int leader = __ffs(mask) - 1;
           unsigned int tpos = cub::ShuffleIndex(pos, leader);
           unsigned int tlen = cub::ShuffleIndex(len, leader);
           if(leader == lane_id){
              stat[node] = false;
              twvalue[warp_id] = value[node];
              len = 0;
              pos = 0;
           }
           __sync_warp(1);
           value_t scatter = twvalue[warp_id];
           for(unsigned int ii = lane_id; ii < tlen; ii += WPSIZE){
                Edge<vertex_t,value_t> ng = nlist[tpos+ii];
#ifdef WEIGHT
                if(cond<value_t>(scatter,value[ng.ngr],ng.wgh) && update<value_t>(scatter,value[ng.ngr],ng.wgh)){
                   stat[ng.ngr] = true;
                }
#else
                if(cond<value_t>(scatter,value[ng.ngr]) && update<value_t>(scatter,value[ng.ngr])){
                   stat[ng.ngr] = true;
                }
#endif
           }
        }
        if(len > 0){
           stat[node] = false;
           value_t scatter = value[node];
           for(unsigned int ii = 0; ii < len; ii++){
                Edge<vertex_t,value_t> ng = nlist[pos+ii];
#ifdef WEIGHT
                if(cond<value_t>(scatter,value[ng.ngr],ng.wgh) && update<value_t>(scatter,value[ng.ngr],ng.wgh)){
                   stat[ng.ngr] = true;
                }
#else
                if(cond<value_t>(scatter,value[ng.ngr]) && update<value_t>(scatter,value[ng.ngr])){
                   stat[ng.ngr] = true;
                }
#endif
           }
        }
        tid += blockDim.x;
        __syncthreads();
    }
}

template<typename vertex_t>
__global__ void get_page_nodes(bool *stat,vertex_t vert_count,vertex_t *level_bin)
{
    vertex_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    vertex_t lid = threadIdx.x;
    const vertex_t GRNLTY = blockDim.x*gridDim.x;
    __shared__ vertex_t cache[256];
    vertex_t counter = 0;
    while(tid < vert_count){
        if(stat[tid]){
            counter++;
        }
        tid += GRNLTY;
    }
    cache[lid] = counter;
    __syncthreads();
    int index = blockDim.x>>1;
    while(index){
        if(threadIdx.x < index){
            cache[threadIdx.x] += cache[threadIdx.x+index];
        }
        __syncthreads();
        index>>=1;
    }
    if(!threadIdx.x){
        level_bin[blockIdx.x] = cache[0];
    }
}

template<typename vertex_t>
__global__ void get_total_nodes(vertex_t *level_bin)
{
    vertex_t tid = threadIdx.x;
    __shared__ vertex_t cache[256];
    cache[tid] = level_bin[tid];

    __syncthreads();
    int index = blockDim.x>>1;
    while(index){
        if(threadIdx.x < index){
            cache[threadIdx.x] += cache[threadIdx.x+index];
        }
        __syncthreads();
        index>>=1;
    }
    if(!threadIdx.x){
        d_total_nodes = cache[0];
    }
}

template<typename vertex_t>
__global__ void get_page_datas(bool *stat,vertex_t *degree,vertex_t *valid, Page<vertex_t> *pglist, vertex_t listsize)
{
    vertex_t tid = threadIdx.x;
    vertex_t bid = blockIdx.x;
    vertex_t stride = blockDim.x;
    __shared__ vertex_t cache[256];
    while(bid < listsize){
      int left = pglist[bid].left;
      int righ = pglist[bid].right;
      int counter = 0;
      int id;
      for(id = left + tid; id <= righ; id += stride){
#ifdef WEIGHT
          counter += stat[id] * (degree[id]*2 + 3);
#else
          counter += stat[id] * (degree[id] + 3);
#endif
      }
      cache[tid] = counter;
      __syncthreads();
      int index = blockDim.x>>1;
      while(index){
          if(threadIdx.x < index){
             cache[tid] += cache[tid+index]; 
          }
          __syncthreads();
          index>>=1;
      }
      if(!threadIdx.x){
         valid[bid] = cache[0];
      }
      __syncthreads();
      bid += gridDim.x;
    }
}

template<typename vertex_t>
__global__ void get_total_datas(vertex_t *valid, vertex_t listsize)
{
    int64_t total = 0;
    for(int i = 0; i < listsize; i++){
        total += valid[i];
    }
    d_total_datas = total;
}



template<typename vertex_t>
__global__ void get_page_info(bool *stat,vertex_t *degree,vertex_t *nodes, vertex_t *datas, Page<vertex_t> *pglist, vertex_t listsize)
{
    vertex_t tid = threadIdx.x;
    vertex_t bid = blockIdx.x;
    vertex_t stride = blockDim.x;
    __shared__ vertex_t node[256];
    __shared__ vertex_t data[256];

    while(bid < listsize){
      int left = pglist[bid].left;
      int righ = pglist[bid].right;
      int cnodes = 0;
      int cdatas = 0;

      int id;
      for(id = left + tid; id <= righ; id += stride){
          cnodes += stat[id];
#ifdef WEIGHT
          cdatas += stat[id] * (degree[id]*2 + 3);
#else
          cdatas += stat[id] * (degree[id] + 3);
#endif
      }
      node[tid] = cnodes;
      data[tid] = cdatas;

      __syncthreads();
      int index = blockDim.x>>1;
      while(index){
          if(threadIdx.x < index){
             node[tid] += node[tid+index];
             data[tid] += data[tid+index];
          }
          __syncthreads();
          index>>=1;
      }
      if(!threadIdx.x){
         nodes[bid] = node[0];
         datas[bid] = data[0];
      }
      __syncthreads();
      bid += gridDim.x;
    }
}

template<typename vertex_t>
__global__ void get_total_info(vertex_t *nodes, vertex_t *datas, vertex_t listsize)
{
    int64_t totalnodes = 0;
    int64_t totaldatas = 0;

    for(int i = 0; i < listsize; i++){
        totalnodes += nodes[i];
        totaldatas += datas[i];
    }

    d_total_nodes = totalnodes;
    d_total_datas = totaldatas;
}

__device__ uint get_smid() {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

__global__ void trick_kernel(int *sm_d,int used){
    int reg  = 0;  
    int64_t big = 8589934592;
    if (threadIdx.x==0)
       sm_d[blockIdx.x] = get_smid();
    __syncthreads();
    if(sm_d[blockIdx.x] < used){
        while(int64_t(reg) < big){
            reg++;
        }
    }
}

