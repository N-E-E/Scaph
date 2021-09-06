#include "app.cuh"
#include <iostream>
#include <string.h>
#include <gflags/gflags.h>

using namespace std;

DECLARE_int32(src);
DECLARE_bool(check);

#define MAXPATH 0x7fffffff

template<typename vertex_t,typename value_t>
__global__ void sssp_init(value_t *value, bool *stat,vertex_t vert_count)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = gridDim.x*blockDim.x;
    while(tid < vert_count){
          stat[tid]  = false;
          value[tid] = MAXPATH;
          tid += stride;
    }
}

template<typename vertex_t,typename value_t>
__global__ void setsrc(value_t *value, bool *stat,vertex_t src)
{
    value[src] = 0;
    stat[src] = true;
}

template<typename vertex_t,typename index_t,typename value_t>
class sssp:public App<vertex_t,index_t,value_t>
{
public:
    sssp(int argc,char **argv):App<vertex_t,index_t,value_t>(argc,argv){};
    using App<vertex_t,index_t,value_t>::h_value;
    using App<vertex_t,index_t,value_t>::d_value;
    using App<vertex_t,index_t,value_t>::d_stat;
    using App<vertex_t,index_t,value_t>::vert_count;
    using App<vertex_t,index_t,value_t>::csr_idx;
    using App<vertex_t,index_t,value_t>::num_blks;
    using App<vertex_t,index_t,value_t>::num_thds;
    using App<vertex_t,index_t,value_t>::copystream;
    virtual void gpu_init();
    virtual bool check();
};

template<typename vertex_t,typename index_t,typename value_t>
void sssp<vertex_t,index_t,value_t>::gpu_init()
{
    sssp_init<vertex_t,value_t><<<num_blks,num_thds,0,copystream>>>(d_value,d_stat,vert_count);
    setsrc<vertex_t,value_t><<<1,1,0,copystream>>>(d_value,d_stat,FLAGS_src);
    cudaStreamSynchronize(copystream);
}

template<typename vertex_t,typename index_t,typename value_t>
bool sssp<vertex_t,index_t,value_t>::check()
{
    return true;
}

int main(int argc,char **argv)
{
	typedef unsigned int  vertex_t;
    typedef unsigned int  index_t;
	typedef unsigned int  value_t;

    sssp<vertex_t,index_t,value_t> mysssp(argc,argv);
    mysssp.load_graph();
    double time1,time2;
    time1 = wtime();
	mysssp.run();
    time2 = wtime();
    cout<<"sssp cost "<<time2 - time1<<" seconds "<<endl;
    if(FLAGS_check && mysssp.check()){
        cout<<"check passed!"<<endl;
    }
    return 0;
}
