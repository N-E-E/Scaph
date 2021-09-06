#include "app.cuh"
#include <iostream>
#include "wtime.h"
#include <string.h>
#include <gflags/gflags.h>

using namespace std;
#define MAXPATH 0x7fffffff

DECLARE_int32(src);

template<typename vertex_t,typename index_t,typename value_t>
class sssp:public App<vertex_t,index_t,value_t>
{
public:
    sssp(int argc,char **argv):App<vertex_t,index_t,value_t>(argc,argv){};
    using App<vertex_t,index_t,value_t>::h_value;
    using App<vertex_t,index_t,value_t>::vert_count;
    using App<vertex_t,index_t,value_t>::csr_idx;
    using App<vertex_t,index_t,value_t>::scv;
    virtual void cpu_init();
    virtual void gpu_init();
    virtual bool check();
};

template<typename vertex_t,typename index_t,typename value_t>
void sssp<vertex_t,index_t,value_t>::cpu_init()
{
    for(int index = 0; index < vert_count; index++){
        h_value[index] = MAXPATH; 
    }
    int src = FLAGS_src;
    h_value[src] = 0;
}

template<typename vertex_t,typename index_t,typename value_t>
void sssp<vertex_t,index_t,value_t>::gpu_init()
{

}

template<typename vertex_t,typename index_t,typename value_t>
bool sssp<vertex_t,index_t,value_t>::check()
{
    value_t *r_value = new value_t[(int)vert_count];
    memcpy(r_value,h_value,vert_count*sizeof(value_t));
    value_t *c_value = h_value;
    for(int index = 0; index < vert_count; index++){
        c_value[index] = MAXPATH; 
    }
    c_value[FLAGS_src] = 0;
    vertexSubset t_frontier(vert_count,FLAGS_src);
    while(!t_frontier.isEmpty()){
        vertexSubset output = App<vertex_t,index_t,value_t>::cpu_iteration(t_frontier);
        t_frontier.del();
        t_frontier = output;
    }
    for(int index = 0; index < vert_count; index++){
        if(c_value[index] != r_value[index]){
            cout<<"Error: "<<index<<" r_value: "<<(int)r_value[index]<<" c_value: "<<(int)c_value[index]<<endl;
        return false;
        }
    }
    /*while(1){
        int node;
        cin>>node;
        if(node < 0 || node > vert_count){
            break;
        }
        else
        {
            cout<<"info: "<<node<<" r_value: "<<(int)r_value[node]<<" c_value: "<<(int)c_value[node]<<endl;
        }
    }*/
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
    if(mysssp.check()){
        cout<<"check passed!"<<endl;
    }
    return 0;
}
