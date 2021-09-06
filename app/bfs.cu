#include <iostream>
#include <string.h>
#include <vector>
// #include <gflags/gflags.h>
#include "app.cuh"
#include "tools.h"

using namespace std;

extern int FLAGS_src;
extern bool FLAGS_check;
extern int FLAGS_threshold;
extern double Total_throughput;

// DECLARE_int32(src);
// DECLARE_bool(check);

#define MAXDEPTH 63

template<typename vertex_t,typename value_t>
__global__ void bfs_init(value_t *value, bool *stat,vertex_t vert_count)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = gridDim.x*blockDim.x;
    while(tid < vert_count){
          stat[tid]  = false;
          value[tid] = MAXDEPTH;
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
class bfs:public App<vertex_t,index_t,value_t>
{
public:
    bfs(int argc,char **argv):App<vertex_t,index_t,value_t>(argc,argv){};
    using App<vertex_t,index_t,value_t>::h_stat;
    using App<vertex_t,index_t,value_t>::h_value;
    using App<vertex_t,index_t,value_t>::d_stat;
    using App<vertex_t,index_t,value_t>::d_value;
    using App<vertex_t,index_t,value_t>::num_blks;
    using App<vertex_t,index_t,value_t>::num_thds;
    using App<vertex_t,index_t,value_t>::copystream;
    using App<vertex_t,index_t,value_t>::vert_count;
    using App<vertex_t,index_t,value_t>::csr_idx;
    using App<vertex_t,index_t,value_t>::csr_ngh;
    virtual bool check();
    virtual void gpu_init();
};

template<typename vertex_t,typename index_t,typename value_t>
void bfs<vertex_t,index_t,value_t>::gpu_init()
{
    bfs_init<vertex_t,value_t><<<num_blks,num_thds,0,copystream>>>(d_value,d_stat,vert_count);
    setsrc<vertex_t,value_t><<<1,1,0,copystream>>>(d_value,d_stat,FLAGS_src);
    // setsrc<vertex_t,value_t><<<1,1,0,copystream>>>(d_value,d_stat,src);
    cudaStreamSynchronize(copystream);
}

template<typename vertex_t,typename index_t,typename value_t>
bool bfs<vertex_t,index_t,value_t>::check()
{
    vector<vertex_t> cur;
    vector<vertex_t> next;

    int sizet = vert_count;
   
    value_t *vertex_value = new value_t[sizet];

    for(vertex_t id = 0; id < vert_count; id++){
        vertex_value[id] = MAXDEPTH;
    }

    cur.push_back(FLAGS_src);
    vertex_value[FLAGS_src] = 0;
    // cur.push_back(src);
    // vertex_value[src] = 0;

    while(!cur.empty()){
        for(int id = 0; id < cur.size(); id++){
            vertex_t vid = cur[id];
            index_t vbg = csr_idx[vid];
            index_t ved = csr_idx[vid+1];
            for(index_t nid = vbg; nid < ved; nid++){
                vertex_t ng = csr_ngh[nid];
                if(vertex_value[vid] + 1 < vertex_value[ng]){
                   vertex_value[ng] = vertex_value[vid] + 1;
                   next.push_back(ng);
                }
            }
        }
        cur.swap(next);
        next.clear();
    }
    int errors = 0;
    for(vertex_t vid = 0; vid < vert_count; vid++){
        if(h_value[vid] != vertex_value[vid]){
           if(errors < 10){
              cout<<"vid: "<<vid<<" gpu: "<<h_value[vid]<<" cpu: "<<vertex_value[vid]<<endl;
              errors++;
           }
           else{
              errors++;
           }
        }
    }
    delete vertex_value;
    cout<<"total errors: "<<errors<<endl;
    return true;
}

int main(int argc,char **argv)
{
	typedef unsigned int  vertex_t;
    typedef unsigned int  index_t;
	typedef unsigned int  value_t;

    printf("|=============================Execute===============================|\n");
    
    
       
    FLAGS_threshold = 3;
    bfs<vertex_t,index_t,value_t> mybfs(argc,argv);
    mybfs.load_graph(); // 50% is original
    double time1,time2,avg_time12;
    
  
    int i;
    time1 = wtime();
	for(i=0;i<10;i++)
    {
        mybfs.run();
    }
    time2 = wtime();
    avg_time12=(time2-time1)/10;
    printf("|========================Results(alpha=37.50%)======================|\n");
    printf("|      Avgrage(10 Rounds) BFS costs     : %.8f seconds       |\n", avg_time12);
    printf("|      Avgrage(10 Rounds) Throughput    : %.8f GTEPS          |\n", Total_throughput/10);
    
    
    printf("|========================Results(alpha=37.50%)======================|\n");
    FLAGS_threshold = 3; 
    bfs<vertex_t,index_t,value_t> mybfs1(argc,argv);
    mybfs1.load_graph();
	mybfs1.run();

/*

    printf("|========================Results(alpha=25.00%)======================|\n");
    FLAGS_threshold = 2; 
    bfs<vertex_t,index_t,value_t> mybfs2(argc,argv);
    mybfs2.load_graph();
	mybfs2.run();    

  
    printf("|========================Results(alpha=37.50%)======================|\n");
    FLAGS_threshold = 3; 
    bfs<vertex_t,index_t,value_t> mybfs3(argc,argv);
    mybfs3.load_graph();   
	mybfs3.run();  

 	
    printf("|========================Results(alpha=50.00%)======================|\n");
    FLAGS_threshold = 4; 
    bfs<vertex_t,index_t,value_t> mybfs4(argc,argv);
    mybfs4.load_graph();
    mybfs4.run();  

  
    printf("|========================Results(alpha=62.50%)======================|\n");
    FLAGS_threshold = 5; 
    bfs<vertex_t,index_t,value_t> mybfs5(argc,argv);
    mybfs5.load_graph();
	mybfs5.run(); 
    

     printf("|========================Results(alpha=75.00%)======================|\n");
    FLAGS_threshold = 6; 
    bfs<vertex_t,index_t,value_t> mybfs6(argc,argv);
    mybfs6.load_graph();
	mybfs6.run(); 



    printf("|========================Results(alpha=87.50%)======================|\n");
    FLAGS_threshold = 7; 
    bfs<vertex_t,index_t,value_t> mybfs7(argc,argv);
    mybfs7.load_graph();
	mybfs7.run(); 


    printf("|========================Results(alpha=100.00%)=====================|\n");
    FLAGS_threshold = 8; 
    bfs<vertex_t,index_t,value_t> mybfs8(argc,argv);
    mybfs8.load_graph();
	mybfs8.run(); 
  */

    printf("|===================================================================|\n");
    
    /*
    // cout<<"BFS cost "<< time2 - time1<<" seconds "<<endl;
    if(FLAGS_check && mybfs.check()){
    // if(check && mybfs.check()){
        cout<<"check passed!"<<endl;
    }
    */
    return 0;
}
