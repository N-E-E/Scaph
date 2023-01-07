#ifndef __APP_CUH__
#define __APP_CUH__

#include <iostream>
#include <pthread.h>
#include <stdio.h>

// #include "gflags/gflags.h"

#include "pipe.h"
#include "page.h"
#include "graph.h"
#include "pagehandle.h"

#include "tools.h"
#include "compress.h"
#include "gpu_engine.cuh"

#include <set>
#include <string>
#include <cassert>
#include <algorithm>
#include <omp.h>
using namespace std;

// int pagesize = 10;
// int max_memory = 4096;
// int multi = 1;
// int num_stream = 5;
// int num_blks = 256;
// int num_thds = 256;
// int threshold = 16;
// bool check = false;
// int tricknum = 0;
// int src = 0;


int FLAGS_pagesize = 10;
int FLAGS_max_memory = 4096;
int FLAGS_multi = 1;
int FLAGS_num_stream = 5;
int FLAGS_num_blks = 256;
int FLAGS_num_thds = 256;

int FLAGS_threshold = 8; // 32 == 100%, 16 == 50%.
bool FLAGS_check = false;
int FLAGS_tricknum = 0;
int FLAGS_src = 0;
double Total_throughput=0;

std::string FLAGS_graphfile = "/home/sxy/lxl/Scaph_Bench/data/data/twitter-2010.edgelist.txt";  // store the graph data
char* graph_file_name_print = "twitter-2010";

//std::string FLAGS_graphfile = "/home/sxy/lxl/Scaph_Bench/uk2007/enterprise/uk-2007-05-edgelist.txt";
//char* graph_file_name_print = "uk-2007";

// DEFINE_int32(pagesize,10,"subgraph pagesize");
// DEFINE_int32(max_memory,4096,"gpu maximum memory size");
// DEFINE_int32(multi,1,"max entry times");
// DEFINE_int32(num_stream,5,"gpu stream number");
// DEFINE_int32(num_blks,256,"kernel mapping blocks");
// DEFINE_int32(num_thds,256,"kernel mapping threads");
// DEFINE_int32(threshold,16,"threshold of compress point");

// DEFINE_bool(check,false,"check result");

// DEFINE_int32(tricknum,0,"trick number");

// DEFINE_int32(src,0,"traversal start point");
// DEFINE_string(graphfile,"/data/dataset/twitter/seraph/unweighted/twitter-2010.edgelist.txt","graphfile path");

template<typename vertex_t, typename index_t, typename value_t>
class App
{
public:
 int64_t  gpu_memsize;
 vertex_t vert_count;
 index_t  edge_count;

 vertex_t pagesize;
 vertex_t chunksize;
 vertex_t threshold;

 cudaStream_t *execstream;
 cudaStream_t  copystream;
 cudaStream_t trickstream;

 int      list_size, pool_size;
 int      num_blks,num_thds,num_stream;

 int      *stream_last_task;
 int       copy_last_task;

 int64_t  h_total_nodes, h_total_datas;

 VPstat   *vpstats;
 DPstat   *dpstats;

 bool     *h_stat,*d_stat;
 value_t  *h_value,*d_value;
 vertex_t *h_degree,*d_degree;

 vertex_t **h_page,**d_page;
 vertex_t *h_page_nodes,*d_page_nodes;
 vertex_t *h_page_datas,*d_page_datas;
 Page<vertex_t> *h_page_list,*d_page_list;

 string    graphfile;
 index_t  *csr_idx;
 vertex_t *csr_ngh;
 vertex_t *csr_deg;
 value_t  *csr_wgh;

 int transferred;
 int fulltrans;
 double *endtimes;
 int  *tids;
 PageHandle<vertex_t> pagehandler;

public:
 App(int argc,char **argv);
 //~App();
 void load_graph();
 void run();
 void iter_init();
 vertex_t gpu_push_batch();
 vertex_t cpu_push_iteration();
 vertex_t cpu_pull_iteration();  
 vertex_t gpu_push_iteration();
 vertex_t gpu_pull_iteration();
 void PageCopy(VPstat &vp);
 virtual void gpu_init() = 0;
 virtual bool check();
 void trick();
};

template<typename vertex_t,typename index_t,typename value_t>
App<vertex_t,index_t,value_t>::App(int argc,char **argv)
{
//  gflags::ParseCommandLineFlags(&argc, &argv, true);

//  int num_stream = atoi(argv[1]);
//  int num_blks = atoi(argv[1]);
 graphfile = FLAGS_graphfile;

 num_stream  = FLAGS_num_stream;
 num_blks    = FLAGS_num_blks;
 num_thds    = FLAGS_num_thds;
 pagesize    = FLAGS_pagesize*1024*1024;
 chunksize   = pagesize/32;
 threshold   = FLAGS_threshold*chunksize;
 gpu_memsize = (int64_t)FLAGS_max_memory*1024*1024;
//  graphfile   = FLAGS_graphfile;
 transferred = 0;
 fulltrans   = 0;
}

/*template<typename vertex_t,typename index_t,typename value_t>
App<vertex_t,index_t,value_t>::~App()
{
  free_graph<index_t,vertex_t,value_t>(csr_idx,csr_ngh,csr_wgh,csr_deg,h_page,h_page_list,list_size);
  cudaFree(d_value);
  cudaFree(d_stat);
  cudaFree(d_degree);
  cudaFree(d_page_valid);
  cudaFree(d_page_list);

  delete stream_last_task;

  cudaFreeHost(h_value);
  cudaFreeHost(h_stat);
  cudaFreeHost(h_page_valid);

  for(int index = 0; index < pool_size; index++){
      cudaFree(d_page[index]);
  }

  delete vpstats;
  delete dpstats;

  vpstats = NULL;
  dpstats = NULL;

  delete endtimes;
  delete tids;
}*/

template<typename vertex_t,typename index_t,typename value_t>
void App<vertex_t,index_t,value_t>::iter_init()
{
 H_ERR(cudaMemcpyAsync(h_stat,d_stat,vert_count*sizeof(bool),cudaMemcpyDeviceToHost,copystream));
 get_page_info<vertex_t><<<num_blks,num_thds,0,execstream[0]>>>(d_stat,d_degree,d_page_nodes,d_page_datas,d_page_list,list_size);
 get_total_info<vertex_t><<<num_blks,num_thds,0,execstream[0]>>>(d_page_nodes,d_page_datas,list_size);
 H_ERR(cudaMemcpyAsync(h_page_nodes,d_page_nodes,list_size*sizeof(vertex_t),cudaMemcpyDeviceToHost,execstream[0]));
 H_ERR(cudaMemcpyAsync(h_page_datas,d_page_datas,list_size*sizeof(vertex_t),cudaMemcpyDeviceToHost,execstream[0]));
 H_ERR(cudaMemcpyFromSymbolAsync(&h_total_nodes,d_total_nodes,sizeof(int64_t),0,cudaMemcpyDeviceToHost,execstream[0]));
 H_ERR(cudaMemcpyFromSymbolAsync(&h_total_datas,d_total_datas,sizeof(int64_t),0,cudaMemcpyDeviceToHost,execstream[0]));
 cudaStreamSynchronize(copystream);
 cudaStreamSynchronize(execstream[0]);
 
 pagehandler.Queue_Reset(h_page_datas);

 for(int i = 0; i < num_stream; i++){
     stream_last_task[i] = -1;
 }
 copy_last_task = -1;
}

template<typename vertex_t,typename index_t,typename value_t>
void App<vertex_t,index_t,value_t>::trick()
{

    int  tricknum = FLAGS_tricknum;
    int *tricksmd;
    H_ERR(cudaMalloc((void **)&tricksmd,4096*sizeof(int)));
    trick_kernel<<<4096,1,0,trickstream>>>(tricksmd,tricknum);
    sleep(5);
}

template<typename vertex_t,typename index_t,typename value_t>
void App<vertex_t,index_t,value_t>::run()
{ 
 gpu_init();
 iter_init();
 int iters = 0;
  double total_times = 0;
 //printf("|=============================Execute===============================|\n");
 
 while(h_total_datas > 0){
  double time1,time2;
  time1 = wtime();


  gpu_push_batch();

  time2 = wtime();

  iter_init();
  /*
  if(iters<=5 |iters==60)
  {
    printf("|                     iter%4d   cost   %.8fs                 |\n", iters, time2 - time1);
  }
  */
  total_times += time2 - time1;
  iters++;
 }
//printf("|                                 ...                               |\n");
 //printf("|=============================Results===============================|\n");
 
 //printf("|                   total iterator times : %-25d|\n", iters);
 double throughputi=edge_count / total_times / 1000.0 / 1000.0/ 1000.0;
 printf("|                         BFS costs     : %.8f seconds        |\n", total_times);
 printf("|                         Throughput    : %.8f GTEPS          |\n", throughputi);
 Total_throughput+=throughputi;
 //printf("|===================================================================|\n");
 


//  cout<<"total iteration times: "<<iters<<endl;
//  cout<<"total transferred: "<<transferred<<" full: "<<fulltrans<<" cached: "<<pool_size<<endl;
 if(FLAGS_check){
    cudaMemcpyAsync(h_value,d_value,vert_count*sizeof(value_t),cudaMemcpyDeviceToHost,copystream);
 }
}

template<typename vertex_t,typename index_t,typename value_t>
void App<vertex_t,index_t,value_t>::load_graph( )
{
  fetch_graph<vertex_t,index_t,value_t>(graphfile.c_str(), \
    vert_count,edge_count, \
    csr_idx,csr_ngh,csr_wgh,csr_deg, \
    h_page,h_page_list,list_size,pagesize);

  printf("|========================Dataset Information========================|\n");
  printf("|                       graph name      : %-26s|\n", graph_file_name_print);
  printf("|                       graph nodes     : %-26d|\n", vert_count);
  printf("|                       graph edges     : %-26u|\n", edge_count);
  

  gpu_memsize -= sizeof(value_t)*vert_count;
  gpu_memsize -= sizeof(bool)*vert_count;
  gpu_memsize -= sizeof(vertex_t)*vert_count;
  gpu_memsize -= sizeof(vertex_t)*list_size;
  gpu_memsize -= sizeof(vertex_t)*list_size;
  gpu_memsize -= sizeof(Page<vertex_t>)*list_size;

  execstream = new cudaStream_t[num_stream];
  for(int index = 0; index < num_stream; index++){
    cudaStreamCreate(&execstream[index]);
  }
  cudaStreamCreate(&trickstream);
  cudaStreamCreate(&copystream);

  stream_last_task = new int[num_stream];

  trick();

  if(gpu_memsize > (int64_t)pagesize*(int64_t)sizeof(vertex_t)){
     H_ERR(cudaMalloc((void **)&d_value,sizeof(value_t)*vert_count));
     H_ERR(cudaMalloc((void **)&d_stat,sizeof(bool)*vert_count));
     H_ERR(cudaMalloc((void **)&d_degree,sizeof(vertex_t)*vert_count));
     H_ERR(cudaMalloc((void **)&d_page_nodes,sizeof(vertex_t)*list_size));
     H_ERR(cudaMalloc((void **)&d_page_datas,sizeof(vertex_t)*list_size));
     H_ERR(cudaMalloc((void **)&d_page_list,sizeof(Page<vertex_t>)*list_size));
     H_ERR(cudaMemcpyAsync(d_degree,csr_deg,vert_count*sizeof(vertex_t),cudaMemcpyHostToDevice,copystream));
     H_ERR(cudaMemcpyAsync(d_page_list,h_page_list,list_size*sizeof(Page<vertex_t>),cudaMemcpyHostToDevice,copystream));
     cudaStreamSynchronize(copystream);
  }
  else{
     cout<<"vertex data out of gpu memory. exit!"<<endl;
     exit(-1);
  }

 endtimes = new double[list_size];
 tids = new int[list_size];

 H_ERR(cudaMallocHost((void **)&h_value,sizeof(value_t)*vert_count));
 H_ERR(cudaMallocHost((void **)&h_stat,sizeof(bool)*vert_count));
 H_ERR(cudaMallocHost((void **)&h_page_nodes,sizeof(vertex_t)*list_size));
 H_ERR(cudaMallocHost((void **)&h_page_datas,sizeof(vertex_t)*list_size));

 pool_size = gpu_memsize/(pagesize*sizeof(vertex_t));
 pool_size = pool_size > list_size ? list_size : pool_size;
 d_page = new vertex_t *[pool_size];

 gpu_memsize -= pool_size*pagesize*sizeof(vertex_t);

//  cout<<"can cache "<<pool_size<<" pages"<<endl;
 for(int index = 0; index < pool_size; index++){
  cudaMalloc((void **)&d_page[index],pagesize*sizeof(vertex_t));
     if(cudaSuccess != cudaGetLastError()){
         pool_size = index;
        // cout<<"actual cache " <<index<<endl;
        break;
     }
  cudaMemcpyAsync(d_page[index],h_page[index],pagesize*sizeof(vertex_t),cudaMemcpyHostToDevice,copystream);
}
cudaStreamSynchronize(copystream);

vpstats = new VPstat[list_size];
dpstats = new DPstat[pool_size];

 pagehandler.Handle_Init(list_size,pool_size,vpstats,dpstats,h_page_list,threshold, FLAGS_multi);
}

template<typename vertex_t,typename index_t,typename value_t>
vertex_t App<vertex_t,index_t,value_t>::cpu_push_iteration()
{
  return 0;
}

template<typename vertex_t,typename index_t,typename value_t>
vertex_t App<vertex_t,index_t,value_t>::cpu_pull_iteration()
{
  return 0;
}

template<typename vertex_t,typename index_t,typename value_t>
vertex_t App<vertex_t,index_t,value_t>::gpu_push_iteration()
{
  return 0;
}

template<typename vertex_t,typename index_t,typename value_t>
vertex_t App<vertex_t,index_t,value_t>::gpu_pull_iteration()
{
  return 0;
}

template<typename vertex_t,typename index_t,typename value_t>
void App<vertex_t,index_t,value_t>::PageCopy(VPstat &vp)
{
  int srcv = vp.vpid;
  int dstp = vp.dpid;
  int size = vp.datanum;
  
  transferred += vp.chunknum;
  fulltrans += vp.chunknum == 32 ? 32 : 0;
  vertex_t *src_addr,*dst_addr;
  if(vp.IsShared()){
     src_addr = h_page[srcv] + pagesize;
     dst_addr = d_page[dstp] + vp.dpoff*chunksize;
     //cout<<"merged page from "<<srcv<<" to "<<dstp<<"."<<vp.dpoff<<" with "<<size<<" chunk "<<vp.chunknum<<endl;
     H_ERR(cudaMemcpyAsync(dst_addr,src_addr,size*sizeof(vertex_t),cudaMemcpyHostToDevice,copystream));
 }
 else{
     src_addr = h_page[srcv];
     dst_addr = d_page[dstp];
     //cout<<"fulled page from "<<srcv<<" to "<<dstp<<endl;
     H_ERR(cudaMemcpyAsync(dst_addr,src_addr,pagesize*sizeof(vertex_t),cudaMemcpyHostToDevice,copystream));
  }
}

template<typename vertex_t,typename index_t,typename value_t>
vertex_t App<vertex_t,index_t,value_t>::gpu_push_batch()
{
  if(h_total_datas == 0){
   return 0;
  }
omp_set_num_threads(2);
omp_set_nested(1);

int exectimes = 0;
int totaltask = pagehandler.expectq.Size();

#pragma omp parallel sections
{
#pragma omp section
{
 parallel_compress_stat_async(h_page,h_stat,h_page_nodes,h_page_datas,h_page_list,vpstats,pagehandler.cached,list_size,pagesize,chunksize,threshold,endtimes,tids,pagehandler.mergeq);
}

#pragma omp section
{
  omp_set_num_threads(1);
  //cout<<"total task number: "<<pagehandler.expectq.size()<<endl;
  while(!pagehandler.Finished()){
    for(int sid = 0; sid < num_stream; sid++){
        int eid;
        if(cudaSuccess == cudaStreamQuery(execstream[sid])){
       if(stream_last_task[sid] != -1){
          //cout<<"finish process page "<<stream_last_task[sid]<<endl;
          pagehandler.ReleaseExecFinished(stream_last_task[sid]);
          stream_last_task[sid] = -1;
       }
       if(pagehandler.GetExecute(eid)){
          vertex_t *d_page_addr  = d_page[vpstats[eid].dpid] + vpstats[eid].dpoff*chunksize;
          vertex_t  d_page_nodes = vpstats[eid].nodenum;
          //cout<<"begin process page "<<eid<<endl; 
          /*if(pagehandler.expectq.Check(eid)){
             cout<<"execute "<<eid<<endl;
          }
          else{
             cout<<"re-execute "<<eid<<endl;
          }*/
          exectimes++;
          gpu_push_csr<vertex_t,index_t,value_t><<<num_blks,num_thds,0,execstream[sid]>>>(d_value,d_stat,d_page_addr,d_page_nodes, eid, d_page_list);
          stream_last_task[sid] = eid;
       }
       else{
          stream_last_task[sid] = -1;
       }
    }
 }

 if(cudaSuccess == cudaStreamQuery(copystream)){
     if(copy_last_task != -1){
        pagehandler.ReleaseCopyFinished(copy_last_task);
        //cout<<"finish copy page "<<copy_last_task<<" to "<<vpstats[copy_last_task].dpid<<"."<<vpstats[copy_last_task].dpoff<<endl;
        copy_last_task = -1;
     }
     int srcv;
     if(pagehandler.GetEmpty(srcv)){
        PageCopy(vpstats[srcv]);
        copy_last_task = srcv;
     }
 }
}
  for(int index = 0; index < num_stream; index++){
      cudaStreamSynchronize(execstream[index]);
  }
  cudaStreamSynchronize(copystream);
}
}
  // cout<<"each task execute "<<(double)exectimes/(totaltask != 0 ? totaltask : 1)<<endl;
  return 1;
}

template<typename vertex_t,typename index_t,typename value_t>
bool App<vertex_t,index_t,value_t>::check()
{
  return true;
}

#endif
