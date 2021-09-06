#ifndef __GRAPH_H__
#define __GRAPH_H__
#include <iostream>
#include <assert.h>
#include <string.h>
#include <cstdio>
#include <stdlib.h>
#include <sys/stat.h>

#include "page.h"

#ifdef  APPSSSP
#define WEIGHT
#endif

inline off_t fsize(const char *filename) {
	struct stat st; 
	if (stat(filename, &st) == 0)
		return st.st_size;
	return -1; 
}


template<typename vertex_t,typename index_t,typename value_t>
bool fetch_graph(const char *graphfile,vertex_t &vert_count,index_t &edge_count,index_t *&csr_idx,vertex_t *&csr_ngh,vertex_t *&csr_wgh,vertex_t *&csr_deg,vertex_t **&h_page,Page<vertex_t> *&h_page_list,int &list_size,vertex_t &pagesize)
{
	typedef Page<vertex_t> PageStruct;
    typedef GraphInfo<index_t,vertex_t> GraphStruct;
    bool ret_flag = false;
	char binfile[256];

    sprintf(binfile,"%s_csr.idx",graphfile);
	vert_count = fsize(binfile)/sizeof(index_t) - 1;

	sprintf(binfile,"%s_csr.ngh",graphfile);
	edge_count = fsize(binfile)/sizeof(vertex_t);

	sprintf(binfile,"%spage.info",graphfile);
	list_size  = fsize(binfile)/sizeof(PageStruct);

	FILE *file=NULL;
    int ret;
 
    
    sprintf(binfile,"%s_csr.ngh",graphfile);

    file = fopen(binfile,"rb");

    csr_ngh = new vertex_t[edge_count];
    ret = fread(csr_ngh,sizeof(vertex_t),edge_count,file);

    assert(ret == edge_count);
    fclose(file);

#ifdef WEIGHT
    sprintf(binfile,"%s_csr.wgh",graphfile);
    file = fopen(binfile,"rb");
    csr_wgh = new vertex_t[edge_count];
    ret = fread(csr_wgh,sizeof(vertex_t),edge_count,file);
    assert(ret == edge_count);
    fclose(file);
#endif

    sprintf(binfile,"%s_csr.deg",graphfile);
    file = fopen(binfile,"rb");
    csr_deg = new vertex_t[vert_count];
    ret = fread(csr_deg,sizeof(vertex_t),vert_count,file);
    assert(ret == vert_count);
    fclose(file);

    sprintf(binfile,"%s_csr.idx",graphfile);
    file = fopen(binfile,"rb");
    csr_idx = new index_t[vert_count+1];
    ret = fread(csr_idx,sizeof(index_t),vert_count+1,file);
    assert(ret == (vert_count+1));
    fclose(file);
   

    sprintf(binfile,"%sgraph.info",graphfile);
    GraphStruct graph;
    file = fopen(binfile,"rb");
    ret = fread(&graph,sizeof(GraphStruct),1,file);
    assert(ret == 1);
    assert(graph.nodenum == vert_count);
    assert(graph.edgenum == edge_count);
    pagesize = graph.pagesize;
    fclose(file);

    sprintf(binfile,"%spage.info",graphfile);
    file = fopen(binfile,"rb");
    h_page_list = new PageStruct[list_size];
    ret = fread(h_page_list,sizeof(PageStruct),list_size,file);
    assert(ret == list_size);
    fclose(file);

    h_page = new vertex_t*[list_size];
    for(int index = 0; index < list_size; index++){
    	sprintf(binfile,"%spage%d",graphfile,index);
    	file = fopen(binfile,"rb");
        cudaMallocHost((void **)&h_page[index],(pagesize + pagesize/2)*sizeof(vertex_t));
        ret = fread(h_page[index],sizeof(vertex_t),pagesize,file);
        assert(ret == pagesize);
        fclose(file);
    }

    ret_flag = true;
    return ret_flag;
}

template<typename vertex_t,typename index_t,typename value_t>
bool free_graph(index_t *&csr_idx,vertex_t *&csr_ngh,vertex_t *&csr_wgh,vertex_t *&csr_deg,vertex_t **&h_page,Page<vertex_t> *&h_page_list,int &list_size)
{
    

    delete csr_idx;
#ifdef WEIGHT
    delete csr_wgh;
#endif
    delete csr_deg;
    delete csr_ngh;

    for(int index = 0; index < list_size; index++){
        cudaFreeHost(h_page[index]);
    }
    delete h_page_list;

    return true;
}
#endif
