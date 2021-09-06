/*
 * Copyright 2016 The George Washington University
 * Written by Hang Liu 
 * Directed by Prof. Howie Huang
 *
 * https://www.seas.gwu.edu/~howie/
 * Contact: iheartgraph@gmail.com
 *
 * 
 * Please cite the following paper:
 * 
 * Hang Liu and H. Howie Huang. 2015. Enterprise: breadth-first graph traversal on GPUs. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '15). ACM, New York, NY, USA, Article 68 , 12 pages. DOI: http://dx.doi.org/10.1145/2807591.2807594
 
 *
 * This file is part of Enterprise.
 *
 * Enterprise is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Enterprise is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Enterprise.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <vector>
#define INFTY int(1<<30)
using namespace std;

#ifdef LARGESIZE
  typedef unsigned long int vertex_t;
  typedef unsigned long int index_t;
  typedef unsigned int value_t;
#elif MIDIUMSIZE
  typedef unsigned int vertex_t;
  typedef unsigned long int index_t;
  typedef unsigned int value_t;
#else
  typedef unsigned int vertex_t;
  typedef unsigned int index_t;
  typedef unsigned int value_t;
#endif


struct Node
{
  vertex_t vtx;
  index_t  idx;
  index_t  len;
};

struct Edge
{
  vertex_t ngr;
  value_t  wgh;
};


struct Page
{
    vertex_t left;
    vertex_t right;
    unsigned int nodenum;
    unsigned int edgenum;
};

struct GraphInfo
{
    unsigned int pagesize;
    unsigned int nodenum;
    unsigned long edgenum;
};

inline off_t fsize(const char *filename) {
    struct stat st; 
    if (stat(filename, &st) == 0)
        return st.st_size;
    return -1; 
}

vertex_t *page_malloc(char *filename,int id,int &fd,index_t pagesize){
    char buffer[80];
	sprintf(buffer,"%spage%d",filename,id);
	fd = open(buffer,O_CREAT|O_RDWR,00666);
	ftruncate(fd, pagesize*sizeof(vertex_t));
	vertex_t* adj = (vertex_t*)mmap(NULL,pagesize*sizeof(vertex_t),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
	assert(adj != MAP_FAILED);
	return adj;
}

bool bsearch(vertex_t left,vertex_t &right,index_t *pos,bool wflag,index_t pagesize)
{
	int bleft  = left;
	int bright = right;
	int bmid   = (left+right)/2;

	if(wflag){
	      if(((pos[bright+1] - pos[bleft])*2 + (bright - left + 1)*3 ) < pagesize){
			       return true;
		    } 
        while(1){
          bool fmid  = ((pos[bmid+1] - pos[left])*2 + (bmid - left + 1)*3 )<pagesize;
          bool fmidr = ((pos[bmid+2] - pos[left])*2 + (bmid - left + 2)*3 )<pagesize;
          if(fmid && fmidr){
        	   bleft = bmid + 1;
             bmid = (bleft + bright)/2;
          }
          else if(!fmid && !fmidr ) {
          	 bright = bmid - 1;
        	   bmid = (bleft + bright)/2;
          }
          else if(fmid && !fmidr){
             right = bmid;
             break;
          }
          else {
             cout<<"surprise"<<endl;
             return false;
          }
        }
    }
    else{
    	if(((pos[bright+1] - pos[bleft]) + (bright - left + 1)*3 ) < pagesize){
			    return true;
		}
    	while(1){
          bool fmid  = ((pos[bmid+1] - pos[left]) + (bmid - left + 1)*3 )<pagesize;
          bool fmidr = ((pos[bmid+2] - pos[left]) + (bmid - left + 2)*3 )<pagesize;
          if(fmid && fmidr){
             bleft = bmid + 1;
             bmid = (bleft + bright)/2;
          }
          else if(!fmid && !fmidr ) {
             bright = bmid - 1;
        	   bmid = (bleft + bright)/2;
          }
          else if(fmid && !fmidr){
             right = bmid ;
             break;
          }
          else {
             cout<<"surprise"<<endl;
             return false;
          }
        }
    }
    return true;
}

void save_page_info(char *filename,vector <Page>  &page_list)
{

    char buffer[80];
    if(page_list.size() > 0){
       sprintf(buffer,"%spage.info",filename);
       int  sfd = open(buffer,O_CREAT|O_RDWR,00666);
       ftruncate(sfd, page_list.size()*sizeof(Page));
       Page *s_page = (Page *)mmap(NULL,page_list.size()*sizeof(Page),PROT_READ|PROT_WRITE,MAP_SHARED,sfd,0);
       assert(s_page != MAP_FAILED);
       memcpy(s_page,(char *)&page_list[0],page_list.size()*sizeof(Page));
       munmap(s_page,page_list.size()*sizeof(Page));
       close(sfd);
    }
}

void save_graph_info(char *filename,vertex_t vert_count,index_t edge_count,vertex_t pagesize)
{
    char buffer[80];
    sprintf(buffer,"%sgraph.info",filename);
    int  sfd = open(buffer,O_CREAT|O_RDWR,00666);
    ftruncate(sfd, sizeof(GraphInfo));
    GraphInfo *s_graph = (GraphInfo *)mmap(NULL,sizeof(GraphInfo),PROT_READ|PROT_WRITE,MAP_SHARED,sfd,0);
    assert(s_graph != MAP_FAILED);
    s_graph->nodenum  = vert_count;
    s_graph->edgenum  = edge_count;
    s_graph->pagesize = pagesize;
    munmap(s_graph,sizeof(GraphInfo));
    cout<<"size of GraphInfo: "<<sizeof(GraphInfo)<<endl;
    close(sfd);
}

void page_gen(char *filename, index_t *idx, vertex_t *csr,vertex_t *wgh,vertex_t vert_count,index_t edge_count,bool wflag,index_t pagesize)
{
	int fd = 0;
	int id = 0;
	vector<Page> page_list;
	Page cur_page;
	vertex_t left = 0;
    while(left < vert_count){
       vertex_t *data = (vertex_t *)page_malloc(filename,id++,fd,pagesize);
       vertex_t right = left + pagesize;
       right = right > vert_count ? vert_count - 1 : right;
       bsearch(left,right,idx,wflag,pagesize);

       int nodenum = right-left + 1;
       int edgenum = idx[right+1] - idx[left];
       cout<<"left: "<<left<<" right: "<<right<<endl;
       cout<<"edgenum: "<<edgenum<<endl;
       cout<<"nodenum: "<<nodenum<<endl<<endl;
       cur_page.left = left;
       cur_page.right = right;
       cur_page.nodenum = nodenum;
       cur_page.edgenum = edgenum;
       page_list.push_back(cur_page);

       Node *lnode = new Node[nodenum];
       for(vertex_t id = left; id <= right; id++){
           int offset =  id - left;
           lnode[offset].vtx = id;
           lnode[offset].idx = idx[id] - idx[left];
           lnode[offset].len = idx[id+1] - idx[id]; 
       }

       if(wflag){
          Edge *lEdge = new Edge[edgenum];
          for(int eid = 0; eid < edgenum; eid++){
              lEdge[eid].ngr = csr[idx[left]+eid];
              lEdge[eid].wgh = wgh[idx[left]+eid];
          }
          memcpy(data,lnode,nodenum*sizeof(Node));
          memcpy(data + nodenum*sizeof(Node)/sizeof(vertex_t), lEdge, edgenum*sizeof(Edge));
          delete lEdge;
       }
       else{
          memcpy(data, lnode, nodenum*sizeof(Node));
          memcpy(data + nodenum*sizeof(Node)/sizeof(vertex_t), csr + idx[left], edgenum*sizeof(vertex_t));
       }
       delete lnode;
       munmap(data,sizeof(vertex_t)*pagesize);
       close(fd);
       left = right + 1;
    }
    cout<<"node: "<<vert_count<<endl;
    cout<<"edge: "<<edge_count<<endl;
    save_page_info(filename,page_list);
    save_graph_info(filename,vert_count,edge_count,pagesize);
}

int main(int argc,char **argv)
{
    if(argc < 4){
       cout<<"usage : ./pagegenslot filename pagesize(MB Vertex) w/u(weighted)"<<endl;
       exit(0);
    }
    char *filename = argv[1];
    size_t pagesize = atol(argv[2])*1024*1024;
    bool wflag = (argv[3][0]=='w');
    char binfile[256];

    sprintf(binfile,"%s_csr.ngh",filename);
    size_t edge_count = fsize(binfile)/sizeof(vertex_t);
    sprintf(binfile,"%s_csr.deg",filename);
    size_t vert_count = fsize(binfile)/sizeof(vertex_t);

    FILE *file=NULL;
    size_t ret;

    sprintf(binfile,"%s_csr.ngh",filename);
    file = fopen(binfile,"rb");
    vertex_t* csr_ngh = new vertex_t[edge_count];
    ret = fread(csr_ngh,sizeof(vertex_t),edge_count,file);
    //free(csr_ngh);
    assert(ret == edge_count);
    fclose(file);

    sprintf(binfile,"%s_csr.wgh",filename);
    file = fopen(binfile,"rb");
    vertex_t* csr_wgh = new vertex_t[edge_count];
    ret = fread(csr_wgh,sizeof(vertex_t),edge_count,file);
    //free(csr_wgh);
    assert(ret == edge_count);
    fclose(file);

    sprintf(binfile,"%s_csr.deg",filename);
    file = fopen(binfile,"rb");
    vertex_t* csr_deg = new vertex_t[vert_count];
    ret = fread(csr_deg,sizeof(vertex_t),vert_count,file);
    //free(csr_deg);
    assert(ret == vert_count);
    fclose(file);


    sprintf(binfile,"%s_csr.idx",filename);
    file = fopen(binfile,"rb");
    index_t* csr_idx = new index_t[vert_count+1];
    ret = fread(csr_idx,sizeof(index_t),vert_count+1,file);
    //free(csr_idx);
    assert(ret == vert_count+1);
    fclose(file);
    
    page_gen(filename,csr_idx,csr_ngh,csr_wgh,vert_count,edge_count,wflag,pagesize);
    delete csr_ngh;
    delete csr_wgh;
    delete csr_idx;
    delete csr_deg;
    return 0;
}
