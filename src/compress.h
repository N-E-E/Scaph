/*************************************************************************
	> File Name: compress.h
	> Author: 
	> Mail: 
	> Created Time: 2019年01月09日 星期三 13时25分21秒
 ************************************************************************/

#ifndef _COMPRESS_H
#define _COMPRESS_H

#include <string.h>
#include "pipe.h"
#include "page.h"
#include "tools.h"
#include "parallel.h"
#include "pagehandle.h"

int compress_stat(unsigned int *Page_src, unsigned int *Page_dst,bool *nodestat, unsigned int left, unsigned int srcnodenum, unsigned int dstnodenum,double &endtime, int &tid)
{
	Edge<unsigned int,unsigned int> *edgesrc = (Edge<unsigned int,unsigned int> *)(Page_src + srcnodenum*3);
	Edge<unsigned int,unsigned int> *edgedst = (Edge<unsigned int,unsigned int> *)(Page_dst + dstnodenum*3);
	Node<unsigned int,unsigned int> *nodesrc = (Node<unsigned int,unsigned int> *)Page_src;
	Node<unsigned int,unsigned int> *nodedst = (Node<unsigned int,unsigned int> *)Page_dst;

	int offset  = 0;
    int counter = 0;
	for(int i = 0; i < srcnodenum; i++){
        unsigned int node = i + left;
		if(nodestat[node]){
		   nodedst[counter].vtx = node;
		   nodedst[counter].idx = offset;
		   nodedst[counter].len = nodesrc[i].len;
		   memcpy(edgedst + offset, edgesrc + nodesrc[i].idx, nodesrc[i].len*sizeof(Edge<unsigned int,unsigned int>));
		   offset += nodesrc[i].len;
		   counter++;
		}
	}
    endtime = wtime();
    return counter;
}

void parallel_compress_stat_async(unsigned int **Pages,bool *nodestat, unsigned int *pagenodes, unsigned int *pagedatas, Page<unsigned int> *pagelist, VPstat *vpstats, bool *cached, int listsize, unsigned int pagesize, unsigned int chunksize, unsigned int thres ,double *endtimes,int *tids,MFinFout<int> &wqueue)
{
    omp_set_num_threads(20);
	parallel_for(int i = 0; i < listsize; i++){
	    if(!cached[i] && pagedatas[i] > 0 && pagedatas[i] <= thres){
		   int counter = compress_stat(Pages[i],Pages[i]+pagesize,nodestat,pagelist[i].left,pagelist[i].nodenum,pagenodes[i],endtimes[i],tids[i]);
		   assert(counter == pagenodes[i]);
           vpstats[i].nodenum  = pagenodes[i];
		   vpstats[i].datanum  = pagedatas[i]; 
		   vpstats[i].chunknum = (pagedatas[i] + chunksize - 1)/chunksize;
		   vpstats[i].shared = true;
		   wqueue.Write(i);
	    }
    }
}

/*void compress_vtx(unsigned int *Page_src, unsigned int *Page_dst, unsigned int *vset,unsigned int left, unsigned int activenodes, unsigned int srcsize, unsigned int dstsize,double &endtime, int &tid)
{
	Edge<unsigned int,unsigned int> *edgesrc = (Edge<unsigned int,unsigned int> *)(Page_src + srcnodenum*3);
	Edge<unsigned int,unsigned int> *edgedst = (Edge<unsigned int,unsigned int> *)(Page_dst + dstnodenum*3);
	Node<unsigned int,unsigned int> *nodesrc = (Node<unsigned int,unsigned int> *)Page_src;
	Node<unsigned int,unsigned int> *nodedst = (Node<unsigned int,unsigned int> *)Page_dst;

	int offset  = 0;
	for(int i = 0; i < activenodes; i++){
        int node = vset[left+i];
		nodedst[i].vtx = nodesrc[node].vtx;
		nodedst[i].idx = offset;
		nodedst[i].len = nodesrc[node].len;
        memcpy(edgedst + offset, edgesrc + nodesrc[node].idx, nodesrc[node].len*sizeof(Edge<unsigned int,unsigned int>));
		offset += nodesrc[node].len;
	}
    endtime = wtime();
}

void parallel_compress_vtx(unsigned int **Pages_src, unsigned int **Pages_dst,unsigned int *vset,unsigned int *validnode,  unsigned int *validsize, Page<unsigned int> *pagelist, int listsize, unsigned int pagesize, unsigned int cmpagesize, double *endtimes, int *tids)
{
	parallel_for(int i = 0; i < listsize; i++){
		if(validsize[i] > 0 && validsize[i] < cmpagesize){
		   compress_vtx(Pages_src[i],Pages_dst[i],vset,pagelist[i].left,validnode[i],pagesize,cmpagesize,endtimes[i],tids[i]);
		}
	}
}*/

#endif
