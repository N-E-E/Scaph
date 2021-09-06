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
#include <fstream>
#include <string>
#include <vector>
#define INFTY int(1<<30)
using namespace std;

#ifdef LARGESIZE
  typedef unsigned long int vertex_t;
  typedef unsigned long int index_t;
#elif MIDIUMSIZE
  typedef unsigned int vertex_t;
  typedef unsigned long int index_t;
#else
  typedef unsigned int vertex_t;
  typedef unsigned int index_t;
#endif

inline off_t fsize(const char *filename) {
    struct stat st; 
    if (stat(filename, &st) == 0)
        return st.st_size;
    return -1; 
}

int main(int argc,char **argv)
{
    char *filename = argv[1];
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
    assert(ret == edge_count);
    fclose(file);

    sprintf(binfile,"%s_csr.idx",filename);
    file = fopen(binfile,"rb");
    index_t* csr_idx = new index_t[vert_count+1];
    ret = fread(csr_idx,sizeof(index_t),vert_count+1,file);
    assert(ret == vert_count+1);
    fclose(file);

    sprintf(binfile,"%s.ligra",filename);
    ofstream myout(binfile);
    myout<<"AdjacencyGraph"<<endl;
    myout<<vert_count<<endl;
    myout<<edge_count<<endl;
    string buffer; 
    for(int index = 0; index < vert_count;index++){
        buffer += to_string(csr_idx[index]) + "\n";
    }
    for(int index = 0; index < edge_count; index++){
        buffer += to_string(csr_ngh[index]) + "\n";
    }
    myout.write(buffer.c_str(),buffer.length());
    myout.close();
    delete csr_ngh;
    delete csr_idx;
    return 0;
}
