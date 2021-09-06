#ifndef	_PAGE_HEADER_
#define	_PAGE_HEADER_

#include <set>
#include <cassert>
#include <algorithm>
#include "pipe.h"
#include "tools.h"

#ifdef APPSSSP
#define WEIGHT
#endif

template <typename index_t, typename vertex_t>
struct GraphInfo
{
    unsigned int  pagesize;
    vertex_t  nodenum;
    index_t   edgenum;
};

template <typename vertex_t>
struct Page
{
    vertex_t left;
    vertex_t right;
    unsigned int nodenum;
    unsigned int edgenum;
};

template <typename index_t, typename vertex_t>
struct Node
{
	vertex_t vtx;
	index_t  idx;
	index_t  len;
};

#ifdef WEIGHT
template <typename vertex_t, typename value_t>
struct Edge
{
	vertex_t ngr;
	value_t  wgh;
};
#else
template <typename vertex_t, typename value_t>
struct Edge
{
	vertex_t ngr;
};
#endif


#endif
