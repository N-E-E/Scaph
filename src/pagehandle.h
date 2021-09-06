/*************************************************************************
	> File Name: pagehandle.h
	> Author: 
	> Mail: 
	> Created Time: 2019年01月17日 星期四 09时30分35秒
 ************************************************************************/

#ifndef _PAGEHANDLE_H
#define _PAGEHANDLE_H

#include "page.h"
#include <map>
#include <deque>
#include <vector>

struct  VPstat
{
	bool cached;
	bool shared;
	int  vpid;
	int  dpid;
	int  dpoff;
	int  nodenum;
	int  datanum;
	int  chunknum;
	
	bool IsShared(){
	     return shared;
	} 
};

struct  DPstat
{
	bool shared;

	int  mark;
	int  vpid;
	int  dpid;
	int  maxm;

    VPstat *vps;


	void ReCycle(){
		 shared = false;

	     vpid   = -1;
	     maxm   = 32;
	     mark   = 0;
	}

	void GetMaxm(){
		 maxm   = get_max_continus_zero(mark);
	}
	
	int  AllocChunk(int chunknum){
		 int off = get_fit_continus_zero(mark,chunknum);
		 if(off != -1){
		 	set_continus_one(mark,off,chunknum);
		 	maxm = get_max_continus_zero(mark);
		 }
         return off;
	}

	int FreeChunk(int off,int chunknum){
		set_continus_zero(mark,off,chunknum);
		maxm = get_max_continus_zero(mark);
        return maxm;
	}

	void SetShared(){
		 if(shared == true){
		    return;
		 }
		 else{
		    assert(vpid == -1);
		    shared = true;
		 }
	}

	bool InsertShared(int pid){
		 assert(vps[pid].shared == true);
		 assert(vps[pid].chunknum <= maxm);
         assert(shared == true);

	     int off = AllocChunk(vps[pid].chunknum);
         if(off != -1){
		    vps[pid].dpid  = dpid;
		    vps[pid].dpoff = off;
            return true;
         }
        else{
            return false;
        }
    }

	int RemoveShared(int pid){
		assert(vps[pid].dpid == dpid);
		assert(vps[pid].shared == true);
		assert(vps[pid].cached == true);
		assert(shared == true);

		FreeChunk(vps[pid].dpoff,vps[pid].chunknum);

		vps[pid].dpid   = -1;
		vps[pid].dpoff  = 0;
		vps[pid].cached = false;

		if(maxm == 32){
		   ReCycle();
		}
        return maxm;
	}
	void BoundPage(int pid){
        assert(vps[pid].shared == false);
		assert(vps[pid].dpid  == -1);
		assert(vpid == -1);
        assert(shared == false);

		vps[pid].dpid  = dpid;
		vps[pid].dpoff = 0;

		vpid = pid;
		maxm = 0;
		set_continus_one(mark,0,32);
    }

	void UnboundPage(){
        if(vpid != -1){
           assert(vps[vpid].shared == false);
           assert(shared == false);

	       vps[vpid].cached = false;
		   vps[vpid].dpid   = -1;
		   vps[vpid].dpoff  = 0;
        }
		ReCycle();
	}

	bool IsShared(){
		return shared;
	}
};

struct ResPool
{
    std::set<int> res;
	bool Find(int &id, int required, DPstat *dplist){
        for(std::set<int>::iterator it = res.begin(); it != res.end(); it++){
			if(dplist[*it].maxm >= required){
			   id = *it;
			   return true;
			}
		}
		return false;
	}

	void Init(){
	}

    bool Check(int id){
        return res.find(id) != res.end();
    }
	void Insert(int id){
		res.insert(id);
	}
	
    void Remove(int id){
		res.erase(id);
	}

	bool Read(int &dpid){
	    if(!res.empty()){
	       dpid = *(res.begin());
	       Remove(dpid);
	       return true;
	    }
	    return false;
	}


	void Reset(){
		res.clear();
	}

	bool Empty(){
		 return res.size() == 0;
	}

    int Size(){
        return res.size();
    }
};


struct PriorityQueue
{
	int maxk;
	vector< deque<int> > pri_execq;

	void Init(int _maxk){
	   maxk = _maxk;
	   pri_execq.resize(maxk);
	}

	int GetHigh(int &rank){
	   for(int i = 0; i < maxk; i++){
	   	   if(!pri_execq[i].empty()){
	   	   	   rank = i;
	   	   	   int vpid = pri_execq[i].front();
               pri_execq[i].pop_front();
               return vpid;
	   	   }
	   }
	   return -1;
	}

	void AddTask(int rank, int vpid){
		assert(rank < maxk);
		pri_execq[rank].push_back(vpid); 
	}

    bool Read(int &vpid){
        for(int i = maxk - 1; i > 0; i--){
            if(!pri_execq[i].empty()){
               vpid = pri_execq[i].back();
               pri_execq[i].pop_back();
               return true;
            }
        }
        return false;
    }

    bool Empty(){
        bool flag = false;
        for(int i = 0; i < maxk; i++){
            flag |= !pri_execq[i].empty();
        }
        return !flag;
    }
    

	void Reset(){
		for(int i = 0; i < maxk; i++){
			pri_execq[i].clear();
		}
	}

	
};

template<typename vertex_t>
struct PageHandle
{
	VPstat *vplist;
	DPstat *dplist;
	Page<vertex_t> *pagelist;

	int vplistsize;
	int dplistsize;

    int maxentry;

    bool *cached;
    vertex_t thres;
    map<int,int> hold;

    ResPool  sharedq;
//  ResPool  emptyq;
    ResPool  expectq;

	MFinFout<int> mergeq;
    FinFout<int> emptyq;

	PriorityQueue execq;

	void Handle_Init(int list_size,int pool_size, VPstat *vpstats, DPstat *dpstats, Page<vertex_t> *h_page_list,vertex_t _thres,int _maxentry){
		 vplistsize = list_size;
		 dplistsize = pool_size;

		 vplist = vpstats;
		 dplist = dpstats;
		 pagelist = h_page_list;

         cached = new bool[list_size];
         thres = _thres;
         maxentry = _maxentry;

		 emptyq.Init(list_size+1);
		 sharedq.Init();
		 expectq.Init();
         mergeq.Init(list_size+1);

		 execq.Init(maxentry);

         for(int id = 0; id < list_size; id++){
             cached[id] = id < pool_size;
         }

         for(int id = 0; id < list_size; id++){
             vplist[id].cached = id < pool_size;
             vplist[id].shared = false;
             vplist[id].vpid = id;
             vplist[id].dpid = id < pool_size ? id : -1;
             vplist[id].dpoff = 0;
             vplist[id].nodenum  = h_page_list[id].nodenum;
             vplist[id].chunknum = 32;
         }

         for(int id = 0; id < pool_size; id++){
             dplist[id].shared = false;
             dplist[id].vpid = id;
             dplist[id].dpid = id;
             dplist[id].maxm = 32;
             dplist[id].mark = 0;
             dplist[id].vps  = vplist;
         }
	}

	bool Finished(){
		//return expectq.Empty();
        return expectq.Empty() && hold.empty() && execq.Empty();
	}


	void Queue_Reset(vertex_t *h_page_valid){
        
        emptyq.Reset();
		sharedq.Reset();
		
		expectq.Reset();
		mergeq.Reset();
		
		execq.Reset();


        int numcached, nummerged, numfulled;

        numcached = 0;
	    for(int i = 0; i < dplistsize; i++){
            if(dplist[i].IsShared()){
               emptyq.Write(i);
            }
            else{
               int pid = dplist[i].vpid;
               if(pid != -1 && h_page_valid[pid] > 0){
                  execq.AddTask(0, pid);
                  numcached++;
               }
               else {
                  emptyq.Write(i);
               }
            }
        }

        numfulled = 0;
        for(int i = 0; i < vplistsize; i++){
            if(h_page_valid[i] > thres && !vplist[i].cached){
               vplist[i].shared   = false;
               vplist[i].nodenum  = pagelist[i].nodenum;
               vplist[i].datanum  = 10485760;
               vplist[i].chunknum = 32;
               mergeq.Write(i);
               numfulled++;
            }
        }

        for(int i = 0; i < vplistsize; i++){
        	if(h_page_valid[i] > 0){
        	   expectq.Insert(i);
        	}
        }

        nummerged = expectq.Size() - numcached - numfulled;
        for(int i = 0; i < vplistsize; i++){
            cached[i] = vplist[i].cached;
        }
        // cout<<"total task : "<<expectq.Size()<<" ( "<<numcached<<" , "<<numfulled<<" , "<< nummerged<<" ) "<<endl;
	}

	void ReleaseExecFinished(int vpid){
		 if(!vplist[vpid].IsShared()){
		 	assert(hold.find(vpid) != hold.end());

		 	int rank = hold[vpid];
		 	if(rank < maxentry - 1){
		 	   execq.AddTask(rank+1,vpid);
		 	}
            else{
               int dpid = vplist[vpid].dpid;
               emptyq.Write(dpid);
            }
         }
         else{
            int dpid  = vplist[vpid].dpid;
            dplist[dpid].RemoveShared(vpid);
            if(!dplist[dpid].IsShared()){
               sharedq.Remove(dpid);
               emptyq.Write(dpid);
            }
         }
         hold.erase(vpid);
         expectq.Remove(vpid);
	}

	void ReleaseCopyFinished(int vpid){
	     vplist[vpid].cached = true;
	     execq.AddTask(0,vpid);
	}

	bool GetExecute(int &vpid){
		 int rank;
		 vpid = execq.GetHigh(rank);
		 if(vpid != -1){
		 	hold[vpid] = rank;
		 	return true;
		 }
		 else{
		    return false;
		 }
	}

	bool GetEmpty(int &srcv){
		int dstp,kick;
		if(mergeq.Query(srcv)){
            if(vplist[srcv].IsShared()){
                if(sharedq.Find(dstp,vplist[srcv].chunknum,dplist)){
                   mergeq.Read(srcv);
                   dplist[dstp].InsertShared(srcv);
                   return true;
                }
                else if(emptyq.Read(dstp)){
                   mergeq.Read(srcv);
                   dplist[dstp].UnboundPage();
                   dplist[dstp].SetShared();                          
                   dplist[dstp].InsertShared(srcv);
                   sharedq.Insert(dstp);
                   return true;
                }
                else if(execq.Read(kick)){
                   mergeq.Read(srcv);
                   dstp = vplist[kick].dpid;
                   dplist[dstp].UnboundPage();
                   dplist[dstp].SetShared();
                   dplist[dstp].InsertShared(srcv);
                   sharedq.Insert(dstp);
                   return true;
                }
                else{
                   return false;
                }
            }
            else{
                if(emptyq.Read(dstp)){
                   mergeq.Read(srcv);
                   dplist[dstp].UnboundPage();
                   dplist[dstp].BoundPage(srcv);
                   return true;
                }
                else if(execq.Read(kick)){
                   mergeq.Read(srcv);
                   dstp = vplist[kick].dpid;
                   dplist[dstp].UnboundPage();
                   dplist[dstp].BoundPage(srcv);
                   return true;
                }
                else{
                   return false;
                }
            }
	   }
       return false;
    }
};

#endif
