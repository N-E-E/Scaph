#ifndef __PIPE_CUH__
#define __PIPE_CUH__

#include <tbb/mutex.h>

template<typename T>
class MFinFout
{
public:
	int capicity;
	int reader;
	int writer;
	tbb::mutex wd;

	T  *queue;
public:
	MFinFout(){};
	MFinFout(int _capicity){
		capicity = _capicity;
		reader = 0;
		writer = 0;
		queue  = new T[capicity];
	}
	void Init(int _capicity){
		capicity = _capicity;
		reader = 0;
		writer = 0;
		queue  = new T[capicity]; 
	}
	void Reset(){
		reader = 0;
		writer = 0;
	}
	void Write(T s){
		wd.lock();
		queue[writer] = s;
		writer++;
		wd.unlock();
	}
	bool Read(T &s){
		if(reader < writer){
			s = queue[reader];
			reader++;
			return true;
		}
		else{
			return false;
		}
	}
	bool Query(T &s){
		if(reader < writer){
			s = queue[reader];
			return true;
		}
		else{
			return false;
		}
	}
};


template<typename T>
class FinFout
{
public:
	int capicity;
	int reader;
	int writer;

	T  *queue;
public:
	FinFout(){};
	FinFout(int _capicity){
		capicity = _capicity;
		reader = 0;
		writer = 0;
		queue  = new T[capicity];
	}
	void Init(int _capicity){
		capicity = _capicity;
		reader = 0;
		writer = 0;
		queue  = new T[capicity];
	}
	void Reset(){
		reader = 0;
		writer = 0;
	}
	void Write(T s){
		queue[writer] = s;
		writer++;
        writer = writer % capicity;
	}
	bool Read(T &s){
		if(reader != writer){
			s = queue[reader];
			reader++;
            reader = reader % capicity;
			return true;
		}
		else{
			return false;
		}
	}
	bool Query(T &s){
		if(reader != writer){
			s = queue[reader];
			return true;
		}
		else{
			return false;
		}
	}
};

#endif
