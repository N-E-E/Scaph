#ifndef __TOOLS_H__
#define __TOOLS_H__
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <sys/time.h>

using namespace std;

// get max continous zero number in a number.
inline int get_max_continus_zero(int mark)
{
	int local = 0;
	int max_continus_zero = 0;
	for(int i = 0; i < 32; i++){
		if(!(mark & 0x1)){
			local++;
		}
		else{
			max_continus_zero = std::max(local,max_continus_zero);
			local = 0;
		}
		mark >>= 1;
	}
	max_continus_zero = std::max(local,max_continus_zero);
	return max_continus_zero;
}

// find the start position of required continuous zero ?
inline int get_fit_continus_zero(int mark, int required)
{
	int local = 0;
	int localoff = 0;

	int fit = get_max_continus_zero(mark);
	int fitoff;

	if(fit < required){
		return -1;
	}

	for(int i = 0; i < 32; i++){
		if(!(mark & 0x1)){
			local++;
		}
		else{
			if(local >= required && local <= fit){
				fit = local;
				fitoff = localoff;
			}
			if(local == required){
				break;
			}
			local = 0;
			localoff = i + 1;
		}
		mark >>= 1;
	}
	if(local >= required && local <= fit){
		fit = local;
		fitoff = localoff;
	}
	return fitoff;
}

// set off-num+1~off as 1
inline int set_continus_one(int &mark, int off, int number)
{
	int mask = 0x1<<off;
	for(int i = 0; i < number; i++){
		mark |= mask;
		mask <<= 1;
	}
	return mark;
}

inline int set_continus_zero(int &mark, int off, int number)
{
	int mask = 0x1<<off;
	for(int i = 0; i < number; i++){
		mark &= (~mask);
		mask <<= 1;
	}
	return mark;
}

template <class ET>
inline bool CAS(ET *ptr, ET oldv, ET newv) {
	if (sizeof(ET) == 1) {
		return __sync_bool_compare_and_swap((bool*)ptr, *((bool*)&oldv), *((bool*)&newv));
	} else if (sizeof(ET) == 4) {
		return __sync_bool_compare_and_swap((int*)ptr, *((int*)&oldv), *((int*)&newv));
	} else if (sizeof(ET) == 8) {
		return __sync_bool_compare_and_swap((long*)ptr, *((long*)&oldv), *((long*)&newv));
	} 
	else {
		std::cout << "CAS bad length : " << sizeof(ET) << std::endl;
		abort();
	}
}

template <class ET>
inline bool writeMin(ET *a, ET b) {
	ET c; bool r=0;
	do c = *a; 
	while (c > b && !(r=CAS(a,c,b)));
	return r;
}

inline double wtime()
{
	double time[2];	
	struct timeval time1;
	gettimeofday(&time1, NULL);

	time[0]=time1.tv_sec;
	time[1]=time1.tv_usec;

	return time[0]+time[1]*1.0e-6;
}


#endif
