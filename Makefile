CUCC = nvcc
CUFLAGS = -arch=sm_60 
CUFLAGS += -Xcompiler -fopenmp -lpthread -ltbb
CUFLAGS += -I src/ -I app/ -I ./ 
# -I ./gflags/include/ -lrt ./gflags/lib/libgflags.a 

ifdef DEBUG
	CUFLAGS += -O0 -G -g
else 
	CUFLAGS += -O3
endif


ifdef APPBFS
exe=bfs
bfs: app/bfs.cu app/bfs_kernel.cuh src/* Makefile
	${CUCC} app/bfs.cu -DAPPBFS -o ${exe} ${CUFLAGS}
# else ifdef APPWCC
# exe=wcc
# wcc: app/wcc.cu app/wcc_kernel.cuh src/* Makefile
# 	${CUCC} app/wcc.cu -DAPPWCC -o ${exe} ${CUFLAGS}
# else ifdef APPSSSP
# exe=sssp
# sssp: app/sssp.cu app/sssp_kernel.cuh src/* Makefile
# 	${CUCC} app/sssp.cu -DAPPSSSP -o ${exe} ${CUFLAGS}
endif

.PHONY: all
all : bfs 
# wcc sssp
bfs: app/bfs.cu app/bfs_kernel.cuh src/* Makefile
	${CUCC} app/bfs.cu -DAPPBFS -o bfs-alpha-uk4 ${CUFLAGS}
# wcc: app/wcc.cu app/wcc_kernel.cuh src/* Makefile
# 	${CUCC} app/wcc.cu -DAPPWCC -o wcc ${CUFLAGS}
# sssp: app/sssp.cu app/sssp_kernel.cuh src/* Makefile
# 	${CUCC} app/sssp.cu -DAPPSSSP -o sssp ${CUFLAGS}

.PHONY : clean
clean :
	rm -f bfs wcc sssp
