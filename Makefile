CC       := nvcc
CCFLAGS  := -O2 -arch=sm_20
CUDASRC	 := cuda_test.cu reduce_kernel.cu cudasort.cu
CPPSRC 	 := main.cpp radixsort.cpp cpu_test.cpp timer.cpp
CUDAPATH := ../cscbigstest/cudpp_src_1.1.1
INC 	 := -I $(CUDAPATH)/cudpp/include/ -I $(CUDAPATH)/common/inc/
LIB	 := -L $(CUDAPATH)/lib/ -L $(CUDAPATH)/common/lib/
LDFLAGS  := -lcudpp_x86_64 -lcutil
NCPATH   := -L ../local/netcdf/lib/ -lnetcdf -I ../local/netcdf/include/


all:
	$(CC) $(CCFLAGS) $(CUDASRC) $(CPPSRC) $(NCPATH) $(INC) $(LIB) $(LDFLAGS) 

silent:
	$(CC) $(CCFLAGS) --disable-warnings $(CUDASRC) $(CPPSRC) $(NCPATH) $(INC) $(LIB) $(LDFLAGS)
clean: 
	rm a.out main.cpp~