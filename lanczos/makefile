all : matrix lanczos
# rand

MKLROOT=/opt/intel/13.1.0.146/composer_xe_2013.2.146/mkl

MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_gnu_thread -lmkl_core -lmkl_rt
CUDALIBS=-lcublas
LIBS=$(MKLLIBS) $(CUDALIBS)

INCLUDE=-I/scratch/santis/bcumming/minlin/release/include -I$(MKLROOT)/include

GPP=/opt/gcc/4.8.1/bin/g++

ARGS=-DUSE_GPU
#ARGS=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

matrix: matrix.cu
	nvcc -arch=sm_35 matrix.cu -o matrix $(INCLUDE)

lanczos: lanczos.cu
	nvcc -arch=sm_35 lanczos.cu mmio.cpp -o lanczos -ccbin=$(GPP) --compiler-options -fopenmp $(INCLUDE) $(LIBS) -DUSE_GPU

gemm: gemm.cu
	nvcc -arch=sm_35 gemm.cu mmio.cpp -o gemm -ccbin=$(GPP) --compiler-options -fopenmp $(INCLUDE) $(LIBS)

driver: driver.cu utilities.h lanczos.h
	nvcc -arch=sm_35 driver.cu mmio.cpp -o driver -ccbin=$(GPP) --compiler-options -fopenmp $(INCLUDE) $(LIBS) $(ARGS)

driver_host: driver.cu utilities.h lanczos.h
	g++ driver.cpp mmio.cpp -o driver -fopenmp $(INCLUDE) $(LIBS) $(ARGS) -I/opt/nvidia/cudatoolkit/5.5.20-1.0501.7700.8.1/include/

test_cast: test_cast.cu utilities.h
	nvcc -arch=sm_35 test_cast.cu mmio.cpp -o test_cast -ccbin=$(GPP) --compiler-options -fopenmp $(INCLUDE) $(LIBS)

test: test.cu utilities.h test.h
	nvcc -arch=sm_35 test.cu -o test -ccbin=$(GPP) --compiler-options -fopenmp $(INCLUDE) $(LIBS)

clean :
	rm -f matrix lanczos gemm driver

