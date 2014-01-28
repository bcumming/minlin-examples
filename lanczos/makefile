all : driver driver_gcc driver_eigen

# set up system-specific settings
# MONCH
SCRATCH=/mnt/lnec/bcumming
EIGENROOT=/users/wsawyer/Software/USI_COMPRESSION/usi_compression/cxx/eigen
MINLINROOT=$(SCRATCH)/minlin/include
MKLROOT=/apps/monch/intel/14.0.1/mkl/
GPP=/apps/monch/gcc/4.8.1/bin/g++
BOOSTROOT=/apps/monch/boost/1.54.0/openmpi-1.7.2_icc-14.0/include
ARGS=-g -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP

# greina
MINLINROOT=/users/bcumming/github/minlin/include
MKLROOT=/cm/shared/apps/intel/composer_xe/2011_sp1.12.361/mkl
GPP=/cm/shared/apps/gcc/4.7.1/bin/g++
BOOSTROOT=/users/bcumming/repos/boost_1_49_0
ARGS=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

# SANTIS
EIGENROOT=/users/wsawyer/Software/USI_COMPRESSION/usi_compression/cxx/eigen
MINLINROOT=/scratch/santis/bcumming/github/minlin/include
MKLROOT=/opt/intel/13.1.0.146/composer_xe_2013.2.146/mkl
GPP=/opt/gcc/4.8.1/bin/g++
BOOSTROOT=/apps/daint/boost/1.54.0/gnu_473/include
ARGS=-g
#ARGS=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_gnu_thread -lmkl_core -lmkl_rt
CUDALIBS=-lcublas
LIBS=$(MKLLIBS)

INCLUDE=-I$(MINLINROOT) -I$(MKLROOT)/include -I$(BOOSTROOT) -I$(EIGENROOT)

driver: driver.cu utilities.h tge_wrappers.h lanczos.h
	nvcc -arch=sm_35 -O3 driver.cu mmio.cpp -o driver -ccbin=$(GPP) --compiler-options "-fopenmp -O3" $(INCLUDE) $(LIBS) $(ARGS) -DUSE_GPU $(CUDALIBS)

driver_host: driver.cu utilities.h tge_wrappers.h lanczos.h
	nvcc -arch=sm_35 driver.cu mmio.cpp -o driver_host -ccbin=$(GPP) --compiler-options "-fopenmp -O3" $(INCLUDE) $(LIBS) $(ARGS) $(CUDALIBS) -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP

driver_eigen: driver.cpp driver.cu utilities.h lanczos_eigen.h
	g++ -O3 driver.cpp mmio.cpp -o driver_eigen -fopenmp -pg -DEIGEN -DEIGEN_USE_MKL_ALL -DFULL_EIGENSOLVE $(INCLUDE) $(LIBS) $(EIGENLIBS) $(ARGS) -I/opt/nvidia/cudatoolkit/5.5.20-1.0501.7700.8.1/include/

driver_gcc: driver.cu utilities.h tge_wrappers.h lanczos.h
	g++ driver.cpp mmio.cpp -o driver_gcc -fopenmp -O3 -pg $(INCLUDE) $(LIBS) $(ARGS) -I/scratch/santis/bcumming/github/thrust -std=c++11 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -D__host__=" " -D__device__=" "

clean :
	rm -f driver driver_host driver_eigen driver_gcc

