all : driver

# set up system-specific settings
# greina
MINLINROOT=/users/bcumming/github/minlin/include
MKLROOT=/cm/shared/apps/intel/composer_xe/2011_sp1.12.361/mkl
GPP=/cm/shared/apps/gcc/4.7.1/bin/g++
BOOSTROOT=/users/bcumming/repos/boost_1_49_0

# SANTIS
MINLINROOT=/scratch/santis/bcumming/github/minlin/include
MKLROOT=/opt/intel/13.1.0.146/composer_xe_2013.2.146/mkl
GPP=/opt/gcc/4.8.1/bin/g++
BOOSTROOT=/apps/daint/boost/1.54.0/gnu_473/include

MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_gnu_thread -lmkl_core -lmkl_rt
CUDALIBS=-lcublas
LIBS=$(MKLLIBS) $(CUDALIBS)

INCLUDE=-I$(MINLINROOT) -I$(MKLROOT)/include -I${BOOSTROOT}

ARGS=-DUSE_GPU -g
#ARGS=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

driver: driver.cu utilities.h lanczos.h
	nvcc -arch=sm_35 driver.cu mmio.cpp -o driver -ccbin=$(GPP) --compiler-options "-fopenmp" $(INCLUDE) $(LIBS) $(ARGS)
#	nvcc -arch=sm_35 driver.cu mmio.cpp -o driver -ccbin=$(GPP) --compiler-options "-fopenmp -pg" $(INCLUDE) $(LIBS) $(ARGS)

driver_host: driver.cu utilities.h lanczos.h
	g++ driver.cpp mmio.cpp -o driver -fopenmp $(INCLUDE) $(LIBS) $(ARGS) -I/opt/nvidia/cudatoolkit/5.5.20-1.0501.7700.8.1/include/


clean :
	rm -f driver driver_host

