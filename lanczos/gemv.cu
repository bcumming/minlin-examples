#include <iostream>
#include <limits>
#include <vector>

#include <omp.h>

typedef double ScalarType;

// include mkl blas implementations
#include <mkl.h>

// required to read matrix market files
#include "mm_matrix.h"

// Generic utilities needed for all versions
#include "utilities.h"

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
using namespace minlin::threx; // just dump the namespace for this example

#include <cublas_v2.h>

MINLIN_INIT

/*******************************************************************************
  Driver for Lanczos example

  loads matrix stored in test.mtx and finds dominant eigenvalues
*******************************************************************************/
int main(int argc, char* argv[])
{

    Params params(argc-1, argv+1);
    //params.print();

    // force initialization of CUBLAS
    cublasHandle_t handle = CublasState::instance()->handle();

    // load matrix from file
    mm_matrix<ScalarType> mat(params.fname);
    // print matrix info to screen
    //mat.stats();

    int N = mat.rows();

    // copy to device
    HostMatrix<ScalarType> Ah(N,N);
    mat.to_dense(Ah.pointer(), false);

    DeviceMatrix<ScalarType> Ad = Ah;

    // vectors for operations
    HostVector<ScalarType>   xh(N) ;
    HostVector<ScalarType>   yh(N);
    xh(all) = 1.0;

    DeviceVector<ScalarType> xd = xh;
    DeviceVector<ScalarType> yd(N);


    std::cout << "performing " << params.iters << " matrix-vector multiplications with dimension " << N << std::endl;
    std::cout << omp_get_max_threads() << " OpenMP threads" << std::endl;

    yd(all) = Ad*xd;
    yh(all) = Ah*xh;

    double time_host = -omp_get_wtime();
    for(int i=0; i<params.iters; i++)
        yh(all) = Ah*xh;
    time_host += omp_get_wtime();
    std::cout << "OpenMP took :: " << time_host*1000. << " miliseconds" << std::endl;

    cudaDeviceSynchronize();
    double time_device = -omp_get_wtime();
    for(int i=0; i<params.iters; i++)
        yd(all) = Ad*xd;
    cudaDeviceSynchronize();
    time_device += omp_get_wtime();
    std::cout << "CUDA took   :: " << time_device*1000. << " miliseconds " << std::endl;

    HostVector<ScalarType> ytmp = yd;
    //std::cout << yd(0,10) << std::endl;
    //std::cout << yh(0,10) << std::endl;
    ScalarType error = norm(ytmp - yh)/norm(yh);
    std::cout << "the error is " << error << std::endl;
}
