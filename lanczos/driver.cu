#include <iostream>
#include <limits>
#include <vector>

typedef double ScalarType;

// include mkl blas implementations
#include <mkl.h>

// required to read matrix market files
#include "mm_matrix.h"

// Generic utilities needed for all versions
#include "utilities.h"

#if defined(EIGEN)

// include eigen stuff
#include "Eigen/Dense"
using namespace Eigen;
typedef Matrix<ScalarType, Dynamic, Dynamic, ColMajor> MatrixXX;
typedef Matrix<ScalarType, Dynamic, 1> VectorX;
typedef MatrixXX GenericMatrix;
#include "lanczos_eigen.h"

#else

// include minlin stuff
#include "tge_wrappers.h"
#include "lanczos.h"

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
using namespace minlin::threx; // just dump the namespace for this example

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
MINLIN_INIT
#include <cublas_v2.h>
#endif

#endif

/*******************************************************************************
  Driver for Lanczos example

  loads matrix stored in test.mtx and finds dominant eigenvalues
*******************************************************************************/
int main(int argc, char* argv[])
{

    Params params(argc-1, argv+1);
    params.print();

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
    // force initialization of cublas, so that the intialization over head doesn't turn up
    // in the lanczos routine
    cublasHandle_t handle = CublasState::instance()->handle();
#endif

    // load matrix from file
    mm_matrix<ScalarType> mat(params.fname);
    // print matrix info to screen
    mat.stats();

#if defined( EIGEN )
    Eigen::initParallel();
    // initilize eigen implementation
    MatrixXX A( mat.rows(), mat.cols() );
    mat.to_dense( A.data(), false );
    MatrixXX V(A.rows(), params.num_eigs);
    const int handle = -1;  //dummy handle
#else
    // copy over to the device

    HostMatrix<ScalarType> A_host(mat.rows(),mat.cols());
    mat.to_dense(A_host.pointer(), false);

    #ifdef USE_GPU
    DeviceMatrix<ScalarType> A = A_host;
    #else
    HostMatrix<ScalarType> A = A_host;
    #endif

    // create storage for eigenpairs
    #ifdef USE_GPU
    DeviceMatrix<ScalarType> V(A.rows(), params.num_eigs);
    #else
    HostMatrix<ScalarType> V(A.rows(), params.num_eigs);
    #endif
#endif
    std::vector<ScalarType> eigs;

    // call lanczos routine

#if defined( EIGEN )
    // a first run to remove any MKL initialization overheads
    lanczos_eigen(A, params.num_eigs, params.iters, 100, eigs, V, params.reorthogonalize);

    double time = -omp_get_wtime();
    bool success = lanczos_eigen(A, params.num_eigs, params.iters, params.tol, eigs, V, params.reorthogonalize);
#else
    // a first run to remove any MKL initialization overheads
    lanczos(A, params.num_eigs, params.iters, 100., eigs, V, params.reorthogonalize);

    //cudaDeviceSynchronize();
    double time = -omp_get_wtime();
    bool success = lanczos(A, params.num_eigs, params.iters, params.tol, eigs, V, params.reorthogonalize);
    //cudaDeviceSynchronize();
#endif
    time += omp_get_wtime();
    std::cout << "======= took " << time*1000. << " miliseconds" << std::endl;

    std::cout << "we were " << (success ? "succesfull" : "a failure") << std::endl;
    if(success)
    {
        std::cout << "eigenvalues : ";
        for(int i=0; i<params.num_eigs; i++)
            std::cout << eigs[i] << " ";
        std::cout << std::endl;
    }
}
