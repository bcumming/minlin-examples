#include <iostream>
#include <limits>

// include minlin stuff
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
using namespace minlin::threx; // just dump the namespace for this example

// include cuda and mkl blas implementations
#include <cublas_v2.h>
#include <mkl.h>

// required to read matrix market files
#include "mm_matrix.h"

#include "utilities.h"
#include "lanczos.h"

MINLIN_INIT

/*******************************************************************************
  Driver for Lanczos example

  loads matrix stored in test.mtx and finds dominant eigenvalues
*******************************************************************************/
int main(int argc, char* argv[])
{
    typedef double real;

    Params params(argc-1, argv+1);
    params.print();

    #ifdef USE_GPU
    // force initialization of cublas, so that the intialization over head doesn't turn up
    // in the lanczos routine
    cublasHandle_t handle = CublasState::instance()->handle();
    #endif

    // load matrix from file
    mm_matrix<real> mat(params.fname);
    // print matrix info to screen
    mat.stats();

    HostMatrix<real> A_host(mat.rows(),mat.cols());
    mat.to_dense(A_host.pointer(), false);

    // copy over to the device
    #ifdef USE_GPU
    DeviceMatrix<real> A = A_host;
    #else
    HostMatrix<real> A = A_host;
    #endif

    // create storage for eigenpairs
    #ifdef USE_GPU
    DeviceMatrix<real> V(A.rows(), params.num_eigs);
    #else
    HostMatrix<real> V(A.rows(), params.num_eigs);
    #endif
    std::vector<real> eigs;

    // call lanczos routine

    double time = -omp_get_wtime();
    bool success = lanczos(A, params.num_eigs, params.iters, params.tol, eigs, V, params.reorthogonalize);
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
