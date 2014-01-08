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

/*******************************************************************************
  Driver for Lanczos example

  loads matrix stored in test.mtx and finds dominant eigenvalues
*******************************************************************************/
int main(void)
{
    typedef double real;

    // load matrix from file
    mm_matrix<real> mat("./test.mtx");
    // print matrix info to screen
    mat.stats();

    // initilize cuda blas implementation
    cublasHandle_t handle = init_cublas();

    HostMatrix<real> A_host(mat.rows(),mat.cols());
    mat.to_dense(A_host.pointer(), false);

    // copy over to the device
    #ifdef USE_GPU
    DeviceMatrix<real> A = A_host;
    #else
    HostMatrix<real> A = A_host;
    #endif

    // iteration parameters
    real tol = 1.0e-5;
    bool reorthogonalize = true;
    int ne = 2;
    int max_iter = 100;

    // create storage for eigenpairs
    #ifdef USE_GPU
    DeviceMatrix<real> V(A.rows(), ne);
    #else
    HostMatrix<real> V(A.rows(), ne);
    #endif
    std::vector<real> eigs;

    // call lanczos routine

    double time = -omp_get_wtime();
    bool success = lanczos(A, ne, max_iter, tol, eigs, V, handle, reorthogonalize);
    time += omp_get_wtime();
    std::cout << "======= took " << time*1000. << " miliseconds" << std::endl;

    std::cout << "we were " << (success ? "succesfull" : "a failure") << std::endl;
    if(success)
    {
        std::cout << "eigenvalues : ";
        for(int i=0; i<ne; i++)
            std::cout << eigs[i] << " ";
        std::cout << std::endl;
    }

    kill_cublas(handle);
}
