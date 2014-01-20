#pragma once

// include minlin stuff
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
using namespace minlin::threx; // just dump the namespace for this example

// include cuda and mkl blas implementations
#include <cublas_v2.h>
#include <mkl.h>

#include "utilities.h"

#include <omp.h>

// compute the dominant eigenpairs of input matrix A
//  INPUT :
//      A       : symmetric real matrix with dims N*N
//      ne      : number of eigenpairs to estimate
//      m       : maximum subspace size
//      tol     : maximum allowed
//  OUTPUT :
//      EV      : matrix of dimensions N*ne with eigenvectors
//      e       : vector containing the host values
//  RETURN :
//                bool indicating success/failure to converge
//template <typename real, template <typename> class Matrix, template <typename> class Vector>
template <typename real, template <typename> class Matrix>
bool lanczos
(
    const Matrix<real> A,
    int ne,
    int m,
    real tol,
    std::vector<real> &e,
    Matrix<real> EV,
    cublasHandle_t handle,
    bool reorthogonalize=false
)
{
    #ifdef USE_GPU
    typedef DeviceVector<real> Vector;
    #else
    typedef HostVector<real> Vector;
    #endif

    real gamma, delta;
    int N = A.rows();

    // check that output matrix has correct dimensions
    assert(EV.rows() == N);
    assert(EV.cols() == ne);

    // workspace for subspace construction
    std::cout << "input matrix has dimensions " << N << std::endl;
    std::cout << "V has dims " << N << "*" << m << std::endl;
    Matrix<real> V(N,m);

    // storage for tridiagonal matrix
    // storage is on the host for both GPU and OpenMP implementations
    HostMatrix<real>   T(m,m);

    // work space for the eigenvalue calculations
    // storage is on the host for both GPU and OpenMP implementations
    HostVector<real> er(m);        // real components of eigenvalues
    HostVector<real> ei(m);        // imaginary components of eigenvalues

    // initial nozero (normalized) in the first column of V
    V(all,0) = real(1.);
    V(all,0) /= norm(V(all,0));    // Unit vector

    // find product w=A*V(:,0)
    Vector w = A*V(all,0);
    delta = dot(w, V(all,0));
    T(0,0) = delta; // store in tridiagonal matrix

    // preallocate residual storage vector
    Vector r(A.rows());

    // main loop, will terminate earlier if tolerance is reached
    bool converged = false;
    for(int j=1; j<min(m, N) && !converged; ++j) {
        //std::cout << "================= ITERATION " << j << "    ";
        if ( j == 1 )
            w -= delta*V(all,j-1);
        else
            w -= delta*V(all,j-1) + gamma*V(all,j-2);

        gamma = norm(w);
        V(all, j) = (1./gamma)*w;

        // reorthogonalize
        if( reorthogonalize ) {
            for( int jj = 0; jj < j; ++jj )  {
                real alpha =  dot( V(all,jj), V(all,j) );
                V(all,j) -= alpha * V(all,jj);
            }
        }

        // write off-diagonal values in tri-diagonal matrix
        T(j-1,j  ) = gamma;
        T(j  ,j-1) = gamma;

        // find matrix-vector product for next iteration
        w(all) = A*V(all,j);

        // update diagonal of tridiagonal system
        delta = dot(w, V(all,j));
        T(j, j) = delta;

        if ( j >= ne ) {
            // find eigenvectors/eigenvalues for the reduced triangular system
            HostMatrix<real> Tsub = T(0,j,0,j);
            HostMatrix<real> UVhost(j+1,ne);
            assert( geev<real>(Tsub, UVhost, er, ei, ne) );

            // copy eigenvectors for reduced system to the device
            ////////////////////////////////////////////////////////////////////
            // TODO : can we find a way to allocate memory for UV outside the
            //        inner loop?
            //
            //        dimensions of UV are (j+1, ne)
            //        ne is typically 1-5, so this won't ever be that large
            //
            //        first tests with nvprof suggest that the malloc-free
            //        on the device from the line below are not causing very
            //        high overhead, but have to keep an eye on this
            ////////////////////////////////////////////////////////////////////
            Matrix<real> UV = UVhost;
            // find EV = V(:,0:j-1)*UV
            assert(
                gemm_wrapper(
                    N, ne, j+1,
                    V.pointer(), V.rows(),
                    UV.pointer(), UV.rows(),
                    EV.pointer(), EV.rows(),
                    real(1.0), real(0.0), handle
                )
            );

            real max_err = 0.;
            for(int count=0; count<ne && !converged; count++){
                real this_eig = er(count);

                // find the residual
                r(all) = A*EV(all,count);

                // compute the relative error from the residual
                real this_err = abs( norm(r-this_eig*EV(all,count)) / this_eig );
                max_err = max(max_err, this_err);

                // terminate early if the current error exceeds the tolerance
                if(max_err > tol)
                    break;
            } // end-for error estimation
            std::cout << "iteration : " << j << ", error : " << max_err << std::endl;
            // test for convergence
            if(max_err < tol)
                converged = true;
        } // end-if estimate eigenvalues
    } // end-for main

    // return failure if no convergence
    if(!converged)
        return false;

    // pack the eigenvalues into user-provided vector
    e.resize(ne);
    std::copy(er.pointer(), er.pointer()+ne, e.begin());

    return true;
}

