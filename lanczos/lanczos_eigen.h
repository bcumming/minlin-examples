#pragma once

#include <omp.h>

#include <Eigen/Dense>
using namespace Eigen;


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
//template <typename ScalarType, template <typename> class Matrix, template <typename> class Vector>

template <typename ScalarType>
bool lanczos_eigen
(
    const MatrixXX A,
    int ne,
    int m,
    ScalarType tol,
    std::vector<ScalarType> &e,
    MatrixXX EV,
    bool reorthogonalize=false
)
{

    ScalarType gamma, delta;
    int N = A.rows();

    // check that output matrix has correct dimensions
    assert(EV.rows() == N);
    assert(EV.cols() == ne);
    assert(N         >= m);

    // workspace for subspace construction
    std::cout << "EIGEN threads " << Eigen::nbThreads() << std::endl;
    std::cout << "input matrix has dimensions " << N << std::endl;
    std::cout << "V has dims " << N << "*" << m << std::endl;
    MatrixXX V(N,m);

    // storage for tridiagonal matrix
    // storage is on the host for both GPU and OpenMP implementations
    MatrixXX   T = MatrixXX::Zero(m,m);

    // initial nonzero (normalized) in the first column of V
    V.col(0).setOnes();
    V.col(0) /= V.col(0).norm();
    VectorX w = A*V.col(0);
    // find product w=A*V(:,0)
    delta = w.transpose() * V.col(0);
    T(0,0) = delta; // store in tridiagonal matrix

    // preallocate residual storage vector
    VectorX r(N);

    // main loop, will terminate earlier if tolerance is reached
    bool converged = false;
    for(int j=1; j<m && !converged; ++j) {
        ////// timing logic //////
        //double time = -omp_get_wtime();
        //////////////////////////
        // std::cout << "================= ITERATION " << j << "    " << std::endl;

        if ( j == 1 )
            w -= delta*V.col(j-1);
        else
            w -= delta*V.col(j-1) + gamma*V.col(j-2);


        gamma = w.norm();
        V.col(j) = (1./gamma)*w;

        // reorthogonalize
        if( reorthogonalize ) {
            for( int jj = 0; jj < j; ++jj )  {
	      ScalarType alpha =  V.col(jj).transpose() * V.col(j) ;
                V.col(j) -= alpha * V.col(jj);
            }
        }

        // write off-diagonal values in tri-diagonal matrix
        T(j-1,j  ) = gamma;
        T(j  ,j-1) = gamma;

        // find matrix-vector product for next iteration
        w = A*V.col(j);

        // update diagonal of tridiagonal system
        delta = w.transpose() * V.col(j);
        T(j, j) = delta;

        if ( j >= ne ) {
            // find eigenvectors/eigenvalues for the reduced triangular system
	    SelfAdjointEigenSolver<MatrixXX> eigensolver(T.block(0,0,j+1,j+1));
	    if (eigensolver.info() != Success) abort();
	    VectorX  eigs = eigensolver.eigenvalues().block(j+1-ne,0,ne,1);  // ne largest Ritz values, sorted ascending
	    MatrixXX UT = eigensolver.eigenvectors();   // Ritz vectors

            // std::cout << "iteration : " << j << ", Tblock : " << T.block(0,0,j+1,j+1) << std::endl;
            // std::cout << "iteration : " << j << ", ritz values " << eigs << std::endl;
            // std::cout << "iteration : " << j << ", ritz vectors " << UT << std::endl;
            // j or j+1 ??

	    EV = V.block(0,0,N,j+1)*UT.block(0,j+1-ne,j+1,ne);  // Eigenvector approximations for largest ne eigenvalues

            // copy eigenvectors for reduced system to the device
            ////////////////////////////////////////////////////////////////////
            // TODO : can we find a way to allocate memory for UV outside the
            //        inner loop? this memory allocation is probably killing us
            //        particularly if we go to large subspace sizes
            //
            ////////////////////////////////////////////////////////////////////

            ScalarType max_err = 0.;
            const int boundary = j+1-ne;
            for(int count=ne-1; count>=0 && !converged; count--){
                ScalarType this_eig = eigs(count);
                // std::cout << "iteration : " << j << ", this_eig : " << this_eig << std::endl;

                // find the residual
                r = A * EV.col(count) - this_eig*EV.col(count);

                // compute the relative error from the residual
                ScalarType this_err = std::abs( r.norm() / this_eig );
		max_err = std::max(max_err, this_err);
                // terminate early if the current error exceeds the tolerance
                if(max_err > tol)
                    break;
            } // end-for error estimation
            std::cout << "iteration : " << j << ", max_err : " << max_err << std::endl;
            // test for convergence
            if(max_err < tol) {
	      // pack the eigenvalues into user-provided vector
	      e.resize(ne);
	      std::copy(eigs.data(), eigs.data()+ne, e.begin());
	      converged = true;
	    }
        } // end-if estimate eigenvalues
        ////// timing logic //////
        //time += omp_get_wtime();
        //std::cout << "took " << time*1000. << " miliseconds" << std::endl;
        //////////////////////////
    } // end-for main

    // return failure if no convergence
    if(!converged)
      return false;
    else
      return true;
}

