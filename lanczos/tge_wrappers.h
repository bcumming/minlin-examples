#pragma once

#include <omp.h>

#include <mkl.h>

#include <mkl.h>
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
using namespace minlin::threx; // just dump the namespace for this example

template <typename T>
void matlab_vector(const DeviceVector<T> &v, const char *name ) {
    std::cout << name << " = [";
    for(int i=0; i<v.size(); i++)
        std::cout << v(i) << " ";
    std::cout << "]';" << std::endl;
}

//  lapack_int
//  LAPACKE_dsteqr
//      int matrix_order,   LAPACK_COL_MAJOR
//      char compz,         set to 'I'
//      lapack_int n,       dimension of matrix
//      double* d,          diagonal elements (length n)
//      double* e,          off diagonal elements (lengthn-1)
//      double* z,          work array of length n*n
//      lapack_int ldz      n
//  output :    d   :   contains eigenvalues in ascending order
//              e   :   overwritten with arbitrary data
//              z   :   contains n orthonormal eigenvectors stored as columns
lapack_int steqr(lapack_int n, float* d, float* e, float* z) {
    return LAPACKE_ssteqr(LAPACK_COL_MAJOR, 'I', n, d, e, z, n);
}
lapack_int steqr(lapack_int n, double* d, double* e, double* z) {
    return LAPACKE_dsteqr(LAPACK_COL_MAJOR, 'I', n, d, e, z, n);
}

template <typename real>
bool steigs
    (
        real *T,    // nxn input matrix T
        real *V,    // nxn output matrix for eigenvectors
        real *eigs, // output for eigenvalues
        int n,
        int num_eigs=-1
    )
{
    num_eigs = num_eigs<0 ? n : num_eigs;
    num_eigs = num_eigs>n ? n : num_eigs;

    // allocate memory for storing superdiagonal
    real *e = (real*)malloc(sizeof(real)*(n-1));
    // allocate memory for eigenvectors returned by LAPACK
    real *z = (real*)malloc(sizeof(real)*(n*n));
    // point d to eigs (?steqr stores eigenvalues in vector used to pass in diagonal)
    real *d = eigs;

    // pack the diagonal and super diagonal of T
    int pos=0;
    for(int i=0; i<n-1; i++) {
        d[i] = T[pos];       // diagonal at T(i,i)
        e[i] = T[pos+1];     // off diagonal at T(i,i+1)
        pos += (n+1);
    }
    d[n-1] = T[pos];

    // compute eigenvalues
    lapack_int result = steqr(n, d, e, z);
    if(result)
        return false;

    // reverse the order of the eigenvalue storage
    std::reverse(eigs, eigs+n);
    for(int i=0; i<num_eigs; i++) {
        real* ptr_to   = V + i*n;
        real* ptr_from = z + (n-i-1)*n;
        std::copy(ptr_from,  ptr_from+n,  ptr_to);
    }

    // free working array
    free(e);
    free(z);

    return true;
}



lapack_int tgeev(lapack_int n, lapack_int lda, double* A, double* ER, double* EI, double* VL, double* VR)
{
    return LAPACKE_dgeev
        (
         LAPACK_COL_MAJOR, 'N', 'V',
         n, A, lda,
         ER, EI,
         VL, n, VR, n
        );
}
lapack_int tgeev(lapack_int n, lapack_int lda, float* A, float* ER, float* EI, float* VL, float* VR)
{
    return LAPACKE_sgeev
        (
         LAPACK_COL_MAJOR, 'N', 'V',
         n, A, lda,
         ER, EI,
         VL, n, VR, n
        );
}

// double precision generalized eigenvalue problem (host!)
// only returns the right-eigenvectors
// TODO: this should be replaced with call to symmetric tridiagonal eigensolver
//       dstev/sstev
template <typename real>
bool geev (
        HostMatrix<real> &A,  // nxn input matrix
        HostMatrix<real> &V,  // nxn output matrix for eigenvectors
        HostVector<real> &er, // output for real component of eigenvalues
        HostVector<real> &ei, // output for imaginary compeonent of eigenvalues
        int find_max=0          // if >0, sort the eigenvalues, and sort the
        // corresponding find_max eigenvections
        )
{
    lapack_int n = A.rows();
    lapack_int lda = n;
    real *VL = (real*)malloc(sizeof(real)*n*n);
    real *VR = 0;
    real *EI = 0;
    real *ER = 0;
    if( find_max>0 ) {
        EI = (real*)malloc(sizeof(real)*n);
        ER = (real*)malloc(sizeof(real)*n);
        VR = (real*)malloc(sizeof(real)*n*n);
        find_max = std::min(find_max, n);
    }
    else {
        VR = V.pointer();
        EI = ei.pointer();
        ER = er.pointer();
    }

    // call type-overloaded wrapper for LAPACK geev rutine
    lapack_int result = tgeev(n, lda, A.pointer(), ER, EI, VL, VR);

    // did the user ask for the eigenvalues to be sorted?
    if( find_max ) {
        // we can't use pair<real,int> because of a odd nvcc bug.. so just use floats for the sort
        typedef std::pair<real,int> index_pair;

        // generate a vector with index/value pairs for sorting
        std::vector<index_pair> v;
        v.reserve(n);
        for(int i=0; i<n; i++) {
            // calculate the magnitude of each eigenvalue, and push into list with it's index
            // sqrt() not required because the values are only used for sorting
            float mag = ER[i]*ER[i] + EI[i]*EI[i];
            v.push_back(std::make_pair(mag, i));
        }

        // sort the eigenvalues
        std::sort(v.begin(), v.end());

        // copy the vectors from temporay storage to the output array (count backwards)
        // only copy over the largest find_max pairs
        typename std::vector<index_pair >::const_iterator it = v.end()-1;
        for(int i=0; i<find_max; --it, ++i){
            int idx = it->second;
            real *to = V.pointer()+i*lda;
            real *from = VR+idx*n;
            // copy the eigenvector into the output matrix
            std::copy(from, from+n, to);
            // copy the components of the eigenvalues into the output arrays
            er(i) = ER[idx];
            ei(i) = EI[idx];
        }
    }

    // free working memory
    free(VL);
    if( find_max ) {
        // free temporary arrays used for sorting eigenvectors
        free(VR);
        free(ER);
        free(EI);
    }

    // return bool indicating whether we were successful
    return result==0; // LAPACKE_dgeev() returns 0 on success
}
