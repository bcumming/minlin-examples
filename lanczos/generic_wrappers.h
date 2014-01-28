#pragma once

#include <omp.h>

#include <mkl.h>

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

