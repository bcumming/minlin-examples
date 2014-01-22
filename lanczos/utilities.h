#pragma once

// blas routines for cuda and mkl
#include <mkl.h>

#include <sstream>

template <typename T>
void matlab_vector(const DeviceVector<T> &v, const char *name ) {
    std::cout << name << " = [";
    for(int i=0; i<v.size(); i++)
        std::cout << v(i) << " ";
    std::cout << "]';" << std::endl;
}

struct Params {
    std::string fname;
    double tol;
    int iters;
    bool reorthogonalize;
    int num_eigs;

    Params(int argc, char** argv)
    :   tol(1.e-7),
        iters(100),
        reorthogonalize(false),
        num_eigs(1)
    {
        std::string error_string;
        bool fileset = false;

        for(int i=0; i<argc; i++) {
            std::string arg(argv[i]);

            if( arg == "-r" ) {
                reorthogonalize = true;
            }
            else if( arg == "-f" ) {
                i++;
                if(i<argc && argv[i][0]!='-'){
                    fname = argv[i];
                    fileset = true;
                }
                else {
                    error_string += "error: invalid value for input file, must have form -t <char[]>, e.g.: -f file.mtx\n";
                }
            }
            else if( arg == "-t" ) {
                i++;
                if(i<argc && argv[i][0]!='-') {
                    std::istringstream(argv[i]) >> tol;
                }
                else {
                    error_string += "error: invalid value for tolerance, must have form -t <double>, e.g.: -t 1.0e-5\n";
                }
            }
            else if( arg == "-n" ) {
                i++;
                if(i<argc && argv[i][0]!='-') {
                    std::istringstream(argv[i]) >> num_eigs;
                }
                else {
                    error_string += "error: invalid value for number of eigenvalues, must have form -n <int>, e.g.: -n 2\n";
                }
            }
            else if( arg == "-i" ) {
                i++;
                if(i<argc && argv[i][0]!='-') {
                    std::istringstream(argv[i]) >> iters;
                }
                else {
                    error_string += "error: invalid value for max iterations, must have form -i <unsigned int>, e.g.: -i 30\n";
                }
            }
        }

        if(error_string.size() || !fileset) {
            if(error_string.size())
                std::cerr << error_string;
            std::cerr << "usage : driver args" << std::endl;
            std::cerr << "arguments can be:" << std::endl;
            std::cerr << "  -f <string>     mandatory" << std::endl;
            std::cerr << "      filename for matrix market file" << std::endl;
            std::cerr << "  -i <int>        default 100" << std::endl;
            std::cerr << "      maximum number of iterations" << std::endl;
            std::cerr << "  -t <float>      default 1e-7" << std::endl;
            std::cerr << "      tolerance" << std::endl;
            std::cerr << "  -n <int>        default 1" << std::endl;
            std::cerr << "      number of eigenpairs to compute" << std::endl;
            exit(1);
        }
    }

    void print() {
        std::cout << "=====================================================" << std::endl;
        std::cout << "input file        :   " << fname << std::endl;
        std::cout << "tolerance         :   " << tol << std::endl;
        std::cout << "input file        :   " << iters << std::endl;
        std::cout << "reorthogonalize   :   " << (reorthogonalize==true ? "yes" : "no") << std::endl;
        std::cout << "eigenvalues       :   " << num_eigs << std::endl;
        std::cout << "=====================================================" << std::endl;
    }
};

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
    //std::cout << "calling geev wrapper routine for " << traits<real>::print_type() << std::endl;
    //std::cout << "sizeof(real) " << sizeof(real) << std::endl;
    //std::cout << "sizeof(double) " << sizeof(double) << std::endl;
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

