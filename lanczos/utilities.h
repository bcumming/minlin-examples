#pragma once

#include <sstream>

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
            std::cerr << "  -r " << std::endl;
            std::cerr << "      use reorthogonalization in Lanczos (default off)" << std::endl;
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


