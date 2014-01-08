#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

using namespace minlin::threx;

int main() {

    int M = 3;
    int N = 5;

    // allocate storage
    HostMatrix<double> Ahost(M, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            Ahost(i,j) = i + j/10.0;
        }
    }

    DeviceMatrix<double> A = Ahost;
    DeviceMatrix<double> B = A;

    std::cout << Ahost;
    std::cout << A;

    DeviceMatrix<double> C = double(2.5)*B;
    std::cout << C;

    B*=A;
    std::cout << B;
}
