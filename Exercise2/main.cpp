#include "Eigen/Eigen"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Eigen;

void palu(const MatrixXd& A, const VectorXd& b, const VectorXd& expected)
{
    PartialPivLU<MatrixXd> palu(A);
    VectorXd xp = palu.solve(b);
    cout << "xp = \n" << xp << endl;
    double errp = (xp - expected).norm() / expected.norm();
    cout << "Relative error PALU: " << errp << endl;
    cout << "\n" << endl;
    return;
}

void qr(const MatrixXd& A, const VectorXd& b, const VectorXd& expected)
{
    ColPivHouseholderQR<MatrixXd> qr(A);
    VectorXd xq = qr.solve(b);
    cout << "xq= \n" << xq << endl;
    double errq = (xq - expected).norm() / expected.norm();
    cout << "Relative error QR: " << errq << endl;
    cout << "\n" << endl;
    return;
}

int main()
{
    const int n = 2;

    VectorXd expected(n);
    expected << -1.0e+0, -1.0e+00;
    MatrixXd A1(n,n);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01,
        -9.992887623566787e-01;
    cout << "A1 = \n" << scientific << setprecision(15) << A1 << endl;
    VectorXd b1(n);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    cout << "b1 = \n" << b1 << endl;

    palu(A1, b1, expected);
    qr(A1, b1, expected);

    MatrixXd A2(n,n);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,
        -8.324762492991313e-01;
    cout << "A2 = \n" << A2 << endl;
    VectorXd b2(n);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    cout << "b2 = \n" << b2 << endl;

    palu(A2, b2, expected);
    qr(A2, b2, expected);

    MatrixXd A3(n,n);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,
        -8.320502947645361e-01;
    cout << "A3 = \n" << A3 << endl;
    VectorXd b3(n);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    cout << "b3 = \n" << b3 << endl;

    palu(A3, b3, expected);
    qr(A3, b3, expected);

    return 0;
}
