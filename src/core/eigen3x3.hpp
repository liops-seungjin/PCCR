// Symmetric 3x3 eigen-decomposition (cyclic Jacobi), shared by normal estimation
// and PCA. Eigenvalues in w[]; eigenvectors as the columns of v[][]. Internal.
#pragma once

#include <cmath>

namespace cc::detail {

inline void jacobiEigen(double a[3][3], double w[3], double v[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) v[i][j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 50; ++sweep) {
        const double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
        if (off < 1e-300) break;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(a[p][q]) < 1e-300) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                double       t     = (theta >= 0 ? 1.0 : -1.0) /
                           (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                if (theta == 0.0) t = 1.0;
                const double c   = 1.0 / std::sqrt(t * t + 1.0);
                const double s   = t * c;
                const double tau = s / (1.0 + c);
                const double apq = a[p][q];

                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    if (r != p && r != q) {
                        const double arp = a[r][p];
                        const double arq = a[r][q];
                        a[r][p] = arp - s * (arq + tau * arp);
                        a[p][r] = a[r][p];
                        a[r][q] = arq + s * (arp - tau * arq);
                        a[q][r] = a[r][q];
                    }
                }
                for (int r = 0; r < 3; ++r) {
                    const double vrp = v[r][p];
                    const double vrq = v[r][q];
                    v[r][p] = vrp - s * (vrq + tau * vrp);
                    v[r][q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) w[i] = a[i][i];
}

}  // namespace cc::detail
