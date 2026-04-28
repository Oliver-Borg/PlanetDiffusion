
# Miscellaneous code for working with truncated normal distributions.
# Adapted from https://people.sc.fsu.edu/~jburkardt/py_src/truncated_normal/truncated_normal.py,
# but made much more efficient using numpy

import numpy as np


CONST1 = 0.180625
CONST2 = 1.6

C_A = np.array([
    2.5090809287301226727e+3,
    3.3430575583588128105e+4,
    6.7265770927008700853e+4,
    4.5921953931549871457e+4,
    1.3731693765509461125e+4,
    1.9715909503065514427e+3,
    1.3314166789178437745e+2,
    3.3871328727963666080
])

C_B = np.array([
    5.2264952788528545610e+3,
    2.8729085735721942674e+4,
    3.9307895800092710610e+4,
    2.1213794301586595867e+4,
    5.3941960214247511077e+3,
    6.8718700749205790830e+2,
    4.2313330701600911252e+1,
    1.0
])

C_C = np.array([
    7.74545014278341407640e-4,
    2.27238449892691845833e-2,
    2.41780725177450611770e-1,
    1.27045825245236838258,
    3.64784832476320460504,
    5.76949722146069140550,
    4.63033784615654529590,
    1.42343711074968357734
])


C_D = np.array([
    1.05075007164441684324e-9,
    5.47593808499534494600e-4,
    1.51986665636164571966e-2,
    1.48103976427480074590e-1,
    6.89767334985100004550e-1,
    1.67638483018380384940,
    2.05319162663775882187,
    1.0
])


C_E = np.array([
    2.01033439929228813265e-7,
    2.71155556874348757815e-5,
    1.24266094738807843860e-3,
    2.65321895265761230930e-2,
    1.78482653991729133580,
    5.46378491116411436990,
    6.65790464350110377720,
])

C_F = np.array([
    2.04426310338993978564e-15,
    1.42151175831644588870e-7,
    1.84631831751005468180e-5,
    7.86869131145613259100e-4,
    1.48753612908506148525e-2,
    1.36929880922735805310e-1,
    5.99832206555887937690e-1,
    1.0,
])

SPLIT1 = 0.425
SPLIT2 = 5.0


A1 = 0.398942280444
A2 = 0.399903438504
A3 = 5.75885480458
A4 = 29.8213557808
A5 = 2.62433121679
A6 = 48.6959930692
A7 = 5.92885724438
B0 = 0.398942280385
B1 = 3.8052E-08
B2 = 1.00000615302
B3 = 3.98064794E-04
B4 = 1.98615381364
B5 = 0.151679116635
B6 = 5.29330324926
B7 = 4.8385912808
B8 = 15.1508972451
B9 = 0.742380924027
B10 = 30.789933034
B11 = 3.99019417011


def normal_01_cdf_inv(cdf):

    def _helper(qvals, cdf2):

        cdf2[qvals >= 0] = 1 - cdf2[qvals >= 0]

        cdf2 = np.sqrt(-np.log(cdf2))

        # First
        mask = cdf2 <= SPLIT2
        r = cdf2[mask] - CONST2
        cdf2[mask] = np.polyval(C_C, r) / np.polyval(C_D, r)

        # Second
        mask = np.logical_not(mask)
        r = cdf2[mask] - SPLIT2
        cdf2[mask] = np.polyval(C_E, r) / np.polyval(C_F, r)

        cdf2[qvals < 0] *= -1

        return cdf2

    q = cdf - 0.5

    ###
    mask = np.abs(q) <= SPLIT1
    r = CONST1 - q[mask] * q[mask]
    q[mask] *= np.polyval(C_A, r) / np.polyval(C_B, r)

    ###
    mask2 = np.logical_not(mask)  # otherwise
    q[mask2] = _helper(q[mask2], cdf[mask2])

    return q


def normal_01_cdf(x):

    def _case1(aq):
        y = 0.5 * aq * aq
        return 0.5 - aq * (A1 - A2 * y / (y + A3 - A4 / (y + A5 + A6 / (y + A7))))

    def _case2(aq):
        return np.exp(-0.5 * aq * aq) * B0 / (aq - B1 + B2 / (aq + B3 + B4 / (aq - B5 + B6 / (aq + B7 - B8 / (aq + B9 + B10 / (aq + B11))))))

    ax = np.abs(x)

    to_return = np.piecewise(ax, [
        ax <= 1.28,
        ax <= 12.7,
        ax > 12.7
    ], [_case1, _case2, 0.5])

    to_return[x >= 0] = 1 - to_return[x >= 0]

    return to_return


def truncated_normal_ab_cdf_inv(cdf, mu, sigma, a, b):

    # if ((cdf < 0.0).any() or (1.0 < cdf).any()):
    #     raise Exception('CDF < 0 or 1 < CDF')

    alpha = (a - mu) / sigma  # constant
    beta = (b - mu) / sigma  # constant

    alpha_cdf = normal_01_cdf(alpha)
    beta_cdf = normal_01_cdf(beta)

    xi_cdf = (beta_cdf - alpha_cdf) * cdf + alpha_cdf
    xi = normal_01_cdf_inv(xi_cdf)

    return mu + sigma * xi


def truncated_normal_ab_cdf(x, mu, sigma, a, b):
    def _helper(xx):
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        xi = (xx - mu) / sigma

        alpha_cdf = normal_01_cdf(alpha)  # constant
        beta_cdf = normal_01_cdf(beta)  # constant
        xi_cdf = normal_01_cdf(xi)  # variable

        return (xi_cdf - alpha_cdf) / (beta_cdf - alpha_cdf)

    return np.piecewise(x, [
        x < a,
        x > b,
        (a <= x) & (x <= b)
    ], [0, 1, _helper])


def main():
    import time

    x = 0.5
    mu = 0
    sigma = 0.3
    a = -1
    b = 1

    x = np.random.uniform(a, b, (512, 512, 2))

    s = time.time()

    num_times = 100
    for _ in range(num_times):
        q = truncated_normal_ab_cdf(x, mu, sigma, a, b)

        q2 = truncated_normal_ab_cdf_inv(q, mu, sigma, a, b)
        print('INVERTIBLE:', (np.abs(q2 - x) < 1e-3).all())

    print('Average time:', (time.time() - s)/num_times)


if __name__ == '__main__':
    main()
