"""
Estimation input model (README ground rule: estimation is taken as good, only its error and age are modelled).

No estimator is implemented here. An estimate is the true value plus an error, possibly as it was some time ago:
    imu(x, sigma, bias)             own motion from the node's IMU: white noise + constant bias, no delay
    aged(t, x, age)                 the value of x held from `age` seconds earlier (a shared, stale estimate)
    link(t, x, sigma, age)          slow part exchanged over the link: aged + white noise
    noisy(x, sigma)                 a generic estimate with white noise (e.g. the receiver's in-packet residual)
    gauss_markov(n, fs, sigma, tau) correlated error (first-order Gauss-Markov), e.g. the slow drift of an IMU velocity
All functions take the true series x (any shape (N, ...)) and return an estimate of the same shape; the random
generator is passed explicitly for reproducible runs.
"""
import numpy as np


def noisy(x, sigma, rng):
    x = np.asarray(x, float)
    return x + sigma * rng.standard_normal(x.shape)


def imu(x, sigma, rng, bias=0.0):
    """Own-motion estimate from an IMU: no delay, white noise sigma, constant bias."""
    return noisy(x, sigma, rng) + bias


def gauss_markov(n, fs, sigma, tau, rng):
    """First-order Gauss-Markov sequence, stationary std sigma, correlation time tau [s], sampled at fs."""
    a = np.exp(-1.0 / (fs * tau))
    w = rng.standard_normal(n) * sigma * np.sqrt(1 - a * a)
    x = np.empty(n)
    x[0] = sigma * rng.standard_normal()
    for i in range(1, n):
        x[i] = a * x[i - 1] + w[i]
    return x


def aged(t, x, age):
    """x(t - age) on the grid t: the estimate available now is the value measured `age` seconds ago. Before the
    first sample the first value is held."""
    x = np.asarray(x, float)
    return np.interp(np.asarray(t, float) - age, t, x) if x.ndim == 1 else \
        np.stack([np.interp(np.asarray(t, float) - age, t, x[:, i]) for i in range(x.shape[1])], 1)


def link(t, x, sigma, age, rng):
    """Slow part shared over the link: measured `age` seconds ago, with white noise sigma."""
    return noisy(aged(t, x, age), sigma, rng)
