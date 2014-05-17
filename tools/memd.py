"""
    Implementation of multi-variate EMD according to:
    N Rehman and DP Mandic "Multivariate empirical mode decomposition"
    Proc. R. Soc. A (2010) 466, 1291-1302 doi:10.1098/rspa.2009.0502
"""
import copy
import numpy as np
from scipy.interpolate import splrep,splev
from tools.quasirand import quasirand
from tools.signal import find_extrema


def create_mirrored_spline(mini, maxi, s):
    """
        To reduce end effects, we need to extend the signal on both sides and reflect the first and last extrema
        so that interpolation works better at the edges
    """

    #get the values of s at the minima and maxima
    s_min = list(s[mini])
    s_max = list(s[maxi])

    #reflect the extrema on the left side
    Tl = maxi[0]  # index of left-hand (first) maximum
    tl = mini[0]  # index of left-hand (first) minimum

    maxi.insert(0, -tl)
    s_max.insert(0, s_max[0])
    mini.insert(0, -Tl)
    s_min.insert(0, s_min[0])

    #reflect the extrema on the right side
    T = len(s)
    Tr = maxi[-1]  # index of right hand (last) maximum
    tr = mini[-1]  # index of right hand (last) minimum

    maxi.append((T-tr) + T)
    s_max.append(s_max[-1])
    mini.append((T-Tr) + T)
    s_min.append(s_min[-1])

    #interpolate the upper and lower envelopes
    upper_env_spline = splrep(maxi, s_max, k=3)
    lower_env_spline = splrep(mini, s_min, k=3)

    return lower_env_spline,upper_env_spline


def compute_mean_envelope(s, nsamps=1000):
    """ Use random sampling to compute the mean envelope of a multi-dimensional signal.

    Args:
        s (np.ndarray): an NxT matrix describing a multi-variate signal. N is the number of channels, T is the number of time points.
        nsamps (int): the number of N dimensional projections to use in computing the multi-variate envelope.

    Returns:
        env (np.ndarray): an NxT matrix giving the multi-dimensional envelope of s.
    """

    N,T = s.shape

    #pre-allocate the mean envelope matrix
    mean_env = np.zeros([N, T])

    #generate quasi-random points on an N-dimensional sphere
    R = quasirand(N, nsamps, spherical=True)

    for k in range(nsamps):
        r = R[:, k].squeeze()

        #project s onto a scalar time series using random vector
        p = np.dot(s, r)

        #identify minima and maxima of projection
        mini_p,maxi_p = find_extrema(p)

        #for each signal dimension, fit maxima with cubic spline to produce envelope

        t = np.arange(T)
        for k in range(N):
            mini = copy.copy(mini_p)
            maxi = copy.copy(maxi_p)

            #extrapolate edges using mirroring
            lower_env_spline, upper_env_spline = create_mirrored_spline(mini, maxi, s[k, :].squeeze())

            #evaluate upper and lower envelopes
            upper_env = splev(t, upper_env_spline)
            lower_env = splev(t, lower_env_spline)

            #compute the envelope for this projected dimension
            env = (upper_env + lower_env) / 2.0

            #update the mean for this dimension in an online way
            delta = env - mean_env[k, :]
            mean_env[k, :] += delta / (k+1)

    return mean_env


def sift(s, nsamps=100):
    """Do a single iteration of multi-variate empirical mode decomposition (MEMD) on the multi-dimensional signal s, obtaining a multi-variate IMF.

    Args:
        s (np.ndarray): an NxT matrix describing a multi-variate signal. N is the number of channels, T is the number of time points.
        nsamps (int): the number of N dimensional projections to use in computing the multi-variate envelope.

    Returns:
        imf (np.ndarray): an NxT matrix giving the multi-dimensional IMF for this sift.

    """

    pass

