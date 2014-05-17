"""
    Implementation of multi-variate EMD according to:
    N Rehman and DP Mandic "Multivariate empirical mode decomposition"
    Proc. R. Soc. A (2010) 466, 1291-1302 doi:10.1098/rspa.2009.0502
"""
import numpy as np

def compute_mean_envelope(s, nsamps=100):
    """ Use random sampling to compute the mean envelope of a multi-dimensional signal.

    Args:
        s (np.ndarray): an NxT matrix describing a multi-variate signal. N is the number of channels, T is the number of time points.
        nsamps (int): the number of N dimensional projections to use in computing the multi-variate envelope.

    Returns:
        env (np.ndarray): an NxT matrix giving the multi-dimensional envelope of s.
    """

    for k in range(nsamps):
        #obtain a random N-dimensional vector

        #project s onto vector

        #identify maxima of projection

        #fit maxima with cubic spline to produce envelope

        #add to mean envelope
        pass


def sift(s, nsamps=100):
    """Do a single iteration of multi-variate empirical mode decomposition (MEMD) on the multi-dimensional signal s, obtaining a multi-variate IMF.

    Args:
        s (np.ndarray): an NxT matrix describing a multi-variate signal. N is the number of channels, T is the number of time points.
        nsamps (int): the number of N dimensional projections to use in computing the multi-variate envelope.

    Returns:
        imf (np.ndarray): an NxT matrix giving the multi-dimensional IMF for this sift.

    """

    pass

