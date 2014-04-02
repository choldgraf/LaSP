
import numpy as np
from sklearn.linear_model import Ridge
import spams
import time


def make_toeplitz(input, lags, include_bias=True, fortran_style=False):
    """
        Assumes input is of dimensionality nt x nf, where nt is the number of time points and
        nf is the number of features.

        lags is an array of integers, representing the time lag from zero. a negative time lag points to the future,
        positive to the past.
    """

    nt = input.shape[0]
    nf = input.shape[1]
    d = len(lags)

    if fortran_style:
        A = np.zeros([nt, nf*d+include_bias], order='F')
    else:
        A = np.zeros([nt, nf*d+include_bias])
    if include_bias:
        A[:, -1] = 1.0 # the bias term

    all_indices = np.arange(d*nf)

    #compute the channel corresponding to each parameter in the reshaped (flattened) filter
    channel_indices = np.floor(all_indices / float(d)).astype('int')

    #compute the lag index corresponding to each parameter in the reshaped (flattened) filter
    lag_indices = all_indices % d
    #print 'lag_indices=',lag_indices

    for k,i in enumerate(all_indices):
        #get lag and channel corresponding to this index
        lag_index = lag_indices[i]
        #print 'k=%d, i=%d, lag_index=%d' % (k, i, lag_index)
        lag = lags[lag_index]
        channel_to_get = channel_indices[i]

        if lag == 0:
            A[:, k] = input[:, channel_to_get]
        else:
            #shift time series for this channel up or down depending on lag
            if lag > 0:
                A[lag:, k] = input[:-lag, channel_to_get]
            else:
                A[:lag, k] = input[-lag:, channel_to_get] #note that lag is negative
    return A


def fit_strf_lasso(input, output, lags, lambda1=1.0, lambda2=1.0):

    #convert the input into a toeplitz-like matrix
    stime = time.time()
    A = make_toeplitz(input, lags, include_bias=True, fortran_style=True)
    etime = time.time() - stime
    print '[fit_strf_lasso] Time to make Toeplitz matrix: %d seconds' % etime

    fy = np.asfortranarray(output.reshape(len(output), 1))
    #print 'fy.shape=',fy.shape
    #print 'fA.shape=',fA.shape

    #fit the STRF
    stime = time.time()
    fit_params = spams.lasso(fy, A, mode=2, lambda1=lambda1, lambda2=lambda2)
    etime = time.time() - stime
    print '[fit_strf_lasso] Time to fit STRF: %d seconds' % etime

    #reshape the STRF so that it makes sense
    nt = input.shape[0]
    nf = input.shape[1]
    d = len(lags)
    strf = np.array(fit_params[:-1].todense()).reshape([nf, d])
    bias = fit_params[-1].todense()[0, 0]

    return strf,bias


def fit_strf_ridge(input, output, lags, alpha=1.0):

    #convert the input into a toeplitz-like matrix
    stime = time.time()
    A = make_toeplitz(input, lags, include_bias=False)
    etime = time.time() - stime
    print '[fit_strf_ridge] Time to make Toeplitz matrix: %d seconds' % etime

    #fit the STRF
    stime = time.time()

    rr = Ridge(alpha=alpha, copy_X=False, fit_intercept=True, normalize=False)
    rr.fit(A, output)
    etime = time.time() - stime
    print '[fit_strf_ridge] Time to fit STRF: %d seconds' % etime

    #reshape the STRF so that it makes sense
    nt = input.shape[0]
    nf = input.shape[1]
    d = len(lags)
    strf = np.array(rr.coef_).reshape([nf, d])
    bias = rr.intercept_

    return strf,bias

