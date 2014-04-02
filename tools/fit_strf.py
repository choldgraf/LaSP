
import numpy as np
import spams


def make_toeplitz(input, lags, include_bias=True):
    """
        Assumes input is of dimensionality nt x nf, where nt is the number of time points and
        nf is the number of features.

        lags is an array of integers, representing the time lag from zero. a negative time lag points to the future,
        positive to the past.
    """

    nt = input.shape[0]
    nf = input.shape[1]
    d = len(lags)

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


def fit_strf(input, output, lags, params={'solver':'lasso', 'lambda1':1.0, 'lambda2':1.0}):

    #convert the input into a toeplitz-like matrix
    A = make_toeplitz(input, lags, include_bias=True)

    """
    #pre-compute A'A
    p = A.shape[1]
    AA = np.zeros([p, p])
    for k in range(p):
        AA[:, k] = np.dot(A.T, A[:, k])

    #pre-compute A'y
    Ay = np.dot(A.T, output)
    """

    fy = np.asfortranarray(output.reshape(len(output), 1))
    fA = np.asfortranarray(A)
    #print 'fy.shape=',fy.shape
    #print 'fA.shape=',fA.shape

    #fit the STRF
    strf = spams.lasso(fy, fA, mode=2, lambda1=params['lambda1'], lambda2=params['lambda2'])

    #reshape the STRF so that it makes sense
    nt = input.shape[0]
    nf = input.shape[1]
    d = len(lags)
    strf = np.array(strf.todense()).reshape([nf, d])

    return strf
