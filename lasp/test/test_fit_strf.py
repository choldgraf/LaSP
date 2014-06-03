from unittest import TestCase

import numpy as np

import matplotlib.pyplot as plt
from lasp.sound import plot_spectrogram

from lasp.fit_strf import *
from lasp.incrowd import fast_conv


class TestFitStrf(object):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fit_strf(self):

        sample_rate = 381.0
        #T = 27*10*int(2.0*sample_rate)
        #nf = 77
        T = 20*5*int(2.0*sample_rate)
        nf = 20
        d = int(0.100*sample_rate)
        lags = np.arange(d)

        print 'T=%d, nf=%d, d=%d, nparams=%d' % (T, nf, d, nf*d)

        #create fake input
        X = np.random.randn(T, nf)

        #create sparse STRF
        strf = np.random.randn(nf, d)
        strf[np.abs(strf) < 0.5] = 0.0
        bias = -1.5

        #create fake response
        y = fast_conv(X, strf, lags, bias)

        #fit the STRF
        strf_lasso,bias_lasso = fit_strf_lasso(X, y, lags, lambda1=1.0)
        strf_ridge,bias_ridge = fit_strf_ridge(X, y, lags, alpha=1.0)

        print 'biases=',(bias, bias_lasso, bias_ridge)

        absmax = max(np.abs(strf).max(), np.abs(strf_ridge).max(), np.abs(strf_ridge).max())

        plt.figure()
        ax = plt.subplot(3, 1, 1)
        plot_spectrogram(lags, np.arange(nf), strf, ax=ax, vmin=-absmax, vmax=absmax)
        plt.title('Real STRF')

        ax = plt.subplot(3, 1, 2)
        plot_spectrogram(lags, np.arange(nf), strf_lasso, ax=ax, vmin=-absmax, vmax=absmax)
        plt.title('Lasso STRF')

        ax = plt.subplot(3, 1, 3)
        plot_spectrogram(lags, np.arange(nf), strf_ridge, ax=ax, vmin=-absmax, vmax=absmax)
        plt.title('Ridge STRF')

        plt.show()


if __name__ == '__main__':

    tc = TestFitStrf()
    tc.test_fit_strf()

