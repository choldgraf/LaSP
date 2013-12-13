import unittest

import numpy as np

import matplotlib.pyplot as plt
from tools.sound import plot_spectrogram

from tools.timefreq import GaussianSpectrumEstimator,MultiTaperSpectrumEstimator,timefreq


class TestTimeFreq(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_timefreq(self):

        #create a monocomponent signal
        np.random.seed(12345)
        sr = 381.4697
        dt = 1.0 / sr
        duration = 20.0 + dt

        t = np.arange(0, int(duration*sr))*dt

        #create a monocomponent signal
        f1 = 30.0
        s1 = np.sin(2*np.pi*t*f1)

        compare_timefreqs(s1, sr)
        plt.show()


def compare_timefreqs(s, sample_rate, win_sizes=[0.050, 0.100, 0.250, 0.500, 1.25]):
    """
        Compare the time frequency representation of a signal using different window sizes and estimators.
    """

    #construct different types of estimators
    gaussian_est = GaussianSpectrumEstimator(nstd=6)
    mt_est_lowbw = MultiTaperSpectrumEstimator(bandwidth=10.0, adaptive=False)
    mt_est_lowbw_adapt = MultiTaperSpectrumEstimator(bandwidth=10.0, adaptive=True)
    mt_est_highbw = MultiTaperSpectrumEstimator(bandwidth=30.0, adaptive=False)
    mt_est_hibhbw_adapt = MultiTaperSpectrumEstimator(bandwidth=30.0, adaptive=True)
    #estimators = [gaussian_est, mt_est_lowbw, mt_est_lowbw_adapt, mt_est_highbw, mt_est_hibhbw_adapt]
    estimators = [gaussian_est, mt_est_lowbw, mt_est_lowbw_adapt]
    enames = ['gauss', 'lowbw', 'lowbw_a', 'highbw', 'highbw_a']

    #run each estimator for each window size and plot the amplitude of the time frequency representation
    plt.figure()
    spnum = 1
    for k,win_size in enumerate(win_sizes):
        increment = win_size / 2.0
        for j,est in enumerate(estimators):
            print 'window_size=%dms, estimator=%s' % (win_size*1000, enames[j])
            t,freq,tf = timefreq(s, sample_rate, win_size, increment, est)
            ax = plt.subplot(len(win_sizes), len(estimators), spnum)
            plot_spectrogram(t, freq, np.abs(tf), ax=ax, colorbar=False, ticks=False)
            if k == 0:
                plt.title(enames[j])
            if j == 0:
                plt.ylabel('%d ms' % (win_size*1000))
            spnum += 1
