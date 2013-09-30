from scipy.signal import hilbert
import unittest

import numpy as np

import matplotlib.pyplot as plt

from tools.hht import HHT


class TestHHT(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    """
    def test_sifting(self):

        #create a sine wave as a test signal
        dt = 1e-6
        duration = 1.0
        t = np.arange(0.0, duration, dt)
        f = 10.0
        s = np.sin(2*np.pi*t*f)

        #compute a single IMF
        hht = HHT()
        imf = hht.compute_imf(s, remove_edge_effects=True, max_iter=5)

        assert imf.mean() < 1e-6

        #create a more complex time series
        s = np.zeros([len(t)])
        for f in [5, 10, 15, 35]:
            s += np.sin(2*np.pi*t*f)

        hht = HHT()
        imf = hht.compute_imf(s, remove_edge_effects=True, max_iter=10)

        assert imf.mean() < 1e-6

        #create a chaotic time series
        duration = 0.00050
        t = np.arange(0.0, duration, dt)
        s = np.zeros([len(t)])
        s[0] = 0.5
        for k in range(1, len(t)):
            s[k] = 3.56997*s[k-1]*(1.0 - s[k-1])

        hht = HHT()
        imf = hht.compute_imf(s, mean_tol=1e-3, remove_edge_effects=True, max_iter=100)

        assert imf.mean() < 1e-3

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, s, 'k-')
        plt.title('Signal')
        plt.subplot(2, 1, 2)
        plt.plot(t, imf, 'b-')
        plt.title('First IMF')
        plt.show()

    def test_emd(self):
        #create a sine wave as a test signal
        dt = 1e-6
        duration = 1.0
        t = np.arange(0.0, duration, dt)

        #create a more complex time series
        s = np.zeros([len(t)])
        for f in [5, 10, 15, 35]:
            s += np.sin(2*np.pi*t*f)

        hht = HHT()
        hht.compute_emd(s)

        assert len(hht.imfs) > 1

        n = len(hht.imfs)
        plt.figure()
        plt.subplot(n+1, 1, 1)
        plt.plot(t, s, 'k-')
        plt.title('Signal')
        plt.axis('tight')
        for k in range(n):
            plt.subplot(n+1, 1, k+2)
            plt.plot(t, hht.imfs[k], 'b-')
            plt.axis('tight')
        plt.show()
    """

    def test_timefreq(self):

        #create a sine wave as a test signal
        dt = 1e-6
        sr = 1.0 / dt
        duration = 1.0
        t = np.arange(0.0, duration, dt)

        #create a more complex time series
        s = np.zeros([len(t)])
        for f in [5, 10, 15, 35]:
            s += np.sin(2*np.pi*t*f)

        hht = HHT(sample_rate=sr)
        hht.compute_emd(s)

        assert len(hht.imfs) > 1

        for k,imf in enumerate(hht.imfs):
            am,fm,phase,ifreq = hht.decompose_imf(imf)
            plt.figure()

            plt.subplot(6, 1, 1)
            plt.plot(t, s, 'k-')
            plt.axis('tight')
            plt.title('Signal')

            plt.subplot(6, 1, 2)
            plt.plot(t, imf, 'b-')
            plt.axis('tight')
            plt.title('IMF #%d' % k)

            plt.subplot(6, 1, 3)
            plt.plot(t, am, 'r-')
            plt.axis('tight')
            plt.title('AM Component')

            plt.subplot(6, 1, 4)
            plt.plot(t, fm, 'g-')
            plt.axis('tight')
            plt.title('FM Component')

            plt.subplot(6, 1, 5)
            plt.plot(t, phase, 'k-')
            plt.axis('tight')
            plt.title('Phase')

            plt.subplot(6, 1, 6)
            plt.plot(t, ifreq, 'k-')
            plt.axis('tight')
            plt.title('Instantaneous Frequency')

        plt.show()