import unittest

import numpy as np
from matplotlib import cm

import matplotlib.pyplot as plt

from tools.signal import cross_coherence


class TestCrossCoherence(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple(self):

        sr = 381.4697
        dt = 1.0 / sr
        duration = 20.0 + dt

        t = np.arange(0, int(duration*sr))*dt

        #create original signal
        s1 = np.zeros_like(t)
        for f in 25.0,110.0:
            s1 += np.sin(2*np.pi*t*f)

        #noise corrupted signal
        s2 = s1 + np.random.randn(len(s1))*0.5

        #corrupt the signal at certain points
        i1 = int(5.0/dt)
        i2 = i1 + int(5.0/dt)
        s2[i1:i2] = np.random.randn(i2-i1)

        winsize = 1.0
        inc = 1.0
        bw = 10.0
        ct,cfreq,ctimefreq = cross_coherence(s1, s2, sr, window_size=winsize, increment=inc, bandwidth=bw)

        assert ctimefreq.shape[0] == len(cfreq)
        assert ctimefreq.shape[1] == len(ct)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, s1, 'k-')
        plt.plot(t, s2, 'r-', alpha=0.75)
        plt.axis('tight')
        plt.legend(['s1', 's2'])

        plt.subplot(2, 1, 2)
        plt.imshow(ctimefreq, interpolation='nearest', aspect='auto', extent=[ct.min(), ct.max(), cfreq.min(), cfreq.max()], cmap=cm.jet, origin='lower')
        plt.axis('tight')
        plt.colorbar()
        plt.show()






