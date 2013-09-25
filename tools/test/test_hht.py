import unittest

import numpy as np

import matplotlib.pyplot as plt

from tools.hht import HHT


class TestHHT(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sifting(self):

        #create a sine wave as a test signal
        dt = 1e-6
        duration = 1.0
        t = np.arange(0.0, duration, dt)
        f = 25.0
        s = np.sin(2*np.pi*t*f)

        #compute a single IMF
        hht = HHT()
        imf = hht.compute_imf(s, reflect=True)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, s, 'k-')
        plt.title('Signal')
        plt.subplot(2, 1, 2)
        plt.plot(t, imf, 'b-')
        plt.title('First IMF')
        plt.show()

        assert 1 == 2





