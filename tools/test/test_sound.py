import tempfile
import unittest
from tools.sound import generate_harmonic_stack,WavFile,play_sound,generate_sine_wave,generate_simple_stack,mps,plot_mps,modulate_wave

import numpy as np
from scipy.fftpack import fft2,fftshift,fftfreq

import matplotlib.pyplot as plt
import matplotlib.cm as cmap


class TestSound(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mps1(self):

        #generate a simple harmonic stack
        sample_rate = 44100.0
        #max_freq = 440.0 * 2**4
        max_freq = 1000.0
        wf = WavFile()
        #wf.data = generate_harmonic_stack(0.500, 440.0, sample_rate, 5, base=2)
        wf.data = generate_simple_stack(1.000, 200.0, sample_rate, num_harmonics=5)
        wf.sample_rate = sample_rate
        wf.num_channels = 1
        wf.analyze(max_freq=max_freq, freq_spacing=100.0, noise_level_db=30.0)
        wf.plot(max_freq=max_freq, colormap=cmap.jet)

        #write it to a file and play it
        output_file = tempfile.mktemp('wav', 'test_mps')
        wf.to_wav(output_file, normalize=True)
        #play_sound(output_file)

        #compute the MPS
        df = np.diff(wf.spectrogram_f)[0]
        dt = np.diff(wf.spectrogram_t)[0]
        temporal_freq,spectral_freq,mps_logamp,mps_phase = mps(wf.spectrogram, df, dt)

        #plot it
        plot_mps(temporal_freq, spectral_freq, mps_logamp, mps_phase)

        #show the figures
        plt.show()

    def test_mps2(self):

        #generate a simple harmonic stack
        sample_rate = 44100.0
        #max_freq = 440.0 * 2**4
        max_freq = 1000.0
        wf = WavFile()
        #wf.data = generate_harmonic_stack(0.500, 440.0, sample_rate, 5, base=2)
        s = generate_simple_stack(4, 200.0, sample_rate, num_harmonics=5)
        cs = modulate_wave(s, sample_rate, freq=2.0)
        wf.data = cs

        wf.sample_rate = sample_rate
        wf.num_channels = 1
        wf.analyze(max_freq=max_freq, freq_spacing=100.0, noise_level_db=30.0)
        wf.plot(max_freq=max_freq, colormap=cmap.jet)

        #write it to a file and play it
        output_file = tempfile.mktemp('wav', 'test_mps')
        wf.to_wav(output_file, normalize=True)
        #play_sound(output_file)

        #compute the MPS
        df = np.diff(wf.spectrogram_f)[0]
        dt = np.diff(wf.spectrogram_t)[0]
        temporal_freq,spectral_freq,mps_logamp,mps_phase = mps(wf.spectrogram, df, dt)

        #plot it
        plot_mps(temporal_freq, spectral_freq, mps_logamp, mps_phase)

        #show the figures
        plt.show()
