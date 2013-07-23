from abc import ABCMeta,abstractmethod

import numpy as np

from scipy.fftpack import fft,fftfreq
from scipy.signal import lfilter, filter_design, resample

import nitime.algorithms as ntalg


def lowpass_filter(s, sample_rate, cutoff_freq, filter_order=5):
    """
        Lowpass filter a signal s, with sample rate sample_rate.

        s: the signal
        sample_rate: the sample rate in Hz of the signal
        cutoff_freq: the cutoff frequency of the filter
        filter_order: the order of the filter...

        Returns the low-pass filtered signal s.
    """

    #create a butterworth filter
    nyq = sample_rate / 2.0
    b,a = filter_design.butter(filter_order, cutoff_freq / nyq)

    #filter the signal
    filtered_s = lfilter(b, a, s)

    #rescale filtered signal
    filtered_s /= filtered_s.max()
    filtered_s *= s.max()

    return filtered_s


def resample_signal(s, sample_rate, desired_sample_rate):
    """
        Resamples a signal from sample rate to desired_sample_rate.

        s: the signal
        sample_rate: the sample rate of the signal
        desired_sample_rate: the desired sample rate

        Returns t_rs,rs where t_rs is the time corresponding to each resampled point, rs is the resampled sigal.
    """

    duration = float(len(s)) / sample_rate
    t = np.arange(len(s)) * (1.0 / sample_rate)
    desired_n = int(duration*desired_sample_rate)
    rs,t_rs = resample(s, desired_n, t=t)
    return t_rs,rs


class PowerSpectrumEstimator(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def estimate(self, signal, sample_rate):
        return NotImplementedError('Use a subclass of PowerSpectrumEstimator!')


class GaussianWindowSpectrumEstimator(PowerSpectrumEstimator):

    def __init__(self, nstd=6):
        PowerSpectrumEstimator.__init__(self)
        self.nstd = nstd

    def estimate(self, signal, sample_rate):
        nwinlen = len(signal)
        if nwinlen % 2 == 0:
            nwinlen += 1
        hnwinlen = nwinlen / 2

        #construct the window
        gauss_t = np.arange(-hnwinlen, hnwinlen+1, 1.0)
        gauss_std = float(nwinlen) / float(self.nstd)
        self.gauss_window = np.exp(-gauss_t**2 / (2.0*gauss_std**2)) / (gauss_std*np.sqrt(2*np.pi))

        #window the signal and take the FFT
        windowed_slice = signal*self.gauss_window
        fft_len = len(signal)
        s_fft = fft(windowed_slice, n=fft_len, overwrite_x=1)
        freq = fftfreq(fft_len, d=1.0 / sample_rate)

        return freq,np.abs(s_fft)


class MultiTaperSpectrumEstimator(PowerSpectrumEstimator):


    def __init__(self, bandwidth=None, adaptive=True, jackknife=False, low_bias=False):
        PowerSpectrumEstimator.__init__(self)
        self.adaptive = adaptive
        self.jackknife = jackknife
        self.low_bias = low_bias
        if bandwidth is None:
            pass

    def estimate(self, signal, sample_rate):
        mt_freqs,mt_ps,var = ntalg.multi_taper_psd(signal, Fs=sample_rate, adaptive=self.adaptive, jackknife=self.jackknife,
                                                   low_bias=self.low_bias, sides='onesided')
        return mt_freqs,mt_ps


def stft(s, sample_rate, window_length, increment, spectrum_estimator, min_freq=0, max_freq=None):
    """
        Compute the spectrogram of a signal s.

        s: the raw waveform.
        sample_rate: the sample rate of the waveform
        increment: the spacing in seconds between points where the spectrum is computed, i.e. inverse of the spectrogram sample rate
        spectrum_estimator: an instance of PowerSpectrumEstimator
        min_freq: the minimum frequency to analyze
        max_freq: the maximum frequency to analyze

        Returns t,freq,spec,rms:

        t: the time axis of the spectrogram
        freq: the frequency axis of the spectrogram
        spec: the log spectrogram
        rms: the running root-mean-square of the sound pressure waveform
    """

    if max_freq is None:
        max_freq = sample_rate / 2.0

    #compute lengths in # of samples
    nwinlen = int(sample_rate*window_length)
    if nwinlen % 2 == 0:
        nwinlen += 1
    hnwinlen = nwinlen / 2

    nincrement = int(sample_rate*increment)
    nwindows = len(s) / nincrement
    #print 'len(s)=%d, nwinlen=%d, hwinlen=%d, nincrement=%d, nwindows=%d' % (len(s), nwinlen, hnwinlen, nincrement, nwindows)

    #pad the signal with zeros
    zs = np.zeros([len(s) + 2*hnwinlen])
    zs[hnwinlen:-hnwinlen] = s

    #get the frequencies corresponding to the FFTs to come
    full_freq = fftfreq(nwinlen, d=1.0 / sample_rate)
    freq_index = (full_freq >= min_freq) & (full_freq <= max_freq)
    freq = full_freq[freq_index]
    nfreq = freq_index.sum()

    #take the FFT of each segment, padding with zeros when necessary to keep window length the same
    timefreq = np.zeros([nfreq, nwindows], dtype='float')
    rms = np.zeros([nwindows])
    for k in range(nwindows):
        center = k*nincrement + hnwinlen
        si = center - hnwinlen
        ei = center + hnwinlen + 1
        rms[k] = zs[si:ei].std(ddof=1)
        ps_freq,ps = spectrum_estimator.estimate(zs[si:ei], sample_rate)
        findex = (ps_freq <= max_freq) & (ps_freq >= min_freq)
        timefreq[:, k] = ps[findex]

    t = np.arange(0, nwindows, 1.0) * increment

    return t,freq,timefreq,rms


def gaussian_stft(s, sample_rate, window_length, increment, min_freq=0, max_freq=None, nstd=6):
    spectrum_estimator = GaussianWindowSpectrumEstimator(nstd=nstd)
    return stft(s, sample_rate, window_length, increment, spectrum_estimator=spectrum_estimator, min_freq=min_freq, max_freq=max_freq)


def mt_stft(s, sample_rate, window_length, increment, bandwidth=None, min_freq=0, max_freq=None, adaptive=True, jackknife=False, low_bias=False):
    spectrum_estimator = MultiTaperSpectrumEstimator(bandwidth=bandwidth, adaptive=True, jackknife=False, low_bias=False)
    return stft(s, sample_rate, window_length, increment, spectrum_estimator=spectrum_estimator, min_freq=min_freq, max_freq=max_freq)





