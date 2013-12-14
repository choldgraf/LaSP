from abc import ABCMeta,abstractmethod
import copy

import numpy as np

from scipy.fftpack import fft,fftfreq

import nitime.algorithms as ntalg
from nitime import utils as ntutils


class PowerSpectrumEstimator(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def estimate(self, signal, sample_rate):
        return NotImplementedError('Use a subclass of PowerSpectrumEstimator!')

    @abstractmethod
    def get_frequencies(self, signal_length, sample_rate):
        return NotImplementedError('Use a subclass of PowerSpectrumEstimator!')

class GaussianSpectrumEstimator(PowerSpectrumEstimator):

    def __init__(self, nstd=6):
        PowerSpectrumEstimator.__init__(self)
        self.nstd = nstd

    def get_frequencies(self, signal_length, sample_rate):
        freq = fftfreq(signal_length, d=1.0/sample_rate)
        nz = freq >= 0.0
        return freq[nz]

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
        fft_len = len(signal)
        windowed_slice = signal[:fft_len]*self.gauss_window[:fft_len]
        s_fft = fft(windowed_slice, n=fft_len, overwrite_x=1)
        freq = fftfreq(fft_len, d=1.0/sample_rate)
        nz = freq >= 0.0

        return freq[nz],s_fft[nz]


class MultiTaperSpectrumEstimator(PowerSpectrumEstimator):

    def __init__(self, bandwidth, adaptive=True, jackknife=False, max_adaptive_iter=150):
        PowerSpectrumEstimator.__init__(self)
        self.bandwidth = bandwidth
        self.jackknife = jackknife
        self.adaptive = adaptive
        self.max_adaptive_iter = max_adaptive_iter

    def get_frequencies(self, signal_length, sample_rate):
        cspec_freq = fftfreq(signal_length, d=1.0/sample_rate)
        nz = cspec_freq >= 0.0
        return cspec_freq[nz]

    def estimate(self, signal, sample_rate, debug=True):

        slen = len(signal)

        #compute DPSS tapers for signals
        NW = max(1, int((slen / sample_rate)*self.bandwidth))
        K = 2*NW - 1
        print 'compute_coherence: slen=%d, NW=%d, K=%d' % (slen, NW, K)
        tapers,eigs = ntalg.dpss_windows(slen, NW, K)
        ntapers = len(tapers)
        if debug:
            print '[MultiTaperSpectrumEstimator.estimate] bandwidth=%0.1f, # of tapers: %d' % (self.bandwidth, len(eigs))

        #compute a set of tapered signals
        s_tap = tapers * signal

        #compute the FFT of each tapered signal
        s_fft = fft(s_tap, axis=1)

        #determine the weights used to combine the tapered signals
        if self.adaptive and ntapers > 1:
            #compute the adaptive weights
            weights,weights_dof = ntutils.adaptive_weights(s_fft, eigs, sides='onesided', max_iter=self.max_adaptive_iter)
        else:
            weights = np.ones([ntapers, slen]) / float(ntapers)
        print 'weights.shape=',weights.shape

        #throw away negative frequencies of the spectrum
        cspec_freq = fftfreq(slen, d=1.0/sample_rate)
        nz = cspec_freq >= 0.0
        s_fft = s_fft[:, nz]
        flen = nz.sum()
        cspec_freq = cspec_freq[nz]
        print '(1)cspec_freq.shape=',cspec_freq.shape
        print '(1)s_fft.shape=',s_fft.shape

        if self.jackknife:
            #do leave-one-out cross validation to estimate the complex mean and standard deviation of the spectrum
            cspec_mean = np.zeros([flen], dtype='complex')
            for k in range(ntapers):
                index = range(ntapers)
                del index[k]
                #compute an estimate of the spectrum using all but the kth weight
                cspec_est = (s_fft[index, :] * weights[index, :]) / weights[index, :].sum()
                cspec_diff = cspec_est - cspec_mean
                #do an online update of the mean spectrum
                cspec_mean += cspec_diff / (k+1)
        else:
            #compute the average complex spectrum weighted across tapers
            cspec_mean = (s_fft * weights) / weights.sum()

        return cspec_freq,cspec_mean


def timefreq(s, sample_rate, window_length, increment, spectrum_estimator, min_freq=0, max_freq=None):
    """
        Compute a time-frequency representation of the signal s.

        s: the raw waveform.
        sample_rate: the sample rate of the waveform
        increment: the spacing in seconds between points where the spectrum is computed, i.e. inverse of the spectrogram sample rate
        spectrum_estimator: an instance of PowerSpectrumEstimator
        min_freq: the minimum frequency to analyze (Hz)
        max_freq: the maximum frequency to analyze (Hz)

        Returns t,freq,spec,rms:

        t: the time axis of the spectrogram
        freq: the frequency axis of the spectrogram
        tf: the time-frequency representation
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

    #get the values for the frequency axis by estimating the spectrum of a dummy slice
    full_freq = spectrum_estimator.get_frequencies(nwinlen, sample_rate)
    freq_index = (full_freq >= min_freq) & (full_freq <= max_freq)
    freq = full_freq[freq_index]
    nfreq = freq_index.sum()

    #take the FFT of each segment, padding with zeros when necessary to keep window length the same
    #tf = np.zeros([nfreq, nwindows], dtype='complex')
    tf = np.zeros([nfreq, nwindows], dtype='complex')
    for k in range(nwindows):
        center = k*nincrement + hnwinlen
        si = center - hnwinlen
        ei = center + hnwinlen + 1

        spec_freq,est = spectrum_estimator.estimate(zs[si:ei], sample_rate)
        findex = (spec_freq <= max_freq) & (spec_freq >= min_freq)
        print 'k=%d' % k
        print 'si=%d, ei=%d' % (si, ei)
        print 'spec_freq.shape=',spec_freq.shape
        print 'tf.shape=',tf.shape
        print 'est.shape=',est.shape
        tf[:, k] = est[findex]

    t = np.arange(0, nwindows, 1.0) * increment

    return t, freq, tf


def gaussian_stft(s, sample_rate, window_length, increment, min_freq=0, max_freq=None, nstd=6, return_phase=False):
    spectrum_estimator = GaussianSpectrumEstimator(nstd=nstd)
    return timefreq(s, sample_rate, window_length, increment, spectrum_estimator=spectrum_estimator, min_freq=min_freq, max_freq=max_freq)


def mt_stft(s, sample_rate, window_length, increment, bandwidth=None, min_freq=0, max_freq=None, adaptive=True, jackknife=False):
    spectrum_estimator = MultiTaperSpectrumEstimator(bandwidth=bandwidth, adaptive=True, jackknife=False, low_bias=False)
    return timefreq(s, sample_rate, window_length, increment, spectrum_estimator=spectrum_estimator, min_freq=min_freq, max_freq=max_freq)
