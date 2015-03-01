import time

from abc import ABCMeta,abstractmethod
import copy

import numpy as np
import pandas as pd

from scipy.fftpack import fft,fftfreq
from scipy.signal import hilbert
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt

import nitime.algorithms as ntalg
from nitime import utils as ntutils
from lasp.signal import lowpass_filter, bandpass_filter

from brian import hears, Hz


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

    def __init__(self, bandwidth, adaptive=False, jackknife=False, max_adaptive_iter=150):
        PowerSpectrumEstimator.__init__(self)
        self.bandwidth = bandwidth
        self.jackknife = jackknife
        self.adaptive = adaptive
        self.max_adaptive_iter = max_adaptive_iter

    def get_frequencies(self, signal_length, sample_rate):
        cspec_freq = fftfreq(signal_length, d=1.0/sample_rate)
        nz = cspec_freq >= 0.0
        return cspec_freq[nz]

    def estimate(self, signal, sample_rate, debug=False):

        slen = len(signal)

        #compute DPSS tapers for signals
        NW = max(1, int((slen / sample_rate)*self.bandwidth))
        K = 2*NW - 1

        tapers, eigs = ntalg.dpss_windows(slen, NW, K)
        ntapers = len(tapers)
        if debug:
            print '[MultiTaperSpectrumEstimator.estimate] slen=%d, NW=%d, K=%d, bandwidth=%0.1f, ntapers: %d' % (slen, NW, K, self.bandwidth, ntapers)

        #compute a set of tapered signals
        s_tap = tapers * signal

        #compute the FFT of each tapered signal
        s_fft = fft(s_tap, axis=1)

        #throw away negative frequencies of the spectrum
        cspec_freq = fftfreq(slen, d=1.0/sample_rate)
        nz = cspec_freq >= 0.0
        s_fft = s_fft[:, nz]
        flen = nz.sum()
        cspec_freq = cspec_freq[nz]
        #print '(1)cspec_freq.shape=',cspec_freq.shape
        #print '(1)s_fft.shape=',s_fft.shape

        #determine the weights used to combine the tapered signals
        if self.adaptive and ntapers > 1:
            #compute the adaptive weights
            weights,weights_dof = ntutils.adaptive_weights(s_fft, eigs, sides='twosided', max_iter=self.max_adaptive_iter)
        else:
            weights = np.ones([ntapers, flen]) / float(ntapers)

        #print '(1)weights.shape=',weights.shape

        def make_spectrum(signal, signal_weights):
            denom = (signal_weights**2).sum(axis=0)
            return (np.abs(signal * signal_weights)**2).sum(axis=0) / denom

        if self.jackknife:
            #do leave-one-out cross validation to estimate the complex mean and standard deviation of the spectrum
            cspec_mean = np.zeros([flen], dtype='complex')
            for k in range(ntapers):
                index = range(ntapers)
                del index[k]
                #compute an estimate of the spectrum using all but the kth weight
                cspec_est = make_spectrum(s_fft[index, :], weights[index, :])
                cspec_diff = cspec_est - cspec_mean
                #do an online update of the mean spectrum
                cspec_mean += cspec_diff / (k+1)
        else:
            #compute the average complex spectrum weighted across tapers
            cspec_mean = make_spectrum(s_fft, weights)

        return cspec_freq,cspec_mean.squeeze()


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

    nincrement = int(np.round(sample_rate*increment))
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
        #print 'k=%d' % k
        #print 'si=%d, ei=%d' % (si, ei)
        #print 'spec_freq.shape=',spec_freq.shape
        #print 'tf.shape=',tf.shape
        #print 'est.shape=',est.shape
        tf[:, k] = est[findex]

    t = np.arange(0, nwindows, 1.0) * increment

    return t, freq, tf


def generate_sliding_windows(N, sample_rate, increment, window_length):
    """
        Generate a list of indices representing windows into a signal of length N.
    """

    #compute lengths in # of samples
    nwinlen = int(sample_rate*window_length)
    if nwinlen % 2 == 0:
        nwinlen += 1
    hnwinlen = nwinlen / 2

    nincrement = int(np.round(sample_rate*increment))
    nwindows = N / nincrement

    windows = list()
    for k in range(nwindows):
        center = k*nincrement
        si = center - hnwinlen
        ei = center + hnwinlen + 1
        windows.append( (center, si, ei) )

    #compute the centers of each window
    t = np.arange(0, nwindows, 1.0) * increment

    return t, np.array(windows)


def gaussian_stft(s, sample_rate, window_length, increment, min_freq=0,
                  max_freq=None, nstd=6):
    spectrum_estimator = GaussianSpectrumEstimator(nstd=nstd)
    t, freq, tf = timefreq(s, sample_rate, window_length, increment,
                           spectrum_estimator=spectrum_estimator,
                           min_freq=min_freq, max_freq=max_freq)
    ps = np.abs(tf)
    rms = ps.sum(axis=0)
    return t, freq, tf, rms


def mt_stft(s, sample_rate, window_length, increment, bandwidth=None, min_freq=0, max_freq=None, adaptive=True, jackknife=False):
    spectrum_estimator = MultiTaperSpectrumEstimator(bandwidth=bandwidth, adaptive=adaptive, jackknife=jackknife)
    return timefreq(s, sample_rate, window_length, increment, spectrum_estimator=spectrum_estimator, min_freq=min_freq, max_freq=max_freq)


class TimeFrequencyReassignment(object):

    def __init__(self):
        pass

    @abstractmethod
    def reassign(self, spec_t, spec_freq, spec):
        raise NotImplementedError('Use a subclass!')


class AmplitudeReassignment(object):
    """
        NOTE: doesn't work...
    """

    def __init__(self):
        pass

    def reassign(self, spec_t, spec_f, spec):

        #get power spectrum
        ps = np.abs(spec)

        #take the spectral and temporal derivatives
        dt = spec_t[1] - spec_t[0]
        df = spec_f[1] - spec_f[0]
        ps_df,ps_dt = np.gradient(ps)
        ps_df /= df
        ps_dt /= dt

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.hist(ps_df.ravel(), bins=15)
        plt.title('ps_df')
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.hist(ps_dt.ravel(), bins=15)
        plt.title('ps_dt')
        plt.axis('tight')

        #construct the empty reassigned time frequency representation
        ps_r = np.zeros_like(ps)
        for k,freq in enumerate(spec_f):
            for j,t in enumerate(spec_t):
                inst_freq = ps_df[k, j]
                group_delay = ps_dt[k, j]
                print 'inst_freq=%0.6f, group_delay=%0.6f' % (inst_freq, group_delay)
                fnew = freq + inst_freq
                tnew = group_delay + t
                print 'fnew=%0.0f, tnew=%0.0f' % (fnew, tnew)
                row = np.array(np.nonzero(spec_f <= fnew)).max()
                col = np.array(np.nonzero(spec_t <= tnew)).max()
                print 'row=',row
                print 'col=',col
                ps_r[row, col] += 1.0

        ps_r /= len(spec_t)*len(spec_f)

        return ps_r


class PhaseReassignment(object):
    """
        NOTE: doesn't work...
    """

    def __init__(self):
        pass

    def reassign(self, spec_t, spec_f, spec):

        #get phase
        phase = np.angle(spec)

        #take the spectral and temporal derivatives
        dt = spec_t[1] - spec_t[0]
        df = spec_f[1] - spec_f[0]
        ps_df,ps_dt = np.gradient(phase)
        ps_df /= df
        ps_dt /= dt

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.hist(ps_df.ravel(), bins=15)
        plt.title('ps_df')
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.hist(ps_dt.ravel(), bins=15)
        plt.title('ps_dt')
        plt.axis('tight')

        #construct the empty reassigned time frequency representation
        ps_r = np.zeros_like(phase)

        for k,freq in enumerate(spec_f):
            for j,t in enumerate(spec_t):
                tnew = max(0, t - (ps_df[k, j] / (2*np.pi)))
                fnew = max(0, ps_dt[k, j] / (2*np.pi))
                print 'fnew=%0.0f, tnew=%0.0f' % (fnew, tnew)
                row = np.array(np.nonzero(spec_f <= fnew)).max()
                col = np.array(np.nonzero(spec_t <= tnew)).max()
                print 'row=',row
                print 'col=',col
                ps_r[row, col] += 1.0

        ps_r /= len(spec_t)*len(spec_f)

        return ps_r


def log_spectrogram(spec):
    """
        Compute the log spectrogram.
    """
    lspec = np.zeros_like(spec)

    nz = spec > 0.0
    lspec[nz] = np.log10(spec[nz])
    lspec *= 10
    lspec += 100

    return lspec


def bandpass_timefreq(s, frequencies, sample_rate):
    """
        Bandpass filter signal s at the given frequency bands, and then use the Hilber transform
        to produce a complex-valued time-frequency representation of the bandpass filtered signal.
    """

    freqs = sorted(frequencies)
    tf_raw = np.zeros([len(frequencies), len(s)], dtype='float')
    tf_freqs = list()

    for k,f in enumerate(freqs):
        #bandpass filter signal
        if k == 0:
            tf_raw[k, :] = lowpass_filter(s, sample_rate, f)
            tf_freqs.append( (0.0, f) )
        else:
            tf_raw[k, :] = bandpass_filter(s, sample_rate,  freqs[k-1], f)
            tf_freqs.append( (freqs[k-1], f) )

    #compute analytic signal
    tf = hilbert(tf_raw, axis=1)
    #print 'tf_raw.shape=',tf_raw.shape
    #print 'tf.shape=',tf.shape

    return np.array(tf_freqs),tf_raw,tf


def resample_spectrogram(t, freq, spec, dt_new, df_new):

    #print 'len(t)=%d, len(freq)=%d, spec.shape=(%d, %d)' % (len(t), len(freq), spec.shape[0], spec.shape[1])
    spline = RectBivariateSpline(freq, t, spec)

    ntnew = int(np.ceil((t.max() - t.min()) / dt_new))
    nfnew = int(np.ceil((freq.max() - freq.min()) / df_new))

    tnew = np.arange(ntnew)*dt_new
    fnew = np.arange(nfnew)*df_new

    new_spec = spline(fnew, tnew)

    return tnew,fnew,new_spec


def compute_mean_spectrogram(s, sample_rate, win_sizes, increment=None, num_freq_bands=100,
                             spec_estimator=GaussianSpectrumEstimator(nstd=6), mask=False, mask_gain=3.0):
    """
        Compute a spectrogram for each time window, and average across time windows to get better time-frequency
        resolution. Post-processing is done with applying the log to change the power spectrum to decibels, and
        then a hard threshold is applied to zero-out the lowest 10% of the pixels.
    """

    #compute spectrograms
    stime = time.time()
    timefreqs = list()
    for k,win_size in enumerate(win_sizes):
        if increment is None:
            inc = win_sizes[0] / 2
        else:
            inc = increment
        t,freq,tf = timefreq(s, sample_rate, win_size, inc, spec_estimator)
        ps = np.abs(tf)
        ps_log = log_spectrogram(ps)
        timefreqs.append( (t, freq, ps_log) )
    etime = time.time() - stime
    #print 'time to compute %d spectrograms: %0.6fs' % (len(win_sizes), etime)

    #compute the mean spectrogram across window sizes
    nyquist_freq = sample_rate / 2.0
    df = nyquist_freq / num_freq_bands
    f_smallest = np.arange(num_freq_bands)*df
    t_smallest = timefreqs[0][0]  # best temporal resolution
    df_smallest = f_smallest[1] - f_smallest[0]
    dt_smallest = t_smallest[1] - t_smallest[0]

    #resample the spectrograms so they all have the same frequency spacing
    stime = time.time()
    rs_specs = list()
    for t,freq,ps in timefreqs:
        rs_t,rs_freq,rs_ps = resample_spectrogram(t, freq, ps, dt_smallest, df_smallest)
        rs_specs.append(rs_ps)
    etime = time.time() - stime
    #print 'time to resample %d spectrograms: %0.6fs' % (len(win_sizes), etime)

    #get the shortest spectrogram length
    min_freq_len = np.min([rs_ps.shape[0] for rs_ps in rs_specs])
    min_t_len = np.min([rs_ps.shape[1] for rs_ps in rs_specs])
    rs_specs_arr = np.array([rs_ps[:min_freq_len, :min_t_len] for rs_ps in rs_specs])
    t_smallest = np.arange(min_t_len)*dt_smallest
    f_smallest = np.arange(min_freq_len)*df_smallest

    #compute mean, std, and zscored power spectrum across window sizes
    tf_mean = rs_specs_arr.mean(axis=0)

    if mask:
        #compute the standard deviation across window sizes
        tf_std = rs_specs_arr.std(axis=0, ddof=1)
        #compute something that is close to the maximum std. we use the 95th pecentile to avoid outliers
        tf_std /= np.percentile(tf_std.ravel(), 95)
        #compute a sigmoidal mask that will zero out pixels in tf_mean that have high standard deviations
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(1)**(-mask_gain*x))
        sigmoid_mask = 1.0 - sigmoid(tf_std)
        #mask the mean time frequency representation
        tf_mean *= sigmoid_mask

    return t_smallest, f_smallest, tf_mean


def define_f_bands(stt=180, stp=7000, n_bands=32, kind='log'):
    '''
    Defines log-spaced frequency bands...generally this is for auditory
    spectrogram extraction. Brian used 180 - 7000 Hz, so for now those
    are the defaults.

    INPUTS
    --------
        stt : int
            The starting frequency
        stp : int
            The end frequency
        n_bands : int
            The number of bands to calculate
        kind : string, ['log', 'erb']
            What kind of spacing will we use for the frequency bands.
    '''
    if kind == 'log':
        aud_fs = np.logspace(np.log10(stt), np.log10(stp), n_bands).astype(int)
    elif kind == 'erb':
        aud_fs = hears.erbspace(stt*Hz, stp*Hz, n_bands)
    else:
        raise NameError("I don't know what kind of spacing that is")
    return aud_fs


def extract_nsl_spectrogram(sig, Fs, cfs):
    '''Implements a version of the "wav2aud" function in the NSL toolbox.
    Uses Brian hears to chain most of the computations to be done online.

    This is effectively what it does:
        1. Gammatone filterbank at provided cfs (erbspace recommended)
        2. Half-wave rectification
        3. Low-pass filtering at 2Khz
        4. First-order derivative across frequencies (basically just
            taking the diff of successive frequencies to sharpen output)
        5. Half-wave rectification #2
        6. An exponentially-decaying average, with time constant chosen
            to be similar to that reported in the NSL toolbox (8ms)

    INPUTS
    --------
    sig : array
        The auditory signals we'll use to extract. Should be time x feats, or 1-d
    Fs : float, int
        The sampling rate of the signal
    cfs : list of floats, ints
        The center frequencies that we'll use for initial filtering.

    OUTPUTS
    --------
    out : array, [tpts, len(cfs)]
        The auditory spectrogram of the signal
    '''
    Fs = float(Fs)*Hz
    snd = hears.Sound(sig, samplerate=Fs)

    # Cochlear model
    snd_filt = hears.Gammatone(snd, cfs)

    # Hair cell stages
    clp = lambda x: np.clip(x, 0, np.inf)
    snd_hwr = hears.FunctionFilterbank(snd_filt, clp)
    snd_lpf = hears.LowPass(snd_hwr, 2000)

    # Lateral inhibitory network
    rands = lambda x: sigp.roll_and_subtract(x, hwr=True)
    snd_lin = hears.FunctionFilterbank(snd_lpf, rands)

    # Initial processing
    out = snd_lin.process()

    # Time integration.
    # Time constant is 8ms, which we approximate with halfwidth of 12
    half_pt = (12. / 1000) * Fs
    out = pd.stats.moments.ewma(out, halflife=half_pt)
    return out


def roll_and_subtract(sig, amt=1, axis=1, hwr=False):
    '''Rolls the input matrix along the specifies axis, then
    subtracts this from the original signal. This is meant to
    be similar to the lateral inhibitory network from Shamma's
    NSL toolbox. hwr specifies whether to include a half-wave
    rectification after doing the subtraction.'''
    diff = np.roll(sig, -amt, axis=axis)
    diff[:, -amt:] = 0
    diff = np.subtract(sig, diff)
    if hwr is True:
        diff = np.clip(diff, 0, np.inf)
    return diff
