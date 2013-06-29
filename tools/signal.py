import numpy as np

from scipy.fftpack import fft,fftfreq
from scipy.signal import lfilter, filter_design, resample


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


def gaussian_stft(s, sample_rate, window_length, increment, nstd=6, min_freq=0, max_freq=None):
    """
        Compute the spectrogram of a signal s with a Gaussian window.

        s: the raw waveform.
        sample_rate: the sample rate of the waveform
        increment: the spacing in seconds between points where the spectrum is computed, i.e. inverse of the spectrogram sample rate
        freq_spacing: the spacing in Hz between frequency bands
        min_freq: the minimum frequency to analyze
        max_freq: the maximum frequency to analyze
        nstd: number of standard deviations for Gaussian window centered at each point

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

    #construct the window
    gauss_t = np.arange(-hnwinlen, hnwinlen, 1.0)
    gauss_std = float(nwinlen) / float(nstd)
    gauss_window = np.exp(-gauss_t**2 / (2.0*gauss_std**2)) / (gauss_std*np.sqrt(2*np.pi))

    #pad the signal with zeros
    zs = np.zeros([len(s) + 2*hnwinlen])
    zs[hnwinlen:-hnwinlen] = s

    #get the frequencies corresponding to the FFTs to come
    fft_len = nwinlen+1
    full_freq = fftfreq(nwinlen+1, d=1.0 / sample_rate)
    freq_index = (full_freq >= min_freq) & (full_freq <= max_freq)
    freq = full_freq[freq_index]
    nfreq = freq_index.sum()

    #take the FFT of each segment, padding with zeros when necessary to keep window length the same
    timefreq = np.zeros([nfreq, nwindows], dtype='complex')
    rms = np.zeros([nwindows])
    for k in range(nwindows):
        center = k*nincrement + hnwinlen
        si = center - hnwinlen
        ei = center + hnwinlen
        rms[k] = zs[si:ei].std(ddof=1)
        windowed_slice = zs[si:ei]*gauss_window
        zs_fft = fft(windowed_slice, n=fft_len, overwrite_x=1)
        timefreq[:, k] = zs_fft[freq_index]

    t = np.arange(0, nwindows, 1.0) * increment

    return t,freq,timefreq,rms


