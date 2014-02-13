import numpy as np

from scipy.fftpack import fft,fftfreq,ifft
from scipy.signal import filter_design, resample,filtfilt

import nitime.algorithms as ntalg
import time
from tools.coherence import compute_mtcoherence


def lowpass_filter(s, sample_rate, cutoff_freq, filter_order=5, rescale=False):
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
    filtered_s = filtfilt(b, a, s)

    if rescale:
        #rescale filtered signal
        filtered_s /= filtered_s.max()
        filtered_s *= s.max()

    return filtered_s

def highpass_filter(s, sample_rate, cutoff_freq, filter_order=5, rescale=False):
    """
        Highpass filter a signal s, with sample rate sample_rate.

        s: the signal
        sample_rate: the sample rate in Hz of the signal
        cutoff_freq: the cutoff frequency of the filter
        filter_order: the order of the filter...

        Returns the low-pass filtered signal s.
    """

    #create a butterworth filter
    nyq = sample_rate / 2.0
    b,a = filter_design.butter(filter_order, cutoff_freq / nyq, btype='high')

    #filter the signal
    filtered_s = filtfilt(b, a, s)

    if rescale:
        #rescale filtered signal
        filtered_s /= filtered_s.max()
        filtered_s *= s.max()

    return filtered_s


def bandpass_filter(s, sample_rate, low_freq, high_freq, filter_order=5, rescale=False):
    """
        Bandpass filter a signal s.

        s: the signal
        sample_rate: the sample rate in Hz of the signal
        low_freq: the lower cutoff frequency
        upper_freq: the upper cutoff frequency
        filter_order: the order of the filter...

        Returns the bandpass filtered signal s.
    """


    #create a butterworth filter
    nyq = sample_rate / 2.0
    f = np.array([low_freq, high_freq]) / nyq
    b,a = filter_design.butter(filter_order, f, btype='bandpass')

    #filter the signal
    filtered_s = filtfilt(b, a, s)

    if rescale:
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


def cross_coherence(s1, s2, sample_rate, window_size=5.0, increment=1.0, bandwidth=10.0, noise_floor=False, num_noise_trials=1, debug=False):
    """
        Compute the running cross coherence between two time series.

        s1,s2: the signals

        sample_rate: sample rate in Hz for the signal

        window_size: the size in seconds of the sliding window used to compute the coherence

        increment: the amount of time in seconds to slide the window forward per time point

        bandwidth: related to the number of tapers used to compute the cross spectral density

        noise_floor: whether or not to compute a lower bound on the coherence for each time point. The lower bound
            is defined by the average coherence between two signals that have the same power spectrum as s1 and s2
            but randomly shuffled phases.
    """

    assert len(s1) == len(s2)

    #compute lengths in # of samples
    nwinlen = int(sample_rate*window_size)
    if nwinlen % 2 == 0:
        nwinlen += 1
    hnwinlen = nwinlen / 2

    #compute increment in number of samples
    slen = len(s1)
    nincrement = int(sample_rate*increment)

    #compute number of windows
    nwindows = slen / nincrement

    #get frequency axis values by computing coherence between dummy slice
    win1 = np.zeros([nwinlen])
    win2 = np.zeros([nwinlen])
    cdata = compute_mtcoherence(win1+1.0, win2+1.0, sample_rate, window_size=window_size, bandwidth=bandwidth)
    freq = cdata.frequency

    #construct the time-frequency representation for time-varying coherence
    timefreq = np.zeros([len(freq), nwindows])
    if noise_floor:
        floor_window_index_min = int(np.ceil(hnwinlen / float(nincrement)))
        floor_window_index_max = nwindows - floor_window_index_min
        timefreq_floor = np.zeros([len(freq), nwindows])

    if debug:
        print '[cross_coherence] length=%0.3f, slen=%d, window_size=%0.3f, increment=%0.3f, bandwidth=%0.1f, nwindows=%d' % \
              (slen/sample_rate, slen, window_size, increment, bandwidth, nwindows)

    #compute the coherence for each window
    #print 'nwinlen=%d, hnwinlen=%d, nwindows=%d' % (nwinlen, hnwinlen, nwindows)
    for k in range(nwindows):
        if debug:
            stime = time.time()

        #get the indices of the window within the signals
        center = k*nincrement
        si = center - hnwinlen
        ei = center + hnwinlen + 1

        #adjust indices to deal with edge-padding
        sii = 0
        if si < 0:
            sii = abs(si)
            si = 0
        eii = sii + nwinlen
        if ei > slen:
            eii = sii + nwinlen - (ei - slen)
            ei = slen

        #set the content of the windows
        win1[:] = 0.0
        win2[:] = 0.0
        win1[sii:eii] = s1[si:ei]
        win2[sii:eii] = s2[si:ei]
        #print '(%0.2f, %0.2f, %0.2f), s1sum=%0.0f, s2sum=%0.0f, k=%d, center=%d, si=%d, ei=%d, sii=%d, eii=%d' % \
        #      ((center-hnwinlen)/sample_rate, (center+hnwinlen+1)/sample_rate, center/sample_rate, s1sum, s2sum, k, center, si, ei, sii, eii)

        #compute the coherence
        cdata = compute_mtcoherence(win1, win2, sample_rate, window_size=window_size, bandwidth=bandwidth)
        timefreq[:, k] = cdata.coherence
        if debug:
            total_time = 0.0
            etime = time.time() - stime
            total_time += etime
            print '\twindow %d: time = %0.2fs' % (k, etime)

        #compute the noise floor
        if noise_floor:

            csum = np.zeros([len(cdata.coherence)])

            for m in range(num_noise_trials):
                if debug:
                    stime = time.time()

                #compute coherence between win1 and randomly selected slice of s2
                win2_shift_index = k
                while win2_shift_index == k or win2_shift_index < floor_window_index_min or win2_shift_index > floor_window_index_max:
                    win2_shift_index = np.random.randint(nwindows)
                w2center = win2_shift_index*nincrement
                w2si = w2center - hnwinlen
                w2ei = w2center + hnwinlen + 1
                win2_shift = s2[w2si:w2ei]
                #print 'len(s2)=%d, win2_shift_index=%d, w2si=%d, w2ei=%d, len(win1)=%d, len(win2_shift)=%d' % \
                #      (len(s2), win2_shift_index, w2si, w2ei, len(win1), len(win2_shift))
                cdata1 = compute_mtcoherence(win1, win2_shift, sample_rate, window_size=window_size, bandwidth=bandwidth)
                csum += cdata1.coherence

                #compute coherence between win2 and randomly selected slice of s1
                win1_shift_index = k
                while win1_shift_index == k or win1_shift_index < floor_window_index_min or win1_shift_index > floor_window_index_max:
                    win1_shift_index = np.random.randint(nwindows)
                w1center = win1_shift_index*nincrement
                w1si = w1center - hnwinlen
                w1ei = w1center + hnwinlen + 1
                win1_shift = s1[w1si:w1ei]
                #print 'nwindows=%d, len(s1)=%d, win1_shift_index=%d, w1si=%d, w1ei=%d, len(win2)=%d, len(win1_shift)=%d' % \
                #      (nwindows, len(s1), win1_shift_index, w1si, w1ei, len(win2), len(win1_shift))
                cdata2 = compute_mtcoherence(win2, win1_shift, sample_rate, window_size=window_size, bandwidth=bandwidth)
                csum += cdata2.coherence

                if debug:
                    etime = time.time() - stime
                    total_time += etime
                    print '\t\tnoise trial %d: time = %0.2fs' % (m, etime)

            timefreq_floor[:, k] = csum / (2*num_noise_trials)

        if debug:
            print '\tTotal time for window %d: %0.2fs' % (k, total_time)
            print '\tExpected total time for all iterations: %0.2f min' % (total_time*nwindows / 60.0)

    t = np.arange(nwindows)*increment
    if noise_floor:
        return t,freq,timefreq,timefreq_floor
    else:
        return t,freq,timefreq


def power_spectrum(s, sr, log=False, max_val=None):
    f = fft(s)
    freq = fftfreq(len(s), d=1.0/sr)
    findex = freq >= 0.0
    ps = np.abs(f)
    if log:
        if max_val is None:
            max_val = ps.max()
        ps /= max_val
        ps = 20.0*np.log10(ps)

    return freq[findex], ps[findex]


def mt_power_spectrum(s, sample_rate, window_size, low_bias=False):
    """
        Computes a jackknifed multi-taper power spectrum of a given signal. The jackknife is over
        windowed segments of the signal, specified by window_size.
    """

    sample_length_bins = min(len(s), int(window_size * sample_rate))

    #break signal into chunks and estimate coherence for each chunk
    nchunks = int(np.floor(len(s) / float(sample_length_bins)))
    nleft = len(s) % sample_length_bins
    #ignore the last chunk if it's too short
    if nleft > (sample_length_bins / 2.0):
        nchunks += 1

    ps_freq = None
    ps_ests = list()
    for k in range(nchunks):
        si = k*sample_length_bins
        ei = min(len(s), si + sample_length_bins)
        print 'si=%d, ei=%d, len(s)=%d' % (si, ei, len(s))

        ps_freq,mt_ps,var = ntalg.multi_taper_psd(s[si:ei], Fs=sample_rate, adaptive=True, jackknife=False,
                                                  low_bias=low_bias, sides='onesided')
        ps_ests.append(mt_ps)

    ps_ests = np.array(ps_ests)

    ps_mean = ps_ests.mean(axis=0)
    ps_std = ps_ests.std(axis=0, ddof=1)

    return ps_freq,ps_mean,ps_std


def match_power_spectrum(s, sample_rate, nsamps=5, isreal=False):
    """
        Create a signals that have the same power spectrum as s but with randomly shuffled phases. nsamps is the number
        of times the signal is permuted. Returns an nsamps X len(s) matrix.
    """

    #get FT of the signal
    sfft = fft(s)
    amplitude = np.abs(sfft)
    phase = np.angle(sfft)

    s_recon = np.zeros([nsamps, len(s)], dtype='complex128')
    for k in range(nsamps):
        #shuffle the phase
        np.random.shuffle(phase)
        #reconstruct the signal
        sfft_recon = amplitude*(np.cos(phase) + 1j*np.sin(phase))
        s_recon[k, :] = ifft(sfft_recon)

    if isreal:
        return np.real(s_recon)
    return s_recon


def gaussian_window(N, nstd):
    """
        Generate a Gaussian window of length N and standard deviation nstd.
    """
    hnwinlen = (N + (1-N%2)) / 2
    gauss_t = np.arange(-hnwinlen, hnwinlen+1, 1.0)
    gauss_std = float(N) / float(nstd)
    gauss_window = np.exp(-gauss_t**2 / (2.0*gauss_std**2)) / (gauss_std*np.sqrt(2*np.pi))
    return gauss_t,gauss_window
