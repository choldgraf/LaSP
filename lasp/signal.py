import numpy as np
import mne
import pandas as pd
from scipy.fftpack import fft,fftfreq,ifft
from scipy.signal import filter_design, resample,filtfilt

import matplotlib.pyplot as plt

import nitime.algorithms as ntalg
import time
from sklearn.decomposition import PCA,RandomizedPCA
from lasp.coherence import compute_mtcoherence


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


def find_extrema(s):
    """
        Find the max and mins of a signal s.
    """
    max_env = np.logical_and(
                        np.r_[True, s[1:] > s[:-1]],
                        np.r_[s[:-1] > s[1:], True])
    min_env = np.logical_and(
                        np.r_[True, s[1:] < s[:-1]],
                        np.r_[s[:-1] < s[1:], True])
    max_env[0] = max_env[-1] = False

    #exclude endpoints
    mini = [m for m in min_env.nonzero()[0] if m != 0 and m != len(s)-1]
    maxi = [m for m in max_env.nonzero()[0] if m != 0 and m != len(s)-1]

    return mini,maxi


def another_hilbert(s):
    """
        An implementation of the hilbert transform for computing the analytic signal.
    """

    #take FFT
    sfft = fft(s)
    freq = fftfreq(len(s))

    #zero out coefficients at negative frequencies
    sfft[freq < 0.0] = np.zeros(np.sum(freq < 0.0), dtype='complex')
    sfft[freq > 0.0] *= 2.0

    #take the IFFT
    z = ifft(sfft)
    return z


def compute_instantaneous_frequency(z, sample_rate):
    """
        Compute the instantaneous frequency given an analytic signal z.
    """
    x = z.real
    y = z.imag

    dx = np.r_[0.0, np.diff(x)]*sample_rate
    dy = np.r_[0.0, np.diff(y)]*sample_rate

    f = (x*dy - y*dx) / (2*np.pi*(x**2 + y**2))

    return f


def demodulate(Z, over_space=True, depth=1):
    """
        Apply demodulation (Argawal et. al 2014) to a matrix of complex-valued signals Z.

        Args:
            Z: an NxT signal matrix of N complex valued signals, each of length T
            over_space: whether to demodulate across space (does PCA on N dimensions) or time (does PCA on T dimensions)
            depth: how many PCA projection phases to subtract off

        Returns:
            phase: An NxT real-valued matrix of demodulated phases.
            pcs: An NxN complex-valued matrix of principle components.
    """

    #do complex PCA on each IMF
    N,T = Z.shape

    if over_space:

        #construct a matrix with the real and imaginary parts separated
        X = np.zeros([2*N, T], dtype='float')
        X[:N, :] = Z.real
        X[N:, :] = Z.imag

        pca = PCA()
        pca.fit(X.T)

        complex_pcs = np.zeros([N, N], dtype='complex')
        for j in range(N):
            pc = pca.components_[j, :]
            complex_pcs[j, :].real = pc[:N]
            complex_pcs[j, :].imag = pc[N:]

        phase = np.angle(Z)
        for k in range(depth):
            #compute the kth PC projected component
            proj = np.dot(Z.T.squeeze(), complex_pcs[k, :].squeeze())
            phase -= np.angle(proj)

    else:

        first_pc = np.zeros([T], dtype='complex')

        pca_real = RandomizedPCA(n_components=1)
        pca_real.fit(Z.real)
        print 'pca_real.components_.shape=',pca_real.components_.shape
        first_pc.real = pca_real.components_.squeeze()
        
        pca_imag = RandomizedPCA(n_components=1)
        pca_imag.fit(Z.imag)
        print 'pca_imag.components_.shape=',pca_imag.components_.shape
        first_pc.imag = pca_imag.components_.squeeze()

        complex_pcs = np.array([first_pc])

        proj = first_pc

        #demodulate the signal
        phase = np.angle(Z) - np.angle(proj)

    return phase,complex_pcs


def compute_coherence_over_time(signal, trials, Fs, n_perm=5, low=0, high=300):
    '''
    Computes the coherence between the mean of subsets of trails. This can be used
    to assess signal stability in response to a stimulus (repeated or otherwise).

    INPUTS
    --------
    signal : array-like
        The array of neural signals. Should be time x signals

    trials : pd.DataFrame, contains columns 'epoch', and 'time'
             and same first dimension as signal
        A dataframe with time indices and trial number within each epoch (trial)
        This is used to pull out the corresponding timepoints from signal.

    Fs : int
        The sampling rate of the signal

    OUTPUTS
    --------
    coh_perm : np.array, shape (n_perms, n_signals, n_freqs)
        A collection of coherence values for each permutation.

    coh_freqs : np.array, shape (n_freqs)
        The frequency values corresponding to the final dimension of coh_perm
    Output is permutations x signals x frequency bands
    '''
    trials = pd.DataFrame(trials)
    assert ('epoch' in trials.columns and 'time' in trials.columns), 'trials must be a DataFrame with "epoch" column'
    n_trials = np.max(trials['epoch'])

    coh_perm = []
    for perm in xrange(n_perm):
        trial_ixs = np.random.permutation(np.arange(n_trials))
        t1 = trial_ixs[:n_trials/2]
        t2 = trial_ixs[n_trials/2:]
        
        # Split up trials and take the mean of each
        mn1, mn2 = [signal[trials.eval('epoch in @t_ix and time > 0').values].mean(level=('time'))
                    for t_ix in [t1, t2]]

        # Now compute coherence between the two
        coh_all_freqs = []
        for (elec, vals1), (_, vals2) in zip(mn1.iteritems(), mn2.iteritems()):
            ts_arr = np.vstack([vals1, vals2])
            coh, coh_freqs, coh_times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(ts_arr[None, :, :], sfreq=Fs,
                                                                                                   fmin=low, fmax=high, verbose=0)
            coh_all_freqs.append(coh[1, 0, :])
        coh_perm.append(coh_all_freqs)
    coh_perm = np.array(coh_perm)
    return coh_perm, coh_freqs


def segment_envelope(s, threshold_percentile=10):
    """ Segments a one dimensional positive-valued time series into events with start and end times.

    :param s: The signal, a numpy array.
    :return: A list of event start and end times, and the maximum amplitude encountered in that event.
    """

    assert np.sum(s < 0) == 0, "segment_envelope: Can't segment a signal that has negative values!"

    #determine threshold to be the 10th percentile
    thresh = np.percentile(s[s > 0], threshold_percentile)
    print 'thresh=%f' % thresh

    #array to keep track of start and end times of each event
    events = list()

    #scan through the signal, find events
    in_event = False
    max_amp = -np.inf
    start_index = -1
    for t,x in enumerate(s):

        if in_event:
            if x > max_amp:
                #we found a new peak
                max_amp = x
            if x < thresh:
                #the event has ended
                in_event = False
                events.append( (start_index, t, max_amp))
                #print 'Identified event (%d, %d, %0.6f)' % (start_index, t, max_amp)
        else:
            if x > thresh:
                in_event = True
                start_index = t
                max_amp = thresh

    print '# of events: %d' % len(events)
    #TODO: merge small adjacent events, break down long events

    return np.array(events)
