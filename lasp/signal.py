import numpy as np
import mne
import pandas as pd
from scipy.fftpack import fft,fftfreq,ifft
from scipy.ndimage import convolve1d
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


def power_spectrum(s, sr, log=False, max_val=None, hanning=False):

    sw = s
    if hanning:
        sw = s*np.hanning(len(s))

    f = fft(sw)
    freq = fftfreq(len(sw), d=1.0/sr)
    findex = freq >= 0.0
    ps = np.abs(f)**2
    if log:
        if max_val is None:
            max_val = ps.max()
        ps /= max_val
        ps = 20.0*np.log10(ps)

    return freq[findex], ps[findex]


def mt_power_spectrum(s, sample_rate, window_size, low_bias=False, bandwidth=5.0):
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

        ps_freq,mt_ps,var = ntalg.multi_taper_psd(s[si:ei], Fs=sample_rate, adaptive=True, BW=bandwidth, jackknife=False,
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


def analytic_signal(s):
    """
        An implementation of computing the analytic signal.
    """

    #take FFT
    sfft = fft(s)
    freq = fftfreq(len(s))

    #zero out coefficients at negative frequencies
    sfft[freq < 0.0] = np.zeros(np.sum(freq < 0.0), dtype='complex')
    sfft[freq >= 0.0] *= 2.0

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


def break_envelope_into_events(s, threshold=0, merge_thresh=None):
    """ Segments a one dimensional positive-valued time series into events with start and end times.

    :param s: The signal, a numpy array.
    :param threshold: The threshold for determining the onset of an event. When the amplitude of s
            exceeds threshold, an event starts, and when the amplitude of the signal falls below
            threshold, the event ends.
    :param merge_thresh: Events that are separated by less than minimum_len get merged together. minimum_len
            must be specified in number of time points, not actual time.
    :return: A list of event start and end times, and the maximum amplitude encountered in that event.
    """

    assert np.sum(s < 0) == 0, "segment_envelope: Can't segment a signal that has negative values!"

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
            if x <= threshold:
                #the event has ended
                in_event = False
                events.append( (start_index, t, max_amp))
                #print 'Identified event (%d, %d, %0.6f)' % (start_index, t, max_amp)
        else:
            if x > threshold:
                in_event = True
                start_index = t
                max_amp = threshold

    # print '# of events (pre-merge): %d' % len(events)
    events = np.array(events)

    if merge_thresh is None:
        return events

    #compute the inter-event interval, used for merging smaller events into larger ones
    iei = events[1:, 0] - events[:-1, 1]

    #create an empty list for merged events
    merged_events = list()

    #set the "current event" to be the first event
    estart, eend, eamp = events[0, :]

    for k in range(len(events)-1):

        #get the event at time k+1
        stime,etime,amp = events[k+1, :]

        #get the inter-event-interval between the event at time k+1 and k
        the_iei = iei[k]
        #print 'k=%d, the_iei=%d, merge_thresh=%d' % (k, the_iei, merge_thresh)

        if the_iei < merge_thresh:
            #extend the end time of the current event
            eend = etime
            #change the maximum peak of the current event
            eamp = max(eamp, amp)
        else:
            #don't merge, pop the previous event
            merged_events.append( (estart, eend, eamp))

            #set the currente event to be the event at k+1
            estart = stime
            eend = etime
            eamp = amp

    #pop the last event
    merged_events.append( (estart, eend, eamp))

    # print '# of merged events: %d' % len(merged_events)

    return np.array(merged_events)


def power_amplifier(s, thresh, pwr=2):
    """ Amplify elements of a positive-valued signal. Rescale the signal
        so that elements above thresh are equal to or greater than 1,
        and elements below thresh are less than one. Then take a power
        of the signal, which will supress values less than 1, and amplify
        values that are greater than one.
    """

    #normalize the signal
    s /= s.max()

    #shift the signal so elements at the threshold are set to 1
    s += 1.0 - thresh

    #raise the envelope to a power, amplifying values that are above 1
    s = s**pwr

    #re-normalize
    s -= (1.0 - thresh)**pwr
    s /= s.max()

    return s


def phase_locking_value(z1, z2):
    """ Compute the phase-locking-value (PLV) between two complex signals. """

    assert len(z1) == len(z2), "Signals must be same length! len(z1)=%d, len(z2)=%d" % (len(z1), len(z2))
    N = len(z1)
    theta = np.angle(z2) - np.angle(z1)

    p = np.exp(complex(0, 1)*theta)
    plv = np.abs(p.sum()) / N

    return plv


def correlation_function(s1, s2, lags):
    
    assert len(s1) == len(s2), "Signals must be same length! len(s1)=%d, len(s2)=%d" % (len(s1), len(s2))

    s1_mean = s1.mean()
    s2_mean = s2.mean()
    s1_std = s1.std(ddof=1)
    s2_std = s2.std(ddof=1)
    s1_centered = s1 - s1_mean
    s2_centered = s2 - s2_mean
    N = len(s1)
    
    cf = np.zeros([len(lags)])
    for k,lag in enumerate(lags):

        if lag == 0:
            cf[k] = np.dot(s1_centered, s2_centered) / (N-lag)
        elif lag > 0:
            cf[k] = np.dot(s1_centered[:-lag], s2_centered[lag:]) / (N-lag)
        elif lag < 0:
            cf[k] = np.dot(s1_centered[np.abs(lag):], s2_centered[:lag]) / (N-lag)

    cf /= s1_std * s2_std

    return cf


def get_envelope_end(env):
    """ Given an amplitude envelope, get the index that indicates the derivative of the envelope
        has converged to zero, indicating an end point.
    """
    denv = np.diff(env)
    i = np.where(np.abs(denv) > 0)[0]
    true_stop_index = np.max(i)+1
    return true_stop_index


def simple_smooth(s, window_len):

    w = np.hanning(window_len)
    w /= w.sum()
    return convolve1d(s, w)


def quantify_cf(lags, cf, plot=False):
    """ Quantify properties of an auto or cross correlation function. """

    # identify the peak magnitude
    abs_cf = np.abs(cf)
    peak_magnitude = abs_cf.max()

    # identify the peak delay
    imax = abs_cf.argmax()
    peak_delay = lags[imax]

    # compute the area under the curve
    dt = np.diff(lags).max()
    cf_width = abs_cf.sum()*dt

    # compute the skewdness
    p = abs_cf / abs_cf.sum()
    mean = np.sum(lags*p)
    std = np.sqrt(np.sum(p*(abs_cf - mean)**2))
    skew = np.sum(p*(abs_cf - mean)**3) / std**3

    # compute the left and right areas under the curve
    max_width = abs_cf[lags != 0].sum()*dt
    right_width = abs_cf[lags > 0].sum()*dt
    left_width = abs_cf[lags < 0].sum()*dt

    # create a measure of anisotropy from the AUCs
    anisotropy = (right_width - left_width) / max_width

    if plot:
        plt.figure()
        plt.axhline(0, c='k')
        plt.plot(lags, cf, 'r-', linewidth=3)
        plt.axvline(peak_delay, c='g', alpha=0.75)
        plt.ylim(-1, 1)
        plt.axis('tight')
        t = 'width=%0.1f, mean=%0.1f, std=%0.1f, skew=%0.1f, anisotropy=%0.2f' % (cf_width, mean, std, skew, anisotropy)
        plt.title(t)
        plt.show()

    return {'magnitude':peak_magnitude, 'delay':peak_delay, 'width':cf_width,
            'mean':mean, 'std':std, 'skew':skew, 'anisotropy':anisotropy}

