import numpy as np
from scipy import fftpack

import nitime.algorithms as ntalg
from nitime import utils as ntutils

from tools.spikes import compute_psth


class CoherenceData(object):
    """
        This class encapsulates a bunch of data for coherence calculations:

        .frequency: an array of frequencies that the coherence is computed for
        .coherence: the coherence computed, same dimensionality as frequency
        .coherence_upper: the upper bound of the coherence
        .coherence_lower: the lower bound of the coherence
        .sample_rate: the sample rate in Hz of the original signal
        .frequency_cutoff: the frequency at which the lower bound of the coherence goes below zero
        .nmi: the normal mutual information in bits/s
    """

    def __init__(self, frequency_cutoff=None):
        self.frequency = None
        self.coherence = None
        self.coherence_upper = None
        self.coherence_lower = None
        self.sample_rate = None

        self.frequency_cutoff = frequency_cutoff
        self._nmi = None

    @property
    def nmi(self):
        if self._nmi is None and self.coherence is not None and self.frequency is not None and self.coherence_lower is not None:
            nfreq_cutoff,nnminfo = compute_freq_cutoff_and_nmi(self.frequency, self.sample_rate, self.coherence, self.coherence_lower, freq_cutoff=self.frequency_cutoff)
            self.frequency_cutoff = nfreq_cutoff
            self._nmi = nnminfo
        return self._nmi


def merge_spike_trials(trials_by_stim, stim_durations):
    """ Helper function - if you want to analyze  coherence across multiple stimuli, with
        multiple trials per stimulus, construct a list of lists called trials_by_stim, where:

            trials_by_stim[i] = [ [t1, t2, t3, t4, ...],
                                  [t1, t2, ..],
                                  [t1, t2, t3, t4, t5, ...],
                                  ...
                                ]
            i.e. each element of trials_by_stim is comprised a list of trials, each trial is a list of spike times.

            stim_durations is a list of stimulus durations in seconds, where len(stim_durations) == len(trials_by_stim)
    """

    assert len(trials_by_stim) == len(stim_durations)
    ntrials_all = np.array([len(spike_trials) for spike_trials in trials_by_stim])
    ntrials = ntrials_all.max()

    merged_spike_trials = list()
    for k in range(ntrials):
        merged_spike_trials.append(list())

    for k,spike_trials in enumerate(trials_by_stim):

        sd = 0.0
        if k > 0:
            sd = stim_durations[k-1]

        for m,spike_train in enumerate(spike_trials):
            merged_spike_trials[m].extend(np.array(spike_train) + sd)

    return merged_spike_trials


class ModelCoherenceAnalyzer(object):
    """
        You can use this object to compute the coherence upper bound for a set of spike trains,
        along with predictions for the PSTHs.
    """

    def __init__(self, spike_trials_by_stim, preds_by_stim, window_size=3.0, bandwidth=15.0, bin_size=0.001,
                 frequency_cutoff=None, tanh_transform=False):
        """
            spike_trials_by_stim: an array of arrays of arrays, the length is equal to the number of
                stimuli, and element i looks like:
                spike_trials_by_stim[i] = [ [t1, t2, t3, t4, ...],
                                            [t1, t2, ..],
                                            [t1, t2, t3, t4, t5, ...],
                                            ...
                                          ]

            preds_by_stim: a predicted PSTH for each stimulus, so len(preds_by_stim)=len(spike_trials_by_stim).

            window_size: the size in seconds of the window used to compute Fourier Transforms. The signal is
                broken up into segments of length window_size prior to computing the multi-tapered spectral density.

            bandwidth: a quantity related to the number of tapers used in the spectral density estimates

            bin_size: the size in seconds of the bins used to compute the PSTH

            frequency_cutoff: frequency in Hz to stop for computing the normal mutual information

            tanh_transform: whether to transform the coherence prior to computing the jacknifed upper bounds
        """

        if len(spike_trials_by_stim) != len(preds_by_stim):
            print '# of stims for spike trials should equal # of predictions'
            return

        psth_lens = np.array([len(pred) for pred in preds_by_stim])

        self.spike_trails_by_stim = spike_trials_by_stim
        self.window_size = window_size
        self.bin_size = bin_size
        self.bandwidth = bandwidth
        self.cdata_bound,self.ncdata_bound = compute_single_spike_coherence_bound(self.spike_trails_by_stim, self.bandwidth,
                                                                                  self.window_size, psth_lens, bin_size=self.bin_size,
                                                                                  frequency_cutoff=frequency_cutoff,
                                                                                  tanh_transform=tanh_transform)

        concat_preds = np.zeros([psth_lens.sum()])
        offset = 0
        for k,pred in enumerate(preds_by_stim):
            e = offset + psth_lens[k]
            concat_preds[offset:e] = pred
            offset = e

        self.model_data = compute_coherence_model_performance(self.spike_trails_by_stim, concat_preds, self.bandwidth,
                                                              self.window_size, psth_lens, bin_size=self.bin_size,
                                                              cdata_bound=self.cdata_bound, frequency_cutoff=frequency_cutoff,
                                                              tanh_transform=tanh_transform)


def get_concat_split_psths(spike_trials_by_stim, psth_lens, bin_size):
    """
        Takes an array of arrays of spike times, splits each into even and
        odd trials, and computes the PSTH for the even and odd trials.
    """

    N = psth_lens.sum()

    concat_even_psths = np.zeros([N])
    concat_odd_psths = np.zeros([N])

    offset = 0
    for m,spike_trials in enumerate(spike_trials_by_stim):
        even_trials = [ti for k,ti in enumerate(spike_trials) if k % 2]
        odd_trials = [ti for k,ti in enumerate(spike_trials) if not k % 2]
        duration = psth_lens[m] * bin_size
        even_psth = compute_psth(even_trials, duration, bin_size=bin_size)
        odd_psth = compute_psth(odd_trials, duration, bin_size=bin_size)

        e = offset + psth_lens[m]
        concat_even_psths[offset:e] = even_psth
        concat_odd_psths[offset:e] = odd_psth
        offset = e
    return concat_even_psths,concat_odd_psths


def get_concat_psth(spike_trials_by_stim, psth_lens, bin_size):
    """
        Takes a bunch of spike trials, separated by stimulus, creates a PSTH per stimulus,
        and concatenates each PSTH into a long array.
    """

    N = np.sum(psth_lens)

    concat_psths = np.zeros([N])

    offset = 0
    for k,spike_trials in enumerate(spike_trials_by_stim):
        duration = psth_lens[k] * bin_size
        psth = compute_psth(spike_trials, duration, bin_size=bin_size)
        e = offset + psth_lens[k]
        concat_psths[offset:e] = psth
        offset = e
    return concat_psths


def compute_single_spike_coherence_bound(spike_trials_by_stim, bandwidth, window_size, psth_lens, bin_size=0.001,
                                         frequency_cutoff=None, tanh_transform=False):
    """
        Computes the coherence between a set of spike trains and themselves. Useful for producing
        an upper bound on possible coherence in a model-independent way.

        spike_trials_by_stim: an array of arrays of arrays, the length is equal to the number of
                stimuli, and element i looks like:
                spike_trials_by_stim[i] = [ [t1, t2, t3, t4, ...],
                                            [t1, t2, ..],
                                            [t1, t2, t3, t4, t5, ...],
                                            ...
                                          ]

        preds_by_stim: a predicted PSTH for each stimulus, so len(preds_by_stim)=len(spike_trials_by_stim).

        psth_lens: The length in seconds of each stimulus. len(psth_lens) = len(spike_trials_by_stim)

        window_size: the size in seconds of the window used to compute Fourier Transforms. The signal is
            broken up into segments of length window_size prior to computing the multi-tapered spectral density.

        bandwidth: a quantity related to the number of tapers used in the spectral density estimates

        bin_size: the size in seconds of the bins used to compute the PSTH

        frequency_cutoff: frequency in Hz to stop for computing the normal mutual information

        tanh_transform: whether to transform the coherence prior to computing the jacknifed upper bounds

    """

    all_ntrials = np.array([len(spike_trials) for spike_trials in spike_trials_by_stim])
    ntrials = all_ntrials.max()
    even_psth,odd_psth = get_concat_split_psths(spike_trials_by_stim, psth_lens, bin_size)

    def cnormalize(c, num_trials):
        sign = np.sign(c)
        cabs = np.abs(c)
        index = cabs > 0.0
        kdown = (-num_trials + num_trials * np.sqrt(1.0 / cabs[index])) / 2.0
        kdown *= sign[index]
        cnorm = np.zeros(c.shape)
        cnorm[index] = 1.0 / (kdown + 1.0)
        return cnorm

    sample_rate = 1.0 / bin_size
    cdata = compute_coherence(even_psth, odd_psth, sample_rate, window_size, bandwidth=bandwidth,
                              frequency_cutoff=frequency_cutoff, tanh_transform=tanh_transform)

    ncoherence_mean = cnormalize(cdata.coherence, ntrials)
    ncoherence_upper = cnormalize(cdata.coherence_upper, ntrials)
    ncoherence_lower = cnormalize(cdata.coherence_lower, ntrials)

    ncdata = CoherenceData(frequency_cutoff=frequency_cutoff)
    ncdata.frequency = cdata.frequency
    ncdata.sample_rate = sample_rate
    ncdata.coherence = ncoherence_mean
    ncdata.coherence_lower = ncoherence_lower
    ncdata.coherence_upper = ncoherence_upper

    return cdata,ncdata


def compute_coherence_model_performance(spike_trials_by_stim, psth_prediction, bandwidth, window_size, psth_lens, bin_size=0.001,
                                        cdata_bound=None, frequency_cutoff=None, tanh_transform=False):
    """
        Computes coherence of the spike trains themselves, but also the model. Use the ModelCoherenceAnalyzer class
        instead of calling this function directly.
    """

    all_ntrials = np.array([len(spike_trials) for spike_trials in spike_trials_by_stim])
    ntrials = all_ntrials.max()

    #compute upper bound for model performance from real data
    if cdata_bound is None:
        cdata_bound,ncdata_bound = compute_single_spike_coherence_bound(spike_trials_by_stim, bandwidth, window_size,
                                                                        psth_lens, bin_size=bin_size, tanh_transform=tanh_transform)

    psth = get_concat_psth(spike_trials_by_stim, psth_lens, bin_size=bin_size)

    sample_rate = 1.0 / bin_size

    #compute the non-normalized coherence between the real PSTH and the model prediction of the PSTH
    cdata_model = compute_coherence(psth, psth_prediction, sample_rate, window_size, bandwidth=bandwidth, frequency_cutoff=frequency_cutoff)

    def cnormalize(cbound, cpred, num_trials):
        sign = np.sign(cbound)
        cbound_abs = np.abs(cbound)
        index = np.abs(cbound) > 0.0
        rhs = (1.0 + np.sqrt(1.0 / cbound_abs[index])) / (-num_trials + num_trials * np.sqrt(1.0 / cbound_abs[index]) + 2.0)
        rhs *= sign[index]
        ncpred = np.zeros(cpred.shape)
        ncpred[index] = cpred[index] * rhs
        return ncpred

    #correct each model coherence
    ncmean_model = cnormalize(cdata_bound.coherence, cdata_model.coherence, ntrials)
    ncupper_model = cnormalize(cdata_bound.coherence_upper, cdata_model.coherence_upper, ntrials)
    nclower_model = cnormalize(cdata_bound.coherence_lower, cdata_model.coherence_lower, ntrials)

    mcdata = CoherenceData(frequency_cutoff=frequency_cutoff)
    mcdata.frequency = cdata_model.frequency
    mcdata.sample_rate = sample_rate
    mcdata.coherence = ncmean_model
    mcdata.coherence_lower = nclower_model
    mcdata.coherence_upper = ncupper_model

    return mcdata


def compute_coherence_original(s1, s2, sample_rate, bandwidth, jackknife=False, tanh_transform=False):
    """
        An implementation of computing the coherence. Don't use this.
    """

    minlen = min(len(s1), len(s2))
    if s1.shape != s2.shape:
        s1 = s1[:minlen]
        s2 = s2[:minlen]

    window_length = len(s1) / sample_rate
    window_length_bins = int(window_length * sample_rate)

    #compute DPSS tapers for signals
    NW = int(window_length*bandwidth)
    K = 2*NW - 1
    print 'compute_coherence: NW=%d, K=%d' % (NW, K)
    tapers,eigs = ntalg.dpss_windows(window_length_bins, NW, K)

    njn = len(eigs)
    jn_indices = [range(njn)]
    #compute jackknife indices
    if jackknife:
        jn_indices = list()
        for i in range(len(eigs)):
            jn = range(len(eigs))
            jn.remove(i)
            jn_indices.append(jn)

    #taper the signals
    s1_tap = tapers * s1
    s2_tap = tapers * s2

    #compute fft of tapered signals
    s1_fft = fftpack.fft(s1_tap, axis=1)
    s2_fft = fftpack.fft(s2_tap, axis=1)

    #compute adaptive weights for each taper
    w1,nu1 = ntutils.adaptive_weights(s1_fft, eigs, sides='onesided')
    w2,nu2 = ntutils.adaptive_weights(s2_fft, eigs, sides='onesided')

    coherence_estimates = list()
    for jn in jn_indices:

        #compute cross spectral density
        sxy = ntalg.mtm_cross_spectrum(s1_fft[jn, :], s2_fft[jn, :], (w1[jn], w2[jn]), sides='onesided')

        #compute individual power spectrums
        sxx = ntalg.mtm_cross_spectrum(s1_fft[jn, :], s1_fft[jn, :], w1[jn], sides='onesided')
        syy = ntalg.mtm_cross_spectrum(s2_fft[jn, :], s2_fft[jn, :], w2[jn], sides='onesided')

        #compute coherence
        coherence = np.abs(sxy)**2 / (sxx * syy)
        coherence_estimates.append(coherence)

    #compute variance
    coherence_estimates = np.array(coherence_estimates)
    coherence_variance = np.zeros([coherence_estimates.shape[1]])
    coherence_mean = coherence_estimates[0]
    if jackknife:
        coherence_mean = coherence_estimates.mean(axis=0)
        #mean subtract and square
        cv = np.sum((coherence_estimates - coherence_mean)**2, axis=0)
        coherence_variance[:] = (1.0 - 1.0/njn) * cv

    #compute frequencies
    sampint = 1.0 / sample_rate
    L = minlen / 2 + 1
    freq = np.linspace(0, 1 / (2 * sampint), L)

    #compute upper and lower bounds
    cmean = coherence_mean
    coherence_lower = cmean - 2*np.sqrt(coherence_variance)
    coherence_upper = cmean + 2*np.sqrt(coherence_variance)

    cdata = CoherenceData()
    cdata.coherence = coherence_mean
    cdata.coherence_lower = coherence_lower
    cdata.coherence_upper = coherence_upper
    cdata.frequency = freq
    cdata.sample_rate = sample_rate

    return cdata

def compute_coherence(s1, s2, sample_rate, window_size, bandwidth=15.0, chunk_len_percentage_tolerance=0.30, frequency_cutoff=None, tanh_transform=False):
    """
        Computing the coherence between signals s1 and s2. To do so, the signals are broken up into segments of length
        specified by window_size. Then the multi-taper coherence is computed between each segment. The mean coherence
        is computed across segments, and an estimate of the coherence variance is computed across segments.

        sample_rate: the sample rate in Hz of s1 and s2

        window_size: size of the segments in seconds

        bandwidth: related to the # of tapers used to compute the spectral density. The higher the bandwidth, the more tapers.

        chunk_len_percentage_tolerance: If there are leftover segments whose lengths are less than window_size, use them
            if they comprise at least the fraction of window_size specified by chunk_len_percentage_tolerance

        frequency_cutoff: the frequency at which to cut off the coherence when computing the normal mutual information

        tanh_transform: whether to transform the coherences when computing the upper and lower bounds, supposedly
            improves the estimate of variance.
    """

    minlen = min(len(s1), len(s2))
    if s1.shape != s2.shape:
        s1 = s1[:minlen]
        s2 = s2[:minlen]

    sample_length_bins = min(len(s1), int(window_size * sample_rate))

    #compute DPSS tapers for signals
    NW = int(window_size*bandwidth)
    K = 2*NW - 1
    #print 'compute_coherence: NW=%d, K=%d' % (NW, K)
    tapers,eigs = ntalg.dpss_windows(sample_length_bins, NW, K)

    #break signal into chunks and estimate coherence for each chunk
    nchunks = int(np.floor(len(s1) / float(sample_length_bins)))
    nleft = len(s1) % sample_length_bins
    if nleft > 0:
        nchunks += 1
    #print 'sample_length_bins=%d, # of chunks:%d, # samples in last chunk: %d' % (sample_length_bins, nchunks, nleft)
    coherence_estimates = list()
    for k in range(nchunks):
        s = k*sample_length_bins
        e = min(len(s1), s + sample_length_bins)
        chunk_len = e - s
        chunk_percentage = chunk_len / float(sample_length_bins)
        if chunk_percentage < chunk_len_percentage_tolerance:
            #don't compute coherence for a chunk whose length is less than a certain percentage of sample_length_bins
            continue
        s1_chunk = np.zeros([sample_length_bins])
        s2_chunk = np.zeros([sample_length_bins])
        s1_chunk[:chunk_len] = s1[s:e]
        s2_chunk[:chunk_len] = s2[s:e]

        #taper the signals
        s1_tap = tapers * s1_chunk
        s2_tap = tapers * s2_chunk

        #compute fft of tapered signals
        s1_fft = fftpack.fft(s1_tap, axis=1)
        s2_fft = fftpack.fft(s2_tap, axis=1)

        #compute adaptive weights for each taper
        w1,nu1 = ntutils.adaptive_weights(s1_fft, eigs, sides='onesided')
        w2,nu2 = ntutils.adaptive_weights(s2_fft, eigs, sides='onesided')

        #compute cross spectral density
        sxy = ntalg.mtm_cross_spectrum(s1_fft, s2_fft, (w1, w2), sides='onesided')

        #compute individual power spectrums
        sxx = ntalg.mtm_cross_spectrum(s1_fft, s1_fft, w1, sides='onesided')
        syy = ntalg.mtm_cross_spectrum(s2_fft, s2_fft, w2, sides='onesided')

        #compute coherence
        coherence = np.abs(sxy)**2 / (sxx * syy)
        coherence_estimates.append(coherence)

    #compute variance
    coherence_estimates = np.array(coherence_estimates)

    if tanh_transform:
        coherence_estimates = np.arctanh(coherence_estimates)

    coherence_variance = np.zeros([coherence_estimates.shape[1]])
    coherence_mean = coherence_estimates.mean(axis=0)
    #mean subtract and square
    cv = np.sum((coherence_estimates - coherence_mean)**2, axis=0)
    coherence_variance[:] = (1.0 - 1.0/nchunks) * cv

    if tanh_transform:
        coherence_variance = np.tanh(coherence_variance)
        coherence_mean = np.tanh(coherence_mean)

    #compute frequencies
    sampint = 1.0 / sample_rate
    L = sample_length_bins / 2 + 1
    freq = np.linspace(0, 1 / (2 * sampint), L)

    #compute upper and lower bounds
    coherence_lower = coherence_mean - 2*np.sqrt(coherence_variance)
    coherence_upper = coherence_mean + 2*np.sqrt(coherence_variance)

    cdata = CoherenceData(frequency_cutoff=frequency_cutoff)
    cdata.coherence = coherence_mean
    cdata.coherence_lower = coherence_lower
    cdata.coherence_upper = coherence_upper
    cdata.frequency = freq
    cdata.sample_rate = sample_rate

    return cdata


def compute_freq_cutoff_and_nmi(freq, sample_rate, coherence_mean, coherence_lower, freq_cutoff=None):
    """
        Given the coherence and lower bound on coherence, compute the frequency cutoff, which is
        the point at which the lower bound dips below zero.
    """

    if freq_cutoff is None:
        #find frequency at which lower bound dips below zero
        zindices = np.where(coherence_lower <= 0.0)[0]
        freq_cutoff_index = len(coherence_mean)
        if len(zindices) > 0:
            freq_cutoff_index = min(zindices)
        else:
            zindices = np.where(freq < (sample_rate / 2.0))[0]
            if len(zindices) > 0:
                freq_cutoff_index = max(zindices)
        freq_cutoff = freq[freq_cutoff_index]
    else:
        freq_cutoff_index = max(np.where(freq <= freq_cutoff)[0])
        print freq_cutoff_index

    #compute normalized mutual information
    df = freq[1] - freq[0]
    nminfo = -df * np.log2(1.0 - coherence_mean[:freq_cutoff_index]).sum()

    return freq_cutoff,nminfo
