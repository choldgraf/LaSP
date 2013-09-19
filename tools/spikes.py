from scipy.stats import gamma
from matplotlib.patches import Rectangle

import numpy as np

import matplotlib.pyplot as plt


def compute_joint_isi(spike_train1, spike_train2, window_size=0.500, bin_size=0.001):

    half_window_size = window_size / 2.0
    half_nbins = int(half_window_size / bin_size)

    nbins = half_nbins*2 + 1 # ensure an odd number of bins, with zero lag in the middle

    #construct sparse matrix of spike-to-spike distances
    isi_hist = np.zeros([nbins], dtype='int')

    lowest_j = 0
    for i,ti in enumerate(spike_train1):
        #print 'i=%d, ti=%0.3f, lowest_j=%d' % (i, ti, lowest_j)
        if lowest_j > len(spike_train2)-1:
            break
        for j in range(lowest_j, len(spike_train2)):
            tj = spike_train2[j]
            dij = ti - tj
            #print '\tj=%d, tj=%0.3f, dij=%0.3f' % (j, tj, dij)

            if dij > half_window_size:
                #there is no t{i+k}, k>0 such that t{i+k} - tj < half_window_size, so this is the lowest that
                # j has to be for future iterations. we'll keep track of that to reduce the number of iterations
                # of the inner loop for future outer loop iterations
                lowest_j = j+1
                continue

            if dij < -half_window_size:
                #at this point there is no tj such that ti - tj >= -half_window_size, so we should break
                break

            else:
                #add to the histogram
                bin_index = int(np.round(dij / bin_size)) + half_nbins
                #print '\t  added to bin, bin_index=%d' % bin_index
                isi_hist[bin_index] += 1

    sp = window_size / nbins
    isi_vals = np.arange(-half_window_size, half_window_size, sp)   #values of left hand edges of bins
    return isi_vals,isi_hist


def simulate_poisson(psth, duration, num_trials=20):

    dt = 0.001
    trange = np.arange(0.0, duration, dt)
    new_spike_trials = []
    for k in range(num_trials):
        next_spike_time = np.random.exponential(1.0)
        last_spike_index = 0
        spike_times = []
        for k, t in enumerate(trange):
            csum = np.cumsum(psth[last_spike_index:k])
            if len(csum) < 1:
                continue
            if csum[-1] >= next_spike_time:
                last_spike_index = k
                spike_times.append(t)
        new_spike_trials.append(spike_times)
    return new_spike_trials


def simulate_gamma(psth, trials, duration, num_trials=20):

    #rescale the ISIs
    dt = 0.001
    rs_isis = []
    for trial in trials:
        if len(trial) < 1:
            continue
        csum = np.cumsum(psth)*dt
        for k,ti in enumerate(trial[1:]):
            tj = trial[k]
            if ti > duration or tj > duration or ti < 0.0 or tj < 0.0:
                continue
            ti_index = int((ti / duration) * len(psth))
            tj_index = int((tj / duration) * len(psth))
            #print 'k=%d, ti=%0.6f, tj=%0.6f, duration=%0.3f' % (k, ti, tj, duration)
            #print '  ti_index=%d, tj_index=%d, len(psth)=%d, len(csum)=%d' % (ti_index, tj_index, len(psth), len(csum))
            #get rescaled time as difference in cumulative intensity
            ui = csum[ti_index] - csum[tj_index]
            if ui < 0.0:
                print 'ui < 0! ui=%0.6f, csum[ti]=%0.6f, csum[tj]=%0.6f' % (ui, csum[ti_index], csum[tj_index])
            else:
                rs_isis.append(ui)
    rs_isis = np.array(rs_isis)
    rs_isi_x = np.arange(rs_isis.min(), rs_isis.max(), 1e-5)

    #fit a gamma distribution to the rescaled ISIs
    gamma_alpha,gamma_loc,gamma_beta = gamma.fit(rs_isis)
    gamma_pdf = gamma.pdf(rs_isi_x, gamma_alpha, loc=gamma_loc, scale=gamma_beta)
    print 'Rescaled ISI Gamma Fit Params: alpha=%0.3f, beta=%0.3f, loc=%0.3f' % (gamma_alpha, gamma_beta, gamma_loc)

    #simulate new trials using rescaled ISIs
    new_trials = []
    for nt in range(num_trials):
        ntrial = []
        next_rs_time = gamma.rvs(gamma_alpha, loc=gamma_loc,scale=gamma_beta)
        csum = 0.0
        for t_index,pval in enumerate(psth):
            csum += pval*dt
            if csum >= next_rs_time:
                #spike!
                t = t_index*dt
                ntrial.append(t)
                #reset integral and generate new rescaled ISI
                csum = 0.0
                next_rs_time = gamma.rvs(gamma_alpha, loc=gamma_loc,scale=gamma_beta)
        new_trials.append(ntrial)
    #plt.figure()
    #plt.hist(rs_isis, bins=20, normed=True)
    #plt.plot(rs_isi_x, gamma_pdf, 'r-')
    #plt.title('Rescaled ISIs')

    return new_trials


def compute_psth(trials, duration, bin_size=0.001):
    """
        Compute a peri-stimulus time histogram (PSTH), conditioned on an event such as stimulus.

        trials: an array of arrays of spike times in seconds, relative to the onset of the stimulus,
                If a spike precedes a stimulus, it's spike time should be negative. len(trials) = # of trials,
                and len(trials[0]) = number of spikes in first trial
        duration: the duration of the event.
        bin_size: the size in seconds of the bin to use in creating the PSTH (defaults to 0.001s = 1ms)

        Returns the average spike rate in KHz across trials in each time bin.
    """

    nbins = int(np.ceil(duration / bin_size))
    spike_counts = np.zeros(nbins)
    for stimes in trials:
        if len(stimes) == 0:
            continue
        stimes = np.array(stimes)
        if len(stimes.shape) > 0:
            #get index of spike times valid for the conditioned event
            vi = (stimes >= 0.0) & (stimes <= duration)

            #convert spike times to indices in PSTH
            sbins = np.floor(stimes[vi] / bin_size).astype('int')

            #add spike to each bin
            for k in sbins:
                spike_counts[k] += 1

    #compute rate in KHz by dividing by bin size
    spike_counts /= bin_size*1000.0

    #take mean across trials (spikes were already summed across trials)
    spike_counts /= len(trials)

    #construct time axis, represents time point at left hand of bin
    t = np.arange(nbins).astype('float') * bin_size

    return t,spike_counts


def create_random_psth(duration, smooth_win_size=10, samp_rate=1000.0, thresh=0.5):
    nsamps = duration * samp_rate
    psth = np.random.randn(nsamps)
    psth[psth < thresh] = 0.0

    #smooth psth
    kt = np.arange(-smooth_win_size, smooth_win_size+1, 1.0)
    k = np.exp(-kt**2)
    k /= k.sum()
    psth = np.convolve(psth, k, mode='same')

    return psth


def plot_raster(spike_trials, ax=None, duration=None, bin_size=0.001, time_offset=0.0, ylabel='Trial #'):
    """
        Make a raster plot of the trials of spike times.

        spike_trials: an array of arrays of spike times in seconds.
        time_offset: amount of time in seconds to offset the time axis for plotting
    """

    if ax is None:
        ax = plt.gca()

    if duration is None:
        duration = -np.inf
        for trial in spike_trials:
            if len(trial) > 0:
                duration = max(duration, np.max(trial))

    nbins = (duration / bin_size)

    for k,trial in enumerate(spike_trials):

        if len(trial) == 0:
            continue
        for st in trial:
            y = len(spike_trials) - k - 1
            x = int((st - time_offset) / bin_size)
            rect = Rectangle( (x, y), width=1, height=1)
            ax.add_patch(rect)

    ax.autoscale_view()

    ax.set_xlim(0.0, nbins)
    ax.figure.canvas.draw()
    xt_oldlabels = [x.get_text() for x in ax.get_xticklabels()]
    xt_newlabels = []
    for xtl in xt_oldlabels:
        try:
            xt = (float(xtl) * bin_size) + time_offset
            xt_nl = '%0.1f' % xt
        except:
            xt_nl = ''
        xt_newlabels.append(xt_nl)
    ax.set_xticklabels(xt_newlabels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')
    plt.axis('tight')


def xcorr_hist(spike_train1, spike_train2, duration=None, window_size=0.001, sample_rate=1000.0, normalize=True):
    """
        Make a cross-correlation histogram of coincident spike times between spike train 1 and 2. The cross-correlation
        histogram is a function of time. At each moment t im time, the value of the histogram is given as the number
        of spike pairs from train 1 and 2 that are within the window specified by window_size.

        Normalization means to divide by window_size*int(duration*sample_rate), which turns the returned quantity into
        the probability of spikes from the two trains co-occurring.

        Returns t,xhist,clow,chigh where t is the time vector, xhist is the cross-correlation histogram, and clow and chigh
        are the lower and upper 95% confidence intervals. When a normalized xhist falls between these
    """

    if duration is None:
        duration = -np.inf
        for st in spike_train1:
            if len(st) > 0:
                duration = np.max(np.max(st), duration)

    #construct the histogram
    nbins = int(np.ceil(duration*sample_rate))
    xhist = np.zeros([nbins], dtype='int')

    half_window_size = window_size / 2
    #populate the histogram
    for t in range(nbins):
        tmin = t/sample_rate - half_window_size
        tmax = t/sample_rate + half_window_size
        #count the number of spikes that occur in this time window
        ns1 = ((spike_train1 >= tmin) & (spike_train1 <= tmax)).sum()
        ns2 = ((spike_train2 >= tmin) & (spike_train2 <= tmax)).sum()
        #compute the count of all pairs, this is the value for the histogram
        xhist[t] = ns1*ns2

    R = int(duration*sample_rate)
    if normalize:
        xhist = xhist.astype('float') / (window_size * R)

    #compute confidence intervals
    clow = -1.96 / np.sqrt(4*window_size*R)
    chigh = 1.96 / np.sqrt(4*window_size*R)

    t = np.arange(nbins)*(1.0 / sample_rate)
    return t,xhist,clow,chigh
