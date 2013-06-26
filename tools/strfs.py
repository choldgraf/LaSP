import numpy as np


def strf_correlation(strf1, strf2, max_delay=10):
    """
        Computes the dot product between two STRFs across in a way that's invariant to frequency differences and time lags up to max_delay.
    """

    if strf1.shape != strf2.shape:
        raise ValueError('STRF shapes do not match: %s != %s' % (str(strf1.shape), str(strf2.shape)))
    nf = strf1.shape[0]

    time_shifts = np.arange(-max_delay, max_delay+1, 1, dtype='int')
    freq_shifts = np.arange(nf, dtype='int')
    all_ccs = np.zeros([len(time_shifts), len(freq_shifts)])
    s1 = strf1 / np.abs(strf1).max()
    s2 = strf2 / np.abs(strf2).max()
    ns1 = s1 - s1.mean()
    for i,d in enumerate(time_shifts):
        for j,f in enumerate(freq_shifts):
            #roll along frequency axis
            rs2 = np.roll(s2, f, axis=0)
            #shift and zero along time axis
            rs2 = np.roll(rs2, d, axis=1)
            if d > 0:
                rs2[:, :d] = 0.0
            elif d < 0:
                rs2[:, d:] = 0.0
            ns2 = rs2 - rs2.mean()
            cc = (ns1 * ns2).mean() / ( s1.std() * s2.std() )
            all_ccs[i, j] = cc

    all_ccs_abs = np.abs(all_ccs)
    mi = np.unravel_index(all_ccs_abs.argmax(), all_ccs.shape)
    argmax_delay,argmax_freq = time_shifts[mi[0]], freq_shifts[mi[1]]
    return argmax_delay, argmax_freq, all_ccs[mi[0], mi[1]]


def strf_mps(strf, fstep, sample_rate, half=False):

    nchannels,strflen = strf.shape
    fstrf = np.fliplr(strf)
    mps = np.fft.fftshift(np.fft.fft2(fstrf))
    amps = np.real(mps * np.conj(mps))

    #Obtain labels for frequency axis
    dwf = np.zeros([nchannels])
    fcircle = 1.0 / fstep
    for i in range(nchannels):
        dwf[i] = (i/float(nchannels))*fcircle
        if dwf[i] > fcircle/2.0:
            dwf[i] -= fcircle

    dwf = np.fft.fftshift(dwf)
    if dwf[0] > 0.0:
        dwf[0] = -dwf[0]

    #Obtain labels for time axis
    fcircle = sample_rate
    dwt = np.zeros([strflen])
    for i in range(strflen):
        dwt[i] = (i/float(strflen))*fcircle
        if dwt[i] > fcircle/2.0:
            dwt[i] -= fcircle

    dwt = np.fft.fftshift(dwt)
    if dwt[0] > 0.0:
        dwt[0] = -dwt[0]

    if half:
        halfi = np.where(dwf == 0.0)[0][0]
        amps = amps[halfi:, :]
        dwf = dwf[halfi:]

    return dwf,dwt,amps