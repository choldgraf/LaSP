import numpy as np

import matplotlib.pyplot as plt
from lasp.plots import multi_plot

"""
Implementation of S. Zayd Enam's STRF modeling stuff:

S. Zayd Enam, Michael R. DeWeese, "Spectro-Temporal Models of Inferior Colliculus Neuron Receptive Fields"
http://users.soe.ucsc.edu/~afletcher/hdnips2013/papers/strfmodels_plos.pdf

"""

def checkerboard_strf(t, f, t_freq=10.0, t_phase=0.0,
                      f_freq=1e-6, f_phase=0.0, t_c=0.150, f_c=3000.0,
                      t_sigma=0.050, f_sigma=500.0, harmonic=False):

    T,F = np.meshgrid(t, f)
    t_part = np.cos(2*np.pi*t_freq*T + t_phase)
    f_part = np.cos(2*np.pi*f_freq*F + f_phase)
    exp_part = np.exp(  (-(T-t_c)**2 / (2*t_sigma**2)) - ((F - f_c)**2 / (2*f_sigma**2)) )

    if harmonic:
        f_part = np.abs(f_part)

    strf = t_part*f_part*exp_part
    strf /= np.abs(strf).max()
    return strf


def sweep_strf(t, f, theta=0.0, aspect_ratio=1.0, phase=0.0, wavelength=0.5, spread=1.0, f_c=5000.0, t_c=0.0):

    T,F = np.meshgrid(t-t_c, f-f_c)
    T /= np.abs(T).max()
    F /= np.abs(F).max()

    Tp = T*np.cos(theta) + F*np.sin(theta)
    Fp = -T*np.sin(theta) + F*np.cos(theta)

    exp_part = np.exp( -(Tp**2 + (aspect_ratio**2 * Fp**2)) / (2*spread**2) )
    cos_part = np.cos( (2*np.pi*Tp / wavelength) + phase)

    return exp_part*cos_part


if __name__ == '__main__':

    nt = 100
    t = np.linspace(0.0, 0.500)
    nf = 100
    f = np.linspace(300.0, 8000.0, nf)
    plot_extent = [t.min(), t.max(), f.min(), f.max()]


    #build onset STRFs of varying center frequency and temporal bandwidths
    onset_f_sigma = 500
    onset_f_c = np.linspace(300.0, 8000.0, 10)
    onset_t_sigmas = np.array([0.050, 0.150])

    #make onset STRFs
    plist = list()

    for f_c in onset_f_c:
        for t_sigma in onset_t_sigmas:

            strf = checkerboard_strf(t, f,
                                     t_freq=0.100, t_phase=np.pi,
                                     f_freq=1e-6, f_phase=0.0,
                                     t_c=0.150, f_c=f_c,
                                     t_sigma=t_sigma, f_sigma=500.0, harmonic=False)
            plist.append({'strf':strf, 'f_c':f_c, 't_sigma':t_sigma})

    def plot_strf(pdata, ax):
        strf = pdata['strf']
        f_c = pdata['f_c']
        t_sigma = pdata['t_sigma']
        absmax = np.abs(strf).max()
        plt.imshow(strf, interpolation='nearest', aspect='auto', origin='lower',
                   extent=plot_extent, vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.title('f_c=%dHz, t_sigma=%dHz' % (f_c, t_sigma))
        plt.xticks([])
        plt.yticks([])

    multi_plot(plist, plot_strf, nrows=len(onset_f_c), ncols=len(onset_t_sigmas))

    """
    strf = checkerboard_strf(t, f,
                             t_freq=10.0, t_phase=np.pi,
                             f_freq=1e-3, f_phase=0.0,
                             t_c=0.150, f_c=3000.0,
                             t_sigma=0.050, f_sigma=500.0, harmonic=True)

    strf = sweep_strf(t, f, theta=np.pi/4, wavelength=0.5, spread=0.25, t_c=0.100)
    """

    plt.show()
