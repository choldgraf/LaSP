import copy
import fnmatch
import os
import subprocess
import wave
import struct

import numpy as np
from scipy.io.wavfile import read as read_wavfile
from scipy.fftpack import fft,fftfreq

import matplotlib.pyplot as plt
from tools.signal import lowpass_filter,gaussian_stft


class WavFile():
    """ Class for representing a sound and writing it to a .wav file """

    def __init__(self, file_name=None, log_spectrogram=True):

        self.log_spectrogram = log_spectrogram
        if file_name is None:
            self.sample_depth = 2  # in bytes
            self.sample_rate = 44100.0  # in Hz
            self.data = None
            self.num_channels = 1
        else:
            wr = wave.open(file_name, 'r')
            self.num_channels = wr.getnchannels()
            self.sample_depth = wr.getsampwidth()
            wr.close()

            self.sample_rate,self.data = read_wavfile(file_name)

    def to_wav(self, output_file, normalize=False, max_amplitude=32767.0):
        wf = wave.open(output_file, 'w')

        wf.setparams( (self.num_channels, self.sample_depth, self.sample_rate, len(self.data), 'NONE', 'not compressed') )
        #normalize the sample
        if normalize:
            nsound = ((self.data / np.abs(self.data).max())*max_amplitude).astype('int')
        else:
            nsound = self.data
        #print 'nsound.min=%d, max=%d' % (nsound.min(), nsound.max())
        hex_sound = [struct.pack('h', x) for x in nsound]
        wf.writeframes(''.join(hex_sound))
        wf.close()

    def analyze(self, min_freq=0, max_freq=10000.0, spec_sample_rate=1000.0, freq_spacing=125.0, envelope_cutoff_freq=200.0):

        self.data_t = np.arange(0.0, len(self.data), 1.0) / self.sample_rate

        #compute the spectral envelope
        self.envelope = spectral_envelope(self.data, self.sample_rate, envelope_cutoff_freq)

        #compute log power spectrum
        fftx = fft(self.data)
        ps_f = fftfreq(len(self.data), d=(1.0 / self.sample_rate))
        findx = (ps_f > min_freq) & (ps_f < max_freq)
        self.power_spectrum = np.log10(np.abs(fftx[findx]))
        self.power_spectrum_f = ps_f[findx]

        #estimate fundamental frequency from log power spectrum in the simplest way possible
        ps = np.abs(fftx[findx])
        peak_index = ps.argmax()
        try:
            self.fundamental_freq = self.power_spectrum_f[peak_index]
        except IndexError:
            print 'Could not identify fundamental frequency!'
            self.fundamental_freq = 0.0

        #compute log spectrogram
        t,f,spec,spec_rms = spectrogram(self.data, self.sample_rate, spec_sample_rate=spec_sample_rate, freq_spacing=freq_spacing, min_freq=min_freq, max_freq=max_freq, log=self.log_spectrogram)
        self.spectrogram_t = t
        self.spectrogram_f = f
        self.spectrogram = spec
        self.spectrogram_rms = spec_rms

    def plot(self, fig=None, show_envelope=True):

        self.analyze()

        if show_envelope:
            spw_size = 15
            spec_size = 35
        else:
            spw_size = 25
            spec_size = 75

        if fig is None:
            fig = plt.figure()
        gs = plt.GridSpec(100, 1)
        ax = fig.add_subplot(gs[:spw_size])
        plt.plot(self.data_t, self.data, 'k-')
        plt.axis('tight')
        plt.ylabel('Sound Pressure')

        s = (spw_size+5)
        e = s + spec_size
        ax = fig.add_subplot(gs[s:e])
        plot_spectrogram(self.spectrogram_t, self.spectrogram_f, self.spectrogram, ax=ax, ticks=True)

        if show_envelope:
            ax = fig.add_subplot(gs[(e+5):95])
            plt.plot(self.spectrogram_t, self.spectrogram_rms, 'g-')
            plt.xlabel('Time (s)')
            plt.ylabel('Envelope')
            plt.axis('tight')


def plot_spectrogram(t, freq, spec, ax=None, ticks=True, fmin=None, fmax=None):
    if ax is None:
        ax = plt.gca()

    if fmin is None:
        fmin = freq.min()
    if fmax is None:
        fmax = freq.max()
    pfreq = freq[(freq >= fmin) & (freq <= fmax)]

    ex = (t.min(), t.max(), pfreq.min(), pfreq.max())
    iax = ax.imshow(spec, aspect='auto', interpolation='nearest', origin='lower', extent=ex)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_ylabel('Frequency (Hz)')


def play_sound(file_name):
    """ Install sox to get this to work: http://sox.sourceforge.net/ """
    subprocess.call(['play', file_name])


def spectrogram(s, sample_rate, spec_sample_rate, freq_spacing, min_freq=0, max_freq=None, nstd=6, log=True, noise_level_db=80):
    """
        Given a sound pressure waveform, compute the log spectrogram. See documentation on gaussian_stft for arguments and return values.

        log: whether or not to take the log of th power and convert to decibels, defaults to True
        noise_level_db: the threshold noise level in decibels, anything below this is set to zero. unused of log=False
    """

    increment = 1.0 / spec_sample_rate
    window_length = nstd / (2.0*np.pi*freq_spacing)
    t,freq,timefreq,rms = gaussian_stft(s, sample_rate, window_length, increment, nstd=nstd, min_freq=min_freq, max_freq=max_freq)

    if log:
        #create log spectrogram (power in decibels)
        spec = 20.0*np.log10(np.abs(timefreq)) + noise_level_db
        #rectify spectrogram
        spec[spec < 0.0] = 0.0
    else:
        spec = np.abs(timefreq)

    return t,freq,spec,rms


def spectral_envelope(s, sample_rate, cutoff_freq=200.0):
    """
        Get the spectral envelope from the sound pressure waveform.

        s: the signal
        sample_rate: the sample rate of the signal
        cutoff_freq: the cutoff frequency of the low pass filter used to create the envelope

        Returns the spectral envelope of the signal, with same sample rate.
    """

    srect = copy.copy(s)
    #rectify
    srect[srect < 0.0] = 0.0
    #low pass filter
    sfilt = lowpass_filter(srect, sample_rate, cutoff_freq)
    #re-rectify
    sfilt[sfilt < 0.0] = 0.0
    return sfilt


def recursive_ls(root_dir, file_pattern):
    """
        Walks through all the files in root_dir and returns every file whose name matches
        the pattern specified by file_pattern.
    """

    matches = list()
    for root, dirnames, filenames in os.walk(root_dir):
      for filename in fnmatch.filter(filenames, file_pattern):
          matches.append(os.path.join(root, filename))
    return matches


def sox_convert_to_mono(file_path):
    """
        Uses Sox (sox.sourceforge.net) to convert a stereo .wav file to mono.
    """

    root_dir,file_name = os.path.split(file_path)

    base_file_name = file_name[:-4]
    output_file_path = os.path.join(root_dir, '%s_mono.wav' % base_file_name)
    cmd = 'sox \"%s\" -c 1 \"%s\"' % (file_path, output_file_path)
    print '%s' % cmd
    subprocess.call(cmd, shell=True)
