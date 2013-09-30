import copy

import numpy as np
from scipy.interpolate import splrep,splev
from scipy.signal import hilbert
from scipy.stats import pearsonr

import matplotlib.pyplot as plt


class HHT(object):
    """
        An implementation of the Hilbert-Huang transform. Based on code from PyHHT:

        https://github.com/jaidevd/pyhht

        Two useful papers:

        N. E. Huang et al., "The empirical mode decomposition and the Hilbert spectrum for non-linear and non
        stationary time series analysis, Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998

        Rato R.T., Ortigueira M.D., Batista A.G 2008 "On the HHT, its problems, and some solutions." Mechanical Systems
        and Signal Processing 22 1374-1394
    """

    def __init__(self):
        pass

    def find_extrema(self, s):
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

    def compute_imf(self, s, mean_tol=1e-6, stoppage_S=3, max_iter=5, remove_edge_effects=True, plot=False):
        """
            Compute an intrinsic mode function from a signal s using sifting.
        """

        stop = False
        #make a copy of the signal
        imf = copy.copy(s)
        #find extrema for first iteration
        mini,maxi = self.find_extrema(s)
        #keep track of extrema difference
        num_extrema = np.zeros([stoppage_S, 2])  # first column are maxima, second column are minima
        num_extrema[-1, :] = [len(maxi), len(mini)]
        iter = 0
        while not stop:

            #set some things up for the iteration
            s_used = s
            left_padding = 0
            right_padding = 0

            #add an extra oscillation at the beginning and end of the signal to reduce edge effects; from Rato et. al (2008) section 3.2.2
            if remove_edge_effects:
                Tl = maxi[0]  # index of left-hand (first) maximum
                tl = mini[0]  # index of left-hand (first) minimum

                Tr = maxi[-1]  # index of right hand (last) maximum
                tr = mini[-1]  # index of right hand (last) minimum

                #to reduce end effects, we need to extend the signal on both sides and reflect the first and last extrema
                #so that interpolation works better at the edges
                left_padding = max(Tl, tl)
                right_padding = len(s) - min(Tr, tr)

                #pad the original signal with zeros and reflected extrema
                s_used = np.zeros([len(s) + left_padding + right_padding])
                s_used[left_padding:-right_padding] = s

                #reflect the maximum on the left side
                imax_left = left_padding-tl
                s_used[imax_left] = s[Tl]
                #reflect the minimum on the left side
                imin_left = left_padding-Tl
                s_used[imin_left] = s[tl]

                #correct the indices on the right hand side so they're useful
                trr = len(s) - tr
                Trr = len(s) - Tr

                #reflect the maximum on the right side
                roffset = left_padding + len(s)
                imax_right = roffset+trr-1
                s_used[imax_right] = s[Tr]
                #reflect the minimum on the right side
                imin_right = roffset+Trr-1
                s_used[imin_right] = s[tr]

                #extend the array of maxima
                new_maxi = [i + left_padding for i in maxi]
                new_maxi.insert(0, imax_left)
                new_maxi.append(imax_right)
                maxi = new_maxi

                #extend the array of minima
                new_mini = [i + left_padding for i in mini]
                new_mini.insert(0, imin_left)
                new_mini.append(imin_right)
                mini = new_mini

            t = np.arange(0, len(s_used))
            fit_index = range(left_padding, len(s_used)-right_padding)

            #fit minimums with cubic splines
            spline_order = 3
            if len(mini) <= 3:
                spline_order = 1
            min_spline = splrep(mini, s_used[mini], k=spline_order)
            min_fit = splev(t[fit_index], min_spline)

            #fit maximums with cubic splines
            spline_order = 3
            if len(maxi) <= 3:
                spline_order = 1
            max_spline = splrep(maxi, s_used[maxi], k=spline_order)
            max_fit = splev(t[fit_index], max_spline)

            if plot:
                plt.figure()
                plt.plot(t[fit_index], max_fit, 'r-')
                plt.plot(maxi, s_used[maxi], 'ro')
                plt.plot(left_padding, 0.0, 'kx', markersize=10.0)
                plt.plot(left_padding+len(s), 0.0, 'kx', markersize=10.0)
                plt.plot(t, s_used, 'k-')
                plt.plot(t[fit_index], min_fit, 'b-')
                plt.plot(mini, s_used[mini], 'bo')
                plt.suptitle('Iteration %d' % iter)

            #take average of max and min splines
            z = (max_fit + min_fit) / 2.0

            #compute a factor used to dampen the subtraction of the mean spline; Rato et. al 2008, sec 3.2.3
            alpha,palpha = pearsonr(imf, z)
            alpha = min(alpha, 1e-2)

            #subtract off average of the two splines
            d = imf - alpha*z

            #set the IMF to the residual for next iteration
            imf = d

            #check for IMF S-stoppage criteria
            mini,maxi = self.find_extrema(imf)
            num_extrema = np.roll(num_extrema, -1, axis=0)
            num_extrema[-1, :] = [len(mini), len(maxi)]
            if iter >= stoppage_S:
                num_extrema_change = np.diff(num_extrema, axis=0)
                de = np.abs(num_extrema[-1, 0] - num_extrema[-1, 1])
                if np.abs(num_extrema_change).sum() == 0 and de < 2 and np.abs(imf.mean()) < mean_tol:
                    stop = True
            if iter > max_iter:
                stop = True
            print 'Iter %d: len(mini)=%d, len(maxi=%d), imf.mean()=%0.6f, alpha=%0.2f' % (iter, len(mini), len(maxi), imf.mean(), alpha)
            #print 'num_extrema=',num_extrema
            iter += 1
        return imf

    def compute_emd(self, s, max_modes=np.inf, resid_tol=1e-3, max_sift_iter=100):
        """
            Perform the empirical mode decomposition on a signal s.
        """

        self.imfs = list()
        #make a copy of the signal that will hold the residual
        r = copy.copy(s)
        stop = False
        while not stop:
            #compute the IMF from the signal
            imf = self.compute_imf(r, max_iter=max_sift_iter, mean_tol=resid_tol)
            self.imfs.append(imf)

            #subtract the IMF off to produce a new residual
            r -= imf

            #compute extrema for detecting a trend IMF
            maxi,mini = self.find_extrema(r)

            #compute convergence criteria
            if np.abs(r).sum() < resid_tol or len(self.imfs) == max_modes or (len(maxi) == 0 and len(mini) == 0):
                stop = True

        #append the residual as the last mode
        self.imfs.append(r)

    def compute_timefreq(self):
        """
            Compute a time-frequency representation of the signal by taking the instantaneous frequency
            of the hilbert transform of the intrinsic mode functions (IMFs).
        """

        for imf in self.imfs:
            #compute analytic signal of the IMF
            ht = hilbert(imf)
            #the phase of the complex signal is the instantaneous frequency
            ifreq = np.angle(ht)
            iamp = np.abs(ht)

    def decompose_imf(self, s):
        """
            Perform the "Normalized" Hilbert transform (Huang 2008 sec. 3.1) on the IMF s, decomposing the
            signal s into AM and FM components.
        """

        f = copy.copy(s)
        mini,maxi = self.find_extrema(np.abs(f))
        spline_order = 3
        if len(maxi) <= 3:
            spline_order = 1
        #TODO reflect first and last maxima to remove edge effects for interpolation
        max_spline = splrep(maxi, f[maxi], k=spline_order)
        max_fit = splev(t[fit_index], max_spline)


        #find the maximum values of |s|





