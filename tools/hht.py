import copy

import numpy as np
from scipy.interpolate import splrep,splev

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

    def find_extrema(self, s, reflect=False):
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
        mini = min_env.nonzero()[0]
        maxi = max_env.nonzero()[0]

        return mini,maxi

    def compute_imf(self, s, mean_tol=1e-6, stoppage_S=5, max_iter=5, reflect=True):
        """
            Compute an intrinsic mode function from a signal s using sifting.
        """

        stop = False
        #make a copy of the signal
        imf = copy.copy(s)
        #find extrema for first iteration
        mini,maxi = self.find_extrema(s)
        #keep track of extrema difference
        extrema_diffs = [np.abs(len(maxi) - len(mini)), ]
        iter = 0
        while not stop:

            #set some things up for the iteration
            s_used = s
            left_padding = 0
            right_padding = 0

            #use reflection at the endpoints to reduce edge effects, from Rato et. al (2008) section 3.2.2
            if reflect:
                Tl = maxi[0]  # index of left-hand (first) maximum
                tl = mini[0]  # index of left-hand (first) minimum
                dl = abs(Tl - tl)

                Tr = maxi[-1]  # index of right hand (last) maximum
                tr = mini[-1]  # index of right hand (last) minimum
                dr = abs(Tr - tr)

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
            min_spline = splrep(mini, s_used[mini])
            min_fit = splev(t[fit_index], min_spline)

            #fit maximums with cubic splines
            max_spline = splrep(maxi, s_used[maxi])
            max_fit = splev(t[fit_index], max_spline)

            plt.figure()
            plt.plot(t[fit_index], max_fit, 'r-')
            plt.plot(maxi, s_used[maxi], 'ro')
            plt.plot(t, s_used, 'k-')
            plt.plot(t[fit_index], min_fit, 'b-')
            plt.plot(mini, s_used[mini], 'bo')

            #take average of max and min splines
            z = (max_fit + min_fit) / 2.0

            #subtract off average of the two splines
            d = imf - z

            #set the IMF to the residual for next iteration
            imf = d

            #check for IMF S-stoppage criteria
            mini,maxi = self.find_extrema(imf)
            extrema_diffs.append(np.abs(len(mini) - len(maxi)))
            if len(extrema_diffs) >= stoppage_S:
                ed = np.diff(extrema_diffs[-stoppage_S:])
                if np.sum(ed) == 0 and np.abs(imf.mean()) < mean_tol:
                    stop = True
            if iter > max_iter:
                stop = True
            print 'Iter %d: len(mini)=%d, len(maxi=%d), imf.mean()=%0.6f' % (iter, len(mini), len(maxi), imf.mean())
            iter += 1
        return imf

    def compute_emd(self, s, max_modes=np.inf):
        """
            Perform the empirical mode decomposition on a signal s.
        """

        imfs = list()
        #make a copy of the signal that will hold the residual
        r = copy.copy(s)
        stop = False
        while not stop:
            #compute the IMF from the signal
            imf = self.compute_imf(r)
            imfs.append(imf)

            #subtract the IMF off to produce a new residual
            r -= imf

            #compute convergence criteria
            if len(imfs) == max_modes:
                stop = True

