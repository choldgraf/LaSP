import copy
import os
import husl
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import uniform,truncnorm


def create_network(num_neurons, num_input_features, spectral_radius=1.0, nonlinearity='tanh',
                   connection_probability=0.10, lowest_weight=0.01,
                   weight_distribution='truncnorm',
                   bias_mean=0.0, bias_std=1.0, input_connection_probability=0.10):
    """ Create a recurrent complex-valued neural network.

    :param num_neurons:
    :param num_input_features:
    :param spectral_radius:
    :param nonlinearity:
    :param connection_probability:
    :param weight_distribution:
    :param weight_distribution_params:
    :param bias_mean:
    :param bias_std:
    :param input_connection_probability:

    :return: W,b,f,Win: W is the recurrent weight matrix, of size num_neurons X num_neurons. b is a complex valued
             bias vector of length num_neurons. Win is the input gain matrix, of size num_neurons X num_features. f is
             the activation function of the neuron.
    """

    #initialize random variable for weight
    if weight_distribution is None or weight_distribution == 'uniform':
        weight_rv = uniform(loc=lowest_weight, scale=1.0)
    elif weight_distribution == 'truncnorm':
        weight_rv = truncnorm(lowest_weight, 1.0)

    #initialize a random variable for phase
    phase_rv = uniform(loc=-np.pi, scale=2*np.pi)

    #create weight matrix
    W = np.zeros([num_neurons, num_neurons], dtype='complex')

    #randomly connect up each neuron, using the specified weight distribution and sparseness
    indices = np.arange(num_neurons)
    nconnections = int(connection_probability*num_neurons)
    for k in range(num_neurons):
        #shuffle the index matrix, neurons from 0 to nconnections will be selected to connect to
        np.random.shuffle(indices)
        weights = np.zeros([nconnections], dtype='complex')

        #generate random magnitudes according to the weight distribution
        wmags = weight_rv.rvs(nconnections)

        #generate a random phase
        wphase = phase_rv.rvs(nconnections)

        #initialize the weights to random values
        weights.real = wmags*np.cos(wphase)
        weights.imag = wmags*np.sin(wphase)

        #rescale weights
        #weights /= np.max(np.sqrt(weights.real**2 + weights.imag**2))

        #set weights in the connection matrix
        W[k, indices[:nconnections]] = weights

    #remove self-connections in weight matrix
    W[np.diag_indices(num_neurons)] = complex(0.0, 0.0)

    #progressively rescale the weight matrix until it's spectral radius is set
    max_decrease_factor = 0.90
    min_decrease_factor = 0.99
    evals,evecs = np.linalg.eig(W)
    initial_eigenvalue = np.max(np.abs(evals))
    max_eigenvalue = initial_eigenvalue
    while max_eigenvalue > spectral_radius:
        #inverse distance to 1.0, a number between 0.0 (far) and 1.0 (close)
        d = 1.0 - ((max_eigenvalue - spectral_radius) / abs(initial_eigenvalue - spectral_radius))

        decrease_factor = max_decrease_factor + d*(min_decrease_factor-max_decrease_factor)
        W *= decrease_factor
        evals,evecs = np.linalg.eig(W)
        max_eigenvalue = np.max(np.abs(evals))

    #set the output nonlinearity
    f = lambda x: x
    if nonlinearity == 'tanh':
        f = np.tanh

    #set bias weights
    b = np.zeros([num_neurons], dtype='complex')
    b.real = np.random.randn(num_neurons)*bias_std + bias_mean
    b.imag = np.random.randn(num_neurons)*bias_std + bias_mean

    #create input weight matrix
    indices = np.arange(num_neurons)
    Win = np.zeros([num_neurons, num_input_features])
    nconnections = int(input_connection_probability*num_neurons)
    for k in range(num_input_features):
        np.random.shuffle(indices)
        Win[indices[:nconnections], k] = weight_rv.rvs(nconnections)

    return W,b,f,Win


def make_phase_image(amp, phase):
    """
        Turns a phase matrix into an amplitude-modulated image to be plotted with imshow.
    """

    nx,ny = amp.shape
    max_amp = np.percentile(amp, 97)

    img = np.zeros([nx, ny, 4], dtype='float32')

    #set the alpha and color for the bins
    alpha = amp / max_amp
    alpha[alpha > 1.0] = 1.0 #saturate
    alpha[alpha < 0.05] = 0.0 #nonlinear threshold

    cnorm = ((180.0 / np.pi) * phase).astype('int')
    for j in range(nx):
        for ti in range(ny):
            img[j, ti, :3] = husl.husl_to_rgb(cnorm[j, ti], 75.0, 50.0) #use HUSL color space: https://github.com/boronine/pyhusl/tree/v2.1.0

    img[:, :, 3] = alpha

    return img


def network_movie(N, T, temp_dir='/tmp',
                  spectral_radius=1.0, nonlinearity='tanh',
                  connection_probability=0.10, lowest_weight=0.01,
                  weight_distribution='truncnorm',
                  bias_mean=0.0, bias_std=1.0, input_connection_probability=0.10):
    """ Makes a movie of how a set of complex values are changed on the plane by applying an elementwise
        output nonlinearity.

    :param N:
    :param T:
    :return:
    """

    #generate random points
    z = np.zeros([N], dtype='complex')
    z.real = np.random.randn(N)
    z.imag = np.random.randn(N)

    #normalize initial state
    z /= np.max(np.sqrt(z.real**2 + z.imag**2))
    z *= 0.99

    W,b,f,Win = create_network(N, 1, spectral_radius=spectral_radius, nonlinearity=nonlinearity,
                               connection_probability=connection_probability, lowest_weight=lowest_weight,
                               weight_distribution=weight_distribution,
                               bias_mean=bias_mean, bias_std=bias_std,
                               input_connection_probability=input_connection_probability)

    plt.figure()

    #plot weight matrix
    Wamp = np.abs(W)
    Wphase = np.angle(W)
    Wimg = make_phase_image(Wamp, Wphase)

    ax = plt.subplot(2, 3, 1)
    ax.set_axis_bgcolor('black')
    plt.imshow(Wimg, interpolation='nearest', aspect='auto')
    plt.title('Weight Matrix')

    #weight magnitude histogram
    plt.subplot(2, 3, 2)

    nzindex = Wamp > 0.0
    plt.hist(Wamp[nzindex].ravel(), bins=20, color='g')
    plt.axis('tight')
    plt.title('Weight Magnitudes')

    #weight phase histogram
    plt.subplot(2, 3, 3)
    plt.hist(Wphase[nzindex], bins=20, color='r')
    plt.axis('tight')
    plt.title('Weight Phase')

    #eigenvalue plot
    evals,evecs = np.linalg.eig(W)
    evals_x = [e.real for e in evals]
    evals_y = [e.imag for e in evals]

    theta = np.arange(0.0, 2*np.pi, 0.001)
    circ_x = np.cos(theta)
    circ_y = np.sin(theta)

    plt.subplot(2, 3, 4)
    plt.plot(circ_x, circ_y, 'k-', alpha=0.75)
    plt.axhline(0.0, c='k', alpha=0.75)
    plt.axvline(0.0, c='k', alpha=0.75)
    plt.plot(evals_x, evals_y, 'go')
    plt.title('Eigenvalues')

    #bias magnitude histogram
    plt.subplot(2, 3, 5)
    plt.hist(b.real, bins=15, color='g')
    plt.axis('tight')
    plt.title('Real{Bias}')

    #bias phase histogram
    plt.subplot(2, 3, 6)
    plt.hist(b.imag, bins=15, color='r')
    plt.axis('tight')
    plt.title('Imag{Bias}')

    plt.show()

    print 'Maximum eigenvalue of W: %f' % np.max(np.abs(evals))

    #generate random colors
    c = np.random.rand(N, 4)
    c[:, 3] = 1.0

    for t in range(T):

        fname = os.path.join(temp_dir, 'complex_%04d.png' % t)
        if os.path.exists(fname):
            continue

        print 'Rendering time point %d to %s' % (t, fname)
        fig = plt.figure()

        plt.suptitle('t=%d' % t)
        plt.axhline(0.0, c='k')
        plt.axvline(0.0, c='k')

        for k in range(N):
            plt.plot(z[k].real, z[k].imag, 'o', c=c[k])

        plt.xlim(-1.25, 1.25)
        plt.ylim(-1.25, 1.25)
        #plt.show()

        plt.savefig(fname)

        plt.close('all')
        del fig

        u = np.zeros([1], dtype='complex')
        if t > 600 and t < 650:
            u.real = 0.75
            u.imag = 0.0
        else:
            u.real = 0.0
            u.imag = 0.0

        a = np.dot(W, z) + b + np.dot(Win, u)
        z = f(a)

    fps = 10
    input_fmt = os.path.join(temp_dir, 'complex_%04d.png')
    output_file = os.path.join(temp_dir, 'complex.mp4')
    cmd = 'avconv -f image2 -r %d -i %s -c:v libx264 -preset slow -vf "scale=640:trunc(ow/a/2)*2" -crf 0 %s' % (fps, input_fmt, output_file)

    print cmd
    os.system(cmd)


class ComplexMultinomialClassifier(object):
    """A scikits.learn style classifier that takes complex valued input and finds complex-valued weights for a
        multinomial classifier.
    """

    def __init__(self, num_classes):
        self.K = num_classes

    def set_label_order(self, y):
        self.label_order = range(self.K)
        self.label2index = {lbl:j for j,lbl in enumerate(self.label_order)}

    def fit(self, X, y):
        """Fit the complex multinomial classifier using gradient descent.

        :param X: A complex matrix of size n_samples X n_features
        :param y: An integer valued matrix of labels of length n_samples
        :return:
        """

        self.K = len(self.label_order)

        self.N,self.M = X.shape
        assert len(y) == self.N

        self.coef_ = np.zeros([self.M, self.K])

    def compute_P(self, X, W):
        """Compute the conditional probability of each data point, produces a NxK matrix,
            where P[i,j] = P(Y[i] == j | X[i, :], W)

        :param X: NxM complex valued data matrix, N samples, M features
        :param y: N dimensional integer-valued label vector, where len(unique(y)) == K
        :param W: MxK complex-valued weight matrix, K different classes
        :return:
        """

        #get real part of matrix multiplication between X and W, which is an NxK matrix representing the dot product
        #between each of the N data points and the weights of each of the K output neurons
        R = np.dot(X, W).real

        #exponentiate R, making it the numerator of the softmax function
        expR = np.exp(R)

        #compute the conditional probability of each data point, produces a NxK matrix, where
        # P[i,j] = P(Y[i] != j | X[i, :], W), using the event Y[i] != j instead of Y[i] == j
        #because we need that quantity to compute the gradient later
        P = 1.0 - (expR.T / expR.sum(axis=1)).T

        #clean up
        del R
        del expR

        return P

    def log_likelihood(self, X, y, W):

        N,M = X.shape

        #get real part of matrix multiplication between X and W, which is an NxK matrix representing the dot product
        #between each of the N data points and the weights of each of the K output neurons
        R = np.dot(X, W).real

        #exponentiate R, sum across columns to produce the denominator of the softmax function for each data point
        d = np.exp(R).sum(axis=1)

        #grab the dot product corresponding to each Y[i]
        dp = np.zeros([N])
        for i,lbl in enumerate(y):
            dp[i] = R[i, self.label2index[lbl]]

        #compute the log likelihood
        ll = (dp - np.log(d)).sum()

        return -ll

    def grad(self, X, y, W):

        notP = 1.0 - self.compute_P(X, W)

        #initialize an empty gradient
        dW = np.zeros([self.M, self.K], dtype='complex')

        #compute each column of the gradient individually
        for j,lbl in enumerate(self.label_order):
            data_index = y == lbl
            dW[:, j].real = (X[data_index, :].real.T * notP[data_index, j]).sum(axis=1)
            dW[:, j].imag = ( X[data_index, :].imag.T * notP[data_index, j] * -1.0).sum(axis=1)

        return -dW

    def grad_fd(self, X, y, W, eps=1e-6):
        """ Compute a finite-difference approximation to the gradient.
        :param X: an NxM complex-valued matrix of data, with N samples and M features
        :param y: an N dimensional integer valued vector of labels
        :param W: A 2xMxK matrix of real-valued derivatives. The last dimension represents perturbations to the
                real and imaginary components of the weights, respectively.
        :return:
        """

        N,M = X.shape
        assert W.shape[0] == M
        K = W.shape[1]

        ll = self.log_likelihood(X, y, W)

        dW = np.zeros([M, K], dtype='complex')
        for m in range(M):
            for k in range(K):
                Wx = copy.copy(W)
                Wx[m, k] += complex(1e-6, 0.0)
                real_part = (self.log_likelihood(X, y, Wx) - ll) / eps
                del Wx

                Wy = copy.copy(W)
                Wy[m, k] += complex(0.0, 1e-6)
                imag_part = (self.log_likelihood(X, y, Wy) - ll) / eps
                del Wy

                dW[m, k] = complex(real_part, imag_part)

        return dW


if __name__ == '__main__':

    np.random.seed(123456)
    network_movie(100, 1000, temp_dir='/tmp/complex', nonlinearity='none', bias_std=1e-2)