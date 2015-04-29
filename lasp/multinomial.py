from copy import deepcopy

import numpy as np


class MultinomialClassifier(object):

    def __init__(self, l2_lambda=0):
        self.l2 = None
        if l2_lambda > 0:
            self.l2 = L2Regularizer(l2_lambda)

    def set_data(self, X, y, num_classes=None):
        #make sure things are the right size and type
        self.num_samples,self.num_features = X.shape
        assert len(y) == self.num_samples
        assert y.dtype.name.startswith('int')

        #identify classes
        if num_classes is None:
            self.classes = sorted(np.unique(y))
        else:
            self.classes = range(num_classes)
        self.num_classes = len(self.classes)

        #initialize weight matrix, include last column for bias
        self.W = np.zeros([self.num_classes, self.num_features+1])

    def update(self, X, y, step_size, threshold):

        #compute the gradient
        dW = self.gradient(X, y)

        #normalize the gradient
        gnorm = np.sqrt(np.sum(dW**2))
        dW /= gnorm

        #zero out low-valued gradients if using threshold gradient descent
        if threshold > 0.0:
            absmax = np.abs(dW).max()
            dW[np.abs(dW) < threshold*absmax] = 0.0

        #update the weights
        self.W -= step_size*dW

    def update_momentum(self, X, y, step_size, threshold):

        #compute the gradient
        dW = self.gradient(X, y)

        #normalize the gradient
        gnorm = np.sqrt(np.sum(dW**2))
        dW /= gnorm

        #zero out low-valued gradients if using threshold gradient descent
        if threshold > 0.0:
            absmax = np.abs(dW).max()
            dW[np.abs(dW) < threshold*absmax] = 0.0

        #update momentum coefficient
        self.momentum_t += 1
        if self.fixed_momentum is None:
            self.momentum = 1.0 - (3 / (self.momentum_t + 5.0))
        else:
            self.momentum = self.fixed_momentum
        #print 'momentum=%0.6f' % self.momentum

        #update weight matrix
        v = self.momentum_state_W
        vnew = self.momentum*v - step_size*dW
        self.W += vnew
        self.momentum_state_W = vnew

    def update_nesterov(self, X, y, step_size, threshold):

        #compute the gradient in a perturbed version of the weights
        Wp = self.W + self.momentum*self.momentum_state_W
        dW = self.gradient(X, y, W=Wp)

        #update the momentum state
        self.momentum_state_W = self.momentum*self.momentum_state_W - step_size*dW

        #update the weights
        self.W += self.momentum_state_W

        #update momentum coefficient
        self.momentum_t += 1
        if self.fixed_momentum is None:
            self.momentum = 1.0 - (3 / (self.momentum_t + 5.0))
        else:
            self.momentum = self.fixed_momentum
        #print 'momentum=%0.6f' % self.momentum

    def fit(self, X, y, num_classes=None, max_iters=100, threshold=0.0, step_size=1e-3, verbose=True, Xtest=None, ytest=None,
            slope_thresh=1e-6, momentum=False, fixed_momentum=None, nesterov=False):

        self.set_data(X, y, num_classes=num_classes)

        if momentum or nesterov:
            self.momentum = 0.5
            self.momentum_t = 0
            self.fixed_momentum = fixed_momentum
            self.momentum_state_W = np.zeros_like(self.W)

        num_iters = 0
        early_stopping = Xtest is not None and ytest is not None
        test_errs = np.zeros([10])

        for k in range(max_iters):

            # update the weights via gradient descent
            if momentum and not nesterov:
                self.update_momentum(X, y, step_size, threshold)
            elif nesterov:
                self.update_nesterov(X, y, step_size, threshold)
            else:
                self.update(X, y, step_size, threshold)

            # compute the error after update
            test_errs[:-1] = test_errs[1:]
            if early_stopping:
                # estimate error on the early stopping set
                test_err = self.error(Xtest, ytest) / Xtest.shape[0]
            else:
                # estimate error on the training set
                test_err = self.error(X, y) / X.shape[0]
            test_errs[-1] = test_err

            # estimate the slope of the error over iterations
            err_slope,intercept = np.polyfit(range(len(test_errs)), test_errs, 1)

            # check for convergence on the error slope
            if k >= len(test_errs):
                if err_slope >= 0.0 or (err_slope < 0.0 and np.abs(err_slope) < slope_thresh):
                    if verbose:
                        print 'Test error slope converged: err_slope=%0.6f, slope_threshold=%0.6f' % (err_slope, slope_thresh)
                    break

            #print some debug info
            if verbose:
                err = self.error(X, y) / X.shape[0]
                print 'Iteration %d: train_err=%0.6f, test_err=%0.6f, err_slope=%0.6f' % (k+1, err, test_err, err_slope)
            num_iters += 1

        if verbose:
            print 'Converged after %d iterations' % num_iters

    def activation(self, X, W=None):

        assert X.shape[1] == self.num_features

        if W is None:
            W = self.W

        #compute output activation matrix (shape is X.shape[0],self.num_classes)
        A = np.dot(X, W[:, :-1].T) + W[:, -1]
        return A

    def predict(self, X):
        P = self.predict_proba(X)
        return P.argmax(axis=1)

    def predict_proba(self, X):

        A = self.activation(X)
        #transform using softmax
        P = self.softmax(A)

        return P

    def confidence_matrix(self, X, y, labels=None):
        if labels is None:
            labels = sorted(np.unique(y))
        nclasses = len(labels)
        P = self.predict_proba(X)
        confidence = np.zeros([nclasses, nclasses])
        for k,yk in enumerate(y):
            i = labels.index(yk)
            confidence[i, :] += P[k, :]
        for i,lbl in enumerate(labels):
            confidence[i, :] /= np.sum(y == lbl)

        return confidence

    def softmax(self, A):
        expa = np.exp(A)
        expa_sum = expa.sum(axis=1)
        return (expa.T / expa_sum).T

    def error(self, X, y, W=None):

        if W is None:
            W = self.W

        #compute the output activation of each sample
        A = self.activation(X, W=W)
        self.clip_activation(A)

        #sum the exponential of each output activation per sample
        expa_sum = np.exp(A).sum(axis=1)

        #deal with exploding activations
        inf_sums = np.isinf(expa_sum)
        ninfs = inf_sums.sum()
        if ninfs > 0:
            expa_sum[inf_sums] = np.nan
            print 'WARNING: softmax activation is infinity for %d samples, throwing them away when computing error' % ninfs

        #deal with activations that are zero
        expa_sum[expa_sum <= 0.0] = 1e-4

        expa_sum_nans = np.isnan(expa_sum)
        lsum = np.log(expa_sum)
        #compute a vector of output activations for each sample's actual class
        aout = np.array([A[k, yk] for k,yk in enumerate(y)])

        #compute the error per sample
        e = lsum[~expa_sum_nans] - aout[~expa_sum_nans]

        esum = e.sum()

        # compute the regularization term
        if self.l2 is not None:
            esum += self.l2.error(W)

        #return the sum of errors across samples
        return esum

    def clip_activation(self, A, max_val=600):
        islarge = A > max_val
        nlarge = islarge.sum()
        if nlarge > 0:
            print 'WARNING: large numbers in the multinomial activation for %d samples, clipping them to %d' % (nlarge, max_val)
            A[islarge] = max_val

    def error_gradient(self, X, y, W=None):
        """ The gradient of the multinomial log likelihood with respect to each output activation, for each sample in X. """

        num_samps = X.shape[0]
        A = self.activation(X, W=W)
        self.clip_activation(A)

        dE = self.softmax(A)

        #for each sample, subtract one from the activation of the correct class
        for k in range(num_samps):
            dE[k, y[k]] -= 1.0

        return dE

    def gradient(self, X, y, W=None):
        """ Compute the gradient of the data with respect to the weights. """

        if W is None:
            W = self.W

        dE = self.error_gradient(X, y, W=W)

        dW = np.zeros_like(self.W)
        num_samps = X.shape[0]
        for k in range(num_samps):
            #print 'dE[%d, :].shape=%s' % (k, str(dE[k, :].shape))
            #print 'X[%d, :].shape=%s' % (k, str(X[k, :].shape))
            xext = np.concatenate([X[k, :], [1.0]])
            #print 'xext.shape=',xext.shape
            dW += np.outer(dE[k, :], xext)

        if self.l2 is not None:
            dW_l2 = self.l2.gradient(W)
            dW[:, :-1] += dW_l2

        return dW

    def finite_diff(self, X, y, eps=1e-6):

        base_error = self.error(X, y)
        dW = np.zeros_like(self.W)

        for k in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                Wp = deepcopy(self.W)
                Wp[k, j] += eps
                err = self.error(X, y, W=Wp)
                dW[k, j] = (err - base_error) / eps

        return dW


class L2Regularizer(object):
    def __init__(self, l2_lambda):
        self.l2_lambda = l2_lambda

    def error(self, W):
        W_sqsum = np.sum(W[:, :-1]**2)
        err_W = self.l2_lambda*W_sqsum
        return err_W

    def gradient(self, W):
        #compute gradient with respect to W
        dW = self.l2_lambda*2.0*W[:, :-1]
        return dW


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
