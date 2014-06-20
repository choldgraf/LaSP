import unittest

import numpy as np

from lasp.complex import ComplexMultinomialClassifier


class TestComplexMultinomialClassifier(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def create_test_data(self, N, M, K):

        means = np.arange(K)*2.0
        stds = np.ones(K)

        X = np.zeros([N, M], dtype='complex')
        num_total_points = 0
        for k in range(K):
            if k == K-1:
                npts = N - num_total_points
            else:
                npts = int(N / float(K))
            num_total_points += npts

            X.real = np.random.rand(N, M)*stds[k] + means[k]
            X.imag = np.random.rand(N, M)*stds[k] + means[k]

        #create a random weight matrix
        W = np.zeros([M, K], dtype='complex')
        W.real = np.random.randn(M, K)
        W.imag = np.random.randn(M, K)

        #compute the probability of each class for each data point
        R = np.dot(X, W).real
        expR = np.exp(R)
        P = (expR.T / expR.sum(axis=1)).T
        del R
        del expR

        #choose the label as the one with the highest probability
        y = P.argmax(axis=1)

        return X,y,W

    def create_classifier(self, X, y, W):
        N,M = X.shape
        K = W.shape[1]
        c = ComplexMultinomialClassifier(num_classes=K)
        c.M = M
        c.K = K
        c.N = N
        c.set_label_order(y)
        return c

    def test_likelihood(self):
        #create some fake data
        N = 100
        M = 2
        K = 3

        X,y,W = self.create_test_data(N, M, K)
        print y

        c = self.create_classifier(X, y, W)

        ll_best = c.log_likelihood(X, y, W)

        makes_sense = list()

        for k in range(100):
            #create a random weight matrix
            Wrand = np.zeros_like(W)
            Wrand.real = np.random.randn(M, K)
            Wrand.imag = np.random.randn(M, K)

            ll_rand = c.log_likelihood(X, y, Wrand)

            print 'll_best=',ll_best
            print 'll_rand=',ll_rand

            makes_sense.append(ll_best < ll_rand)

        makes_sense = np.array(makes_sense)
        print makes_sense.sum()
        assert makes_sense.sum() > 75

    def test_grad(self):

        """
        np.set_printoptions(precision=3)

        np.random.seed(12345)
        #create some fake data
        N = 100
        M = 2
        K = 3

        X,y,W = self.create_test_data(N, M, K)

        #create the classifier and fit it
        c = self.create_classifier(X, y, W)

        g = c.grad(X, y, W)
        print 'g.shape=',g.shape
        print g

        gfd = c.grad_fd(X, y, W)
        print 'gfd.shape=',gfd.shape
        print gfd

        assert 1 == 2
        """



