import numpy as np

class RecurrentNetwork(object):

    def __init__(self, num_neurons=100, spectral_radius=1.0, ei_ratio=1.0, dales_rule=False, p_connect=1.0,
                 input_gain=1.0, input_dim=1, nl_gain=1.0):

        """
            num_neurons: the number of neurons in the network
            spectral_radius: the maximum of the absolute eigenvalues of the weight matrix
            ei_ratio: the ratio of excitation to inhibition, i.e. the number of excitatory neurons divided by the number
                of inhibitory neurons
            dales_rule: whether the outputs of a neuron are exclusively excitatory or inhibitory
            p_connect: the probability a neuron will connect to another neuron, sets the overall sparseness of the
                connectivity
            input_gain: the maximum absolute value of the input weight matrix
            input_dim: the dimensionality of the input
            nl_gain: the gain factor multiplied by the argument of the sigmoidal output nonlinearity.
        """

        self.num_neurons = num_neurons
        self.input_gain = input_gain
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius

        #create positive-valued weight matrix
        W = np.abs(np.random.randn([num_neurons, num_neurons]))

        #sparsify the weight matrix, making sure each neuron only connects to a fraction of it's neighbors
        if p_connect < 1.0:

            for k in range(num_neurons):
                #create an array of uniform random numbers
                rnums = np.random.rand(num_neurons)
                connections_to_keep = rnums < p_connect
                W[~connections_to_keep, k] = 0.0

        #rescale weight matrix
        W /= W.max()
        W *= spectral_radius

        #compute the fraction of inbition in the network
        inhib_frac = 1.0 / (ei_ratio + 1)

        if dales_rule:
            #interpret ei_ratio as the number of excitatory neurons to inhibitory neurons

            #create an array of uniform random numbers
            rnums = np.random.rand(num_neurons)

            #identify randomly selected inhibitory neurons
            inhib_index = rnums <= inhib_frac

            #ensure inhibitory connections have negative weights
            W[:, inhib_index] *= -1.0

        else:
            #interpret ei_ratio as the number of excitatory synapses to inhibitory synapses
            rnums = np.random.rand(num_neurons**2)




            num_inhib_synapses = int(num_neurons**2 /  (1 + ei_ratio))





