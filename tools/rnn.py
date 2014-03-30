import numpy as np

class RecurrentNetwork(object):

    def __init__(self, num_neurons=100, spectral_radius=1.0, ei_ratio=1.0, input_gain=1.0, input_dim=1, dales_rule=False):

        self.num_neurons = num_neurons
        self.input_gain = input_gain
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius

        #create positive-valued weight matrix
        W = np.abs(np.random.randn([num_neurons, num_neurons]))

        #rescale weight matrix
        W /= W.max()
        W *= spectral_radius

        if dales_rule:
            #interpret ei_ratio as the number of excitatory neurons to inhibitory neurons

            #set ratio of excitation to inhibition, first determine inhibitory neurons
            neuron_index = range(self.num_neurons)
            np.random.shuffle(neuron_index)

            num_inhib_neurons = int(num_neurons /  (1 + ei_ratio))

        else:
            #interpret ei_ratio as the number of excitatory synapses to inhibitory synapses
            num_inhib_synapses = int(num_neurons**2 /  (1 + ei_ratio))





