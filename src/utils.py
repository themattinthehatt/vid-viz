import numpy as np


class SmoothNoise(object):

    def __init__(self, num_samples=10, num_channels=1):
        """
        Returns values from a Gaussian noise process that is smoothed by 
        averaging over previous random values (moving average filter); values
        will lie approximately between -1 and 1
        
        ARGS:
            num_samples - number of samples used in moving average filter
            num_channels - number of independent noise processes
            
        """

        self.num_samples = num_samples
        self.num_channels = num_channels
        self.noise_index = 0                # counter in noise array
        self.noise_samples = np.random.randn(self.num_samples,
                                             self.num_channels)

    def reinitialize(self):
        """Reinitialize noise samples"""

        self.noise_samples = np.random.randn(self.num_samples,
                                             self.num_channels)

    def get_next_vals(self):
        """Update noise_samples and return values for each channel"""

        self.noise_samples[self.noise_index, :] = np.random.randn(
            1, self.num_channels)
        self.noise_index = (self.noise_index + 1) % self.num_samples

        return np.sqrt(self.num_samples) / 3 * \
            np.mean(self.noise_samples, axis=0)
