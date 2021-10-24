import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from components.layers import Dense2d

class SparseLinear(keras.layers.Layer):
    '''
    Input shape: (batch, node, feature)
    Output shape: (batch, node, quantile)
    '''
    def __init__(self, output_dim, quantiles):
        super(SparseLinear, self).__init__()
        self.output_dim = output_dim
        self.quantiles = quantiles
        
        self.median_idx = np.nonzero(np.array(quantiles)==0.5)[0][0]
        
        self.quantile_output = Dense2d(len(quantiles))
        
    def call(self, inputs, training):
        
        y_quantiles = self.quantile_output(inputs)
        y_median = y_quantiles[..., self.median_idx]
        
        # Set quantile axis from last to second dimension, regardless of rank
        y_quantiles = tf.stack(tf.unstack(y_quantiles, axis=-1), axis=1)
        
        return [y_quantiles, y_median]