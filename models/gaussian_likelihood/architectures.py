import tensorflow as tf
from tensorflow import keras

from components.layers import Dense2d
from components.layers import Sampling
from components.layers import StackNTimes

class SparseLinear(keras.layers.Layer):
    def __init__(self, output_dim, min_log_var=-14):
        super(SparseLinear, self).__init__()
        self.output_dim = output_dim
        self.min_log_var = min_log_var
        
        self.stack_inputs = StackNTimes(axis=1)      
        
        self.mean = Dense2d(1)
        
        self.log_var = Dense2d(1, activation='elu')
        
        self.sample_normal = Sampling(dist='normal')
        
    def call(self, inputs, training):
        
        y_p = inputs[0]
        samples = inputs[1]
        
        h_y_p = self.stack_inputs(y_p, samples)
        
        mean = self.mean(h_y_p)
        mean = tf.squeeze(mean, axis=-1)
        
        log_var = (-self.min_log_var)*self.log_var(h_y_p/(-self.min_log_var))
        log_var = tf.squeeze(log_var, axis=-1)
        std = tf.exp(log_var/2)
        
        eps = self.sample_normal(params=[tf.constant(0, dtype=tf.float32), 
                                         tf.constant(1, dtype=tf.float32)], 
                                 shape=tf.shape(mean))
        
        y_params = tf.stack([mean, log_var], axis=-1)
        
        y = mean + std*eps
        
        return [y_params, y]