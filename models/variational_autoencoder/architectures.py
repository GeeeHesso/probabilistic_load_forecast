import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from components.layers import Dense2d
from components.layers import Sampling
from components.layers import StackNTimes

class MLPCovarianceEncoder(keras.Model):
    def __init__(self, latent_dim, min_log_var=12, width=512,
                 dropout_y=0.,
                 concat_x=True, dropout_x=0.):
        super(MLPCovarianceEncoder, self).__init__()
        self.min_log_var = min_log_var
        self.concat_x = concat_x
        
        self.dropout_y = keras.layers.Dropout(rate=dropout_y)
        
        if self.concat_x:
            self.dropout_x = keras.layers.Dropout(rate=dropout_x)
        
        self.stack_inputs = StackNTimes(axis=1)
        
        self.dense_1 = keras.layers.Dense(width, activation='relu')
        self.dense_2 = keras.layers.Dense(width, activation='relu')
        
        self.dense_1_m = keras.layers.Dense(width, activation='relu')
        self.dense_2_m = keras.layers.Dense(width, activation='relu')
        self.mean = keras.layers.Dense(latent_dim)
        
        self.dense_1_lv = keras.layers.Dense(width, activation='relu')
        self.dense_2_lv = keras.layers.Dense(width, activation='relu')
        self.log_var = keras.layers.Dense(latent_dim, activation='elu')
        
        self.dense_1_cov = keras.layers.Dense(width, activation='relu')
        self.dense_2_cov = keras.layers.Dense(width, activation='relu')
        self.cov = keras.layers.Dense(latent_dim*(latent_dim-1)/2)
        
        self.sample_normal = Sampling(dist='normal')
        
    def call(self, inputs, training):
        
        x = inputs[0]
        y = inputs[1]
        samples = inputs[2]
        
        y_on = tf.cast((y!=0), tf.float32)
        
        y = self.dropout_y(y, training)
        
        y = tf.concat([y, y_on], axis=-1)
        
        if self.concat_x:
            x = self.dropout_x(x, training)
            y = tf.concat([y, x], axis=-1)
        
        y = self.stack_inputs(y, samples)
                
        h = self.dense_1(y)
        h = self.dense_2(h)
        
        h_m = self.dense_1_m(h)
        h_m = self.dense_2_m(h_m)
        mean = self.mean(h_m)
        
        h_lv = self.dense_1_lv(h)
        h_lv = self.dense_2_lv(h_lv)
        log_var = self.min_log_var*self.log_var(h_lv/self.min_log_var)   
        std = tf.exp(log_var/2)
        
        h_cov = self.dense_1_cov(h)
        h_cov = self.dense_2_cov(h_cov)
        cov = self.cov(h_cov)
        
        L = tfp.math.fill_triangular(cov, upper=True)
        paddings = tf.zeros(shape=[tf.rank(L)-2, 2], dtype=tf.int32)
        paddings = tf.concat([paddings, tf.constant([[0,1],
                                                     [1,0]])], axis=0)
        L = tf.pad(L, paddings)
        
        z_params = tf.stack([mean, log_var], axis=-1)
        z_params = tf.concat([z_params, L], axis=-1)
        
        eps = self.sample_normal(params=[tf.constant(0, dtype=tf.float32), 
                                         tf.constant(1, dtype=tf.float32)], 
                                 shape=tf.shape(mean))
        
        z = mean + tf.linalg.matvec(L + tf.linalg.diag(std), eps)
        
        return [z_params, z]
    
class MLPPrior(keras.Model):
    def __init__(self, latent_dim, min_log_var=12, width=32,
                 dropout_x=0., fit_variance=True):
        super(MLPPrior, self).__init__()
        self.latent_dim = latent_dim
        self.min_log_var = min_log_var
        self.fit_variance = fit_variance
        
        self.dropout_x = keras.layers.Dropout(rate=dropout_x)
        
        self.stack_inputs = StackNTimes(axis=1)
        
        init = keras.initializers.Zeros()
        
        self.dense2d_1_m = Dense2d(width, activation='relu')
        self.dense2d_2_m = Dense2d(width, activation='relu')
        self.mean = Dense2d(1, kernel_initializer=init, bias_initializer=init)
        
        if self.fit_variance:
            self.dense2d_1_lv = Dense2d(width, activation='relu')
            self.dense2d_2_lv = Dense2d(width, activation='relu')
            self.log_var = Dense2d(1, activation='elu',
                                   kernel_initializer=init,
                                   bias_initializer=init)
        
        self.sample_normal = Sampling(dist='normal')
        
    def call(self, inputs, training):
        x = inputs[0]
        samples = inputs[1]
        
        x = self.dropout_x(x, training)
        
        x = self.stack_inputs(x, samples)
        
        h = tf.stack(self.latent_dim*[x], axis=-2)  
        
        h_m = self.dense2d_1_m(h)
        h_m = self.dense2d_2_m(h_m)
        h_m = self.mean(h_m)
        mean = tf.squeeze(h_m, axis=-1)
        
        if self.fit_variance:
            h_lv = self.dense2d_1_lv(h)
            h_lv = self.dense2d_2_lv(h_lv)
            h_lv = self.min_log_var*self.log_var(h_lv/self.min_log_var)   
            log_var = tf.squeeze(h_lv, axis=-1)
            
        else:
            log_var = tf.zeros(mean.shape)
            
        std = tf.exp(log_var/2)
        
        z_params = tf.stack([mean, log_var], axis=-1)
        
        eps = self.sample_normal(params=[tf.constant(0, dtype=tf.float32), 
                                         tf.constant(1, dtype=tf.float32)], 
                                 shape=tf.shape(mean))
        
        z = mean + std*eps
        
        return [z_params, z]
    
class LinearPrior(keras.Model):
    def __init__(self, latent_dim, min_log_var=12,
                 dropout_x=0., fit_variance=True):
        super(LinearPrior, self).__init__()
        self.latent_dim = latent_dim
        self.min_log_var = min_log_var
        self.fit_variance = fit_variance
        
        self.dropout_x = keras.layers.Dropout(rate=dropout_x)
        
        self.stack_inputs = StackNTimes(axis=1)
        
        init = keras.initializers.Zeros()
        
        self.mean = Dense2d(1, kernel_initializer=init, bias_initializer=init)
        
        if self.fit_variance:
            self.log_var = Dense2d(1, activation='elu',
                                   kernel_initializer=init,
                                   bias_initializer=init)
        
        self.sample_normal = Sampling(dist='normal')
        
    def call(self, inputs, training):
        x = inputs[0]
        samples = inputs[1]
        
        x = self.dropout_x(x, training)
        
        x = self.stack_inputs(x, samples)
        
        h = tf.stack(self.latent_dim*[x], axis=-2)  
        
        h_m = self.mean(h)
        mean = tf.squeeze(h_m, axis=-1)
        
        if self.fit_variance:
            h_lv = self.min_log_var*self.log_var(h/self.min_log_var)   
            log_var = tf.squeeze(h_lv, axis=-1)
            
        else:
            log_var = tf.zeros(mean.shape)
            
        std = tf.exp(log_var/2)
        
        z_params = tf.stack([mean, log_var], axis=-1)
        
        eps = self.sample_normal(params=[tf.constant(0, dtype=tf.float32), 
                                         tf.constant(1, dtype=tf.float32)], 
                                 shape=tf.shape(mean))
        
        z = mean + std*eps
        
        return [z_params, z]
    
class ARPrior(keras.Model):
    def __init__(self, latent_dim, encoder, min_log_var=12, width=32,
                 fit_variance=True):
        super(ARPrior, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.min_log_var = min_log_var
        self.fit_variance = fit_variance
        
        init = keras.initializers.Zeros()
        
        self.mean = Dense2d(1, kernel_initializer=init, bias_initializer=init)
        
        if self.fit_variance:
            self.log_var = Dense2d(1, activation='elu',
                                   kernel_initializer=init,
                                   bias_initializer=init)
        
        self.sample_normal = Sampling(dist='normal')
        
    def call(self, inputs, training):
        x = inputs[0]
        samples = inputs[1]
        
        # Transpose to (batch, lag, node) for compatibility with encoder
        y_p = tf.transpose(x, [0, 2, 1])
        
        _, z_p = self.encoder([None, y_p, samples], training)
        
        # Transpose back to (batch, sample, node, lag)
        z_p = tf.transpose(z_p, [0, 1, 3, 2])
        

        
        h_m = self.mean(z_p)
        mean = tf.squeeze(h_m, axis=-1)
        
        if self.fit_variance:
            h_lv = self.min_log_var*self.log_var(z_p/self.min_log_var)   
            log_var = tf.squeeze(h_lv, axis=-1)
            
        else:
            log_var = tf.zeros(mean.shape)
            
        std = tf.exp(log_var/2)
        
        z_params = tf.stack([mean, log_var], axis=-1)
        
        eps = self.sample_normal(params=[tf.constant(0, dtype=tf.float32), 
                                         tf.constant(1, dtype=tf.float32)], 
                                 shape=tf.shape(mean))
        
        z = mean + std*eps
        
        return [z_params, z]
    
class MLPDecoder(keras.Model):
    def __init__(self, output_dim, min_log_var=12, width=32,
                 conditional_bernoulli=True, min_p=0.01, 
                 skip_connections=False, dropout_x=0.):
        super(MLPDecoder, self).__init__()
        self.output_dim = output_dim
        self.min_log_var = min_log_var
        self.conditional_bernoulli = conditional_bernoulli
        self.min_p = min_p
        self.skip_connections = skip_connections
        
        if skip_connections:
            self.dropout_x = keras.layers.Dropout(rate=dropout_x)
        
        self.stack_inputs = StackNTimes(axis=1)
        
        init = keras.initializers.Zeros()
        
        if self.conditional_bernoulli:
            self.dense2d_1_p = Dense2d(width, activation='relu')
            self.dense2d_2_p = Dense2d(width, activation='relu')
            self.prob = Dense2d(1, activation='sigmoid',
                                kernel_initializer=init,
                                bias_initializer=init)
        
        self.sample_uniform = Sampling(dist='uniform')
        
        self.dense2d_1_m = Dense2d(width, activation='relu')
        self.dense2d_2_m = Dense2d(width, activation='relu')
        self.mean = Dense2d(1)
        
        self.dense2d_1_lv = Dense2d(width, activation='relu')
        self.dense2d_2_lv = Dense2d(width, activation='relu')
        self.log_var = Dense2d(1, activation='elu')
        
        self.sample_normal = Sampling(dist='normal')
        
    def call(self, inputs, training):
        x = inputs[0]
        z = inputs[1]        
        samples = inputs[2]
        
        if self.skip_connections:
            x = self.dropout_x(x, training)
            x = self.stack_inputs(x, samples)
            z = tf.concat([z, x], axis=-1)
        
        h = tf.stack(self.output_dim*[z], axis=-2)
        
        h_m = self.dense2d_1_m(h)
        h_m = self.dense2d_2_m(h_m)
        h_m = self.mean(h_m)
        mean = tf.squeeze(h_m, axis=-1)
        
        h_lv = self.dense2d_1_lv(h)
        h_lv = self.dense2d_2_lv(h_lv)
        h_lv = self.min_log_var*self.log_var(h_lv/self.min_log_var)
        log_var = tf.squeeze(h_lv, axis=-1)
        std = tf.exp(log_var/2)
        
        normal = self.sample_normal(params=[tf.constant(0, dtype=tf.float32), 
                                            tf.constant(1, dtype=tf.float32)], 
                                    shape=tf.shape(mean))
        
        y = mean + std*normal
        
        if self.conditional_bernoulli:
            h_p = self.dense2d_1_p(h)
            h_p = self.dense2d_2_p(h_p)
            h_p = self.min_p + (1-2*self.min_p)*self.prob(h_p)
            prob = tf.squeeze(h_p, axis=-1)
        
            uniform = self.sample_uniform(params=[tf.constant(0, dtype=tf.float32), 
                                                tf.constant(1, dtype=tf.float32)], 
                                          shape=tf.shape(prob))
            y_on = tf.cast((uniform<prob), tf.float32)
            
            y_params = tf.stack([mean, log_var, prob], axis=-1)
            
            y = y_on*y
            
        else:
            y_params = tf.stack([mean, log_var], axis=-1)

        return [y_params, y]   
    
class LinearDecoder(keras.Model):
    def __init__(self, output_dim, min_log_var=12,
                 conditional_bernoulli=True, min_p=0.01, 
                 skip_connections=False, dropout_x=0.):
        super(LinearDecoder, self).__init__()
        self.output_dim = output_dim
        self.min_log_var = min_log_var
        self.conditional_bernoulli = conditional_bernoulli
        self.min_p = min_p
        self.skip_connections = skip_connections
        
        if skip_connections:
            self.dropout_x = keras.layers.Dropout(rate=dropout_x)
        
        self.stack_inputs = StackNTimes(axis=1)
        
        init = keras.initializers.Zeros()
        
        if self.conditional_bernoulli:
            self.prob = Dense2d(1, activation='sigmoid',
                                kernel_initializer=init,
                                bias_initializer=init)
        
        self.sample_uniform = Sampling(dist='uniform')
        
        self.mean = Dense2d(1)
        
        self.log_var = Dense2d(1, activation='elu')
        
        self.sample_normal = Sampling(dist='normal')
        
    def call(self, inputs, training):
        x = inputs[0]
        z = inputs[1]        
        samples = inputs[2]
        
        if self.skip_connections:
            x = self.dropout_x(x, training)
            x = self.stack_inputs(x, samples)
            z = tf.concat([z, x], axis=-1)
        
        h = tf.stack(self.output_dim*[z], axis=-2)
        
        h_m = self.mean(h)
        mean = tf.squeeze(h_m, axis=-1)
        
        h_lv = self.min_log_var*self.log_var(h/self.min_log_var)
        log_var = tf.squeeze(h_lv, axis=-1)
        std = tf.exp(log_var/2)
        
        normal = self.sample_normal(params=[tf.constant(0, dtype=tf.float32), 
                                            tf.constant(1, dtype=tf.float32)], 
                                    shape=tf.shape(mean))
        
        y = mean + std*normal
        
        if self.conditional_bernoulli:
            h_p = self.min_p + (1-2*self.min_p)*self.prob(h)
            prob = tf.squeeze(h_p, axis=-1)
        
            uniform = self.sample_uniform(params=[tf.constant(0, dtype=tf.float32), 
                                                tf.constant(1, dtype=tf.float32)], 
                                          shape=tf.shape(prob))
            y_on = tf.cast((uniform<prob), tf.float32)
            
            y_params = tf.stack([mean, log_var, prob], axis=-1)
            
            y = y_on*y
            
        else:
            y_params = tf.stack([mean, log_var], axis=-1)

        return [y_params, y]   