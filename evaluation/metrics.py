import tensorflow as tf
import tensorflow.keras as keras

class NodalMAEMeans(keras.metrics.Metric):
    def __init__(self, name='nodal_mae_means', scale=1,
                 skip_zeros=False, **kwargs):
        super(NodalMAEMeans, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.skip_zeros = skip_zeros
        
        self.summed_abs_error = tf.Variable(0, dtype=tf.float32)
        self.count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.skip_zeros:
            # Count only non-zero values for means of predictions
            sums = tf.reduce_sum(y_pred, axis=1)
            samples = tf.math.count_nonzero(y_pred, axis=1)
            means = sums/tf.cast(samples, tf.float32)
            
            # Count only non-zero real values
            means = means[y_true!=0]
            y_true = y_true[y_true!=0]
            
        else:
            # Count everything
            means = tf.reduce_mean(y_pred, axis=1)

        abs_error = tf.abs(y_true - means)
        
        self.summed_abs_error.assign_add(tf.reduce_sum(abs_error))
        self.count.assign_add(tf.size(abs_error))
    
    def result(self):
        return self.summed_abs_error/tf.cast(self.count, tf.float32)*self.scale
    
    def reset_state(self):
        self.summed_abs_error.assign(0)
        self.count.assign(0)
        
class NodalMAEMSum(keras.metrics.Metric):
    def __init__(self, name='nodal_maem_sum', scale=1, 
                 sample_axis=False, **kwargs):
        super(NodalMAEMSum, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            y_pred = tf.reduce_mean(y_pred, axis=1)
            
        error = y_true - y_pred
            
        # Take absolute value
        abs_error = tf.abs(error)
        
        # Sum metric over nodal dimension
        abs_error_sum = tf.reduce_sum(abs_error, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(abs_error_sum, axis=0))
        self.batch_count.assign_add(tf.size(abs_error_sum))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class NodalMAEMMax(keras.metrics.Metric):
    def __init__(self, name='nodal_maem_max', scale=1, 
                 sample_axis=False, **kwargs):
        super(NodalMAEMMax, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            y_pred = tf.reduce_mean(y_pred, axis=1)
            
        error = y_true - y_pred
            
        # Take absolute value
        abs_error = tf.abs(error)
        
        # Sum metric over nodal dimension
        abs_error_max = tf.reduce_max(abs_error, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(abs_error_max, axis=0))
        self.batch_count.assign_add(tf.size(abs_error_max))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class NodalMAEMMin(keras.metrics.Metric):
    def __init__(self, name='nodal_maem_min', scale=1, 
                 sample_axis=False, **kwargs):
        super(NodalMAEMMin, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            y_pred = tf.reduce_mean(y_pred, axis=1)
            
        error = y_true - y_pred
            
        # Take absolute value
        abs_error = tf.abs(error)
        
        # Sum metric over nodal dimension
        abs_error_min = tf.reduce_min(abs_error, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(abs_error_min, axis=0))
        self.batch_count.assign_add(tf.size(abs_error_min))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class NodalMAEMAverage(keras.metrics.Metric):
    def __init__(self, name='nodal_maem_average', scale=1, 
                 sample_axis=False, **kwargs):
        super(NodalMAEMAverage, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            y_pred = tf.reduce_mean(y_pred, axis=1)
            
        error = y_true - y_pred
            
        # Take absolute value
        abs_error = tf.abs(error)
        
        # Average metric over nodal dimension
        abs_error_mean = tf.reduce_mean(abs_error, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(abs_error_mean, axis=0))
        self.batch_count.assign_add(tf.size(abs_error_mean))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class NodalMAESum(keras.metrics.Metric):
    def __init__(self, name='nodal_mae_sum', scale=1, 
                 sample_axis=False, **kwargs):
        super(NodalMAESum, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            y_true = tf.expand_dims(y_true, axis=1)
            
        abs_error = tf.abs(y_true - y_pred)
        
        # Average metric over sample dimension
        if self.sample_axis:
            abs_error = tf.reduce_mean(abs_error, axis=1)
        
        # Sum metric over nodal dimension
        abs_error_sum = tf.reduce_sum(abs_error, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(abs_error_sum, axis=0))
        self.batch_count.assign_add(tf.size(abs_error_sum))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class NodalMAEMax(keras.metrics.Metric):
    def __init__(self, name='nodal_mae_max', scale=1, 
                 sample_axis=False, **kwargs):
        super(NodalMAEMax, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            y_true = tf.expand_dims(y_true, axis=1)
            
        error = y_true - y_pred
        
        # Take absolute value
        abs_error = tf.abs(error)
        
        # Average metric over sample dimension
        if self.sample_axis:
            abs_error = tf.reduce_mean(abs_error, axis=1)
        
        # Sum metric over nodal dimension
        abs_error_max = tf.reduce_max(abs_error, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(abs_error_max, axis=0))
        self.batch_count.assign_add(tf.size(abs_error_max))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class NodalMAEMin(keras.metrics.Metric):
    def __init__(self, name='nodal_mae_min', scale=1, 
                 sample_axis=False, **kwargs):
        super(NodalMAEMin, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            y_true = tf.expand_dims(y_true, axis=1)
            
        error = y_true - y_pred
        
        # Take absolute value
        abs_error = tf.abs(error)
        
        # Average metric over sample dimension
        if self.sample_axis:
            abs_error = tf.reduce_mean(abs_error, axis=1)
            
        # Sum metric over nodal dimension
        abs_error_min = tf.reduce_min(abs_error, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(abs_error_min, axis=0))
        self.batch_count.assign_add(tf.size(abs_error_min))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class NodalMAEAverage(keras.metrics.Metric):
    def __init__(self, name='nodal_mae_average', scale=1, 
                 sample_axis=False, **kwargs):
        super(NodalMAEAverage, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            y_true = tf.expand_dims(y_true, axis=1)
            
        error = y_true - y_pred
        
        # Take absolute value
        abs_error = tf.abs(error)
        
        # Average metric over sample dimension
        if self.sample_axis:
            abs_error = tf.reduce_mean(abs_error, axis=1)
        
        # Average metric over nodal dimension
        abs_error_mean = tf.reduce_mean(abs_error, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(abs_error_mean, axis=0))
        self.batch_count.assign_add(tf.size(abs_error_mean))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)

class TotalMAEM(keras.metrics.Metric):
    def __init__(self, name='total_maem', scale=1, 
                 sample_axis=False, **kwargs):
        super(TotalMAEM, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        totals_pred = tf.reduce_sum(y_pred, axis=-1)
        totals_true = tf.reduce_sum(y_true, axis=-1)
        
        # Average prediction over sample dimension
        if self.sample_axis:
            totals_pred = tf.reduce_mean(totals_pred, axis=1)
            
        abs_error = tf.abs(totals_true - totals_pred)
        
        self.batch_sum.assign_add(tf.reduce_sum(abs_error))
        self.batch_count.assign_add(tf.size(abs_error))
    
    def result(self):
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)

class TotalMAE(keras.metrics.Metric):
    def __init__(self, name='total_mae', scale=1, 
                 sample_axis=False, **kwargs):
        super(TotalMAE, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.sample_axis = sample_axis
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        totals_pred = tf.reduce_sum(y_pred, axis=-1)
        totals_true = tf.reduce_sum(y_true, axis=-1)
        
        # Add sample dimension to real data for broadcasting
        if self.sample_axis:
            totals_true = tf.expand_dims(totals_true, axis=1)
        
        abs_error = tf.abs(totals_true - totals_pred)
        
        # Average metric over sample dimension
        if self.sample_axis:
            abs_error = tf.reduce_mean(abs_error, axis=1)
        
        self.batch_sum.assign_add(tf.reduce_sum(abs_error))
        self.batch_count.assign_add(tf.size(abs_error))
    
    def result(self):
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)

class NodalCoverageProbability(keras.metrics.Metric):
    def __init__(self, name='nodal_cov_prob', conf_level=0.8, min_samples=10,
                 skip_zeros=False, **kwargs):
        super(NodalCoverageProbability, self).__init__(name=name, **kwargs)
        self.conf_level = conf_level
        self.min_samples = min_samples
        self.skip_zeros = skip_zeros
        self.lower = (1 - self.conf_level)/2
        self.upper = (1 + self.conf_level)/2
        
        self.covered_predictions = tf.Variable(0, dtype=tf.int32)
        self.total_predictions = tf.Variable(0, dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.transpose(y_pred, perm=[0, 2, 1])
        
        if self.skip_zeros:
            ## Count predictions with non-zero real value and not enough samples
            #y_ne = tf.logical_and(y_true!=0,
            #                      (tf.math.count_nonzero(y_pred, axis=2)
            #                       <self.min_samples))
            #ne_count = tf.reduce_sum(tf.cast(y_ne, tf.int32))
            
            # Keep predictions with real value and enough samples
            mask = tf.logical_and(y_true!=0, 
                                  (tf.math.count_nonzero(y_pred, axis=2)
                                   >=self.min_samples))
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            
            # Sort predictions for nodal confidence interval calculation
            y_pred = tf.sort(y_pred)
            
            # Keep only non-zero samples
            y_pred = tf.ragged.boolean_mask(y_pred, y_pred!=0)
            
            # Count samples in each prediction
            sample_count = tf.cast(y_pred.row_lengths(), tf.float32)
            
        else:
            # Sort predictions for nodal confidence interval calculation
            y_pred = tf.sort(y_pred)
            
            # Flatten into prediction and sample dimensions
            samples = tf.cast(tf.shape(y_pred)[2], tf.float32)
            y_pred = tf.reshape(y_pred, shape=[-1, samples])
            y_true = tf.reshape(y_true, shape=[-1])
            
            # Count samples in each prediction
            sample_count = tf.ones_like(y_true)*samples
            
        # Calculate lower and upper confidence interval indices
        lower_indices = self.lower*(sample_count-1)
        lower_indices = tf.cast(tf.round(lower_indices), tf.int32)
        upper_indices = self.upper*(sample_count-1)
        upper_indices = tf.cast(tf.round(upper_indices), tf.int32)
            
        # Obtain lower and upper confidence interval values
        lower = tf.gather(y_pred, lower_indices, batch_dims=1)
        upper = tf.gather(y_pred, upper_indices, batch_dims=1)
            
        # Check which confidence intervals cover the real values
        covered = tf.logical_and(lower<=y_true, y_true<=upper)
        covered = tf.cast(covered, tf.int32)
            
        # Count covered predictions
        self.covered_predictions.assign_add(tf.reduce_sum(covered))
            
        # Count total predictions 
        self.total_predictions.assign_add(tf.size(covered))
        
        ## Add predictions with non-zero real value and no non-zero sample
        #if self.skip_zeros:
        #    self.total_predictions.assign_add(ne_count)
            
    def result(self):
        covered = tf.cast(self.covered_predictions, tf.float32)
        total = tf.cast(self.total_predictions, tf.float32)
        return covered/total
    
    def reset_state(self):
        self.covered_predictions.assign(0)
        self.total_predictions.assign(0)
        
class TotalCoverageProbability(keras.metrics.Metric):
    def __init__(self, name='total_cov_prob', conf_level=0.8, **kwargs):
        super(TotalCoverageProbability, self).__init__(name=name, **kwargs)
        self.conf_level = conf_level
        self.lower = (1 - self.conf_level)/2
        self.upper = (1 + self.conf_level)/2
        
        self.covered_predictions = tf.Variable(0, dtype=tf.int32)
        self.total_predictions = tf.Variable(0, dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Sum predictions and real values over nodes to get totals
        y_pred = tf.reduce_sum(y_pred, axis=-1)
        y_true = tf.reduce_sum(y_true, axis=-1)
        
        # Sort predictions for nodal confidence interval calculation
        y_pred = tf.sort(y_pred)
        
        # Count samples in each prediction
        samples = tf.cast(tf.shape(y_pred)[1], tf.float32)
        sample_count = tf.ones_like(y_true)*samples
            
        # Calculate lower and upper confidence interval indices
        lower_indices = self.lower*(sample_count-1)
        lower_indices = tf.cast(tf.round(lower_indices), tf.int32)
        upper_indices = self.upper*(sample_count-1)
        upper_indices = tf.cast(tf.round(upper_indices), tf.int32)
            
        # Obtain lower and upper confidence interval values
        lower = tf.gather(y_pred, lower_indices, batch_dims=1)
        upper = tf.gather(y_pred, upper_indices, batch_dims=1)
            
        # Check which confidence intervals cover the real values
        covered = tf.logical_and(lower<=y_true, y_true<=upper)
        covered = tf.cast(covered, tf.int32)
            
        # Count covered predictions
        self.covered_predictions.assign_add(tf.reduce_sum(covered))
            
        # Count total predictions 
        self.total_predictions.assign_add(tf.size(covered))
        
    def result(self):
        covered = tf.cast(self.covered_predictions, tf.float32)
        total = tf.cast(self.total_predictions, tf.float32)
        return covered/total
    
    def reset_state(self):
        self.covered_predictions.assign(0)
        self.total_predictions.assign(0)
        
class NodalConfidenceIntervalWidth(keras.metrics.Metric):
    def __init__(self, name='nodal_conf_width', conf_level=0.8, scale=1,
                 min_samples=10, skip_zeros=False, **kwargs):
        super(NodalConfidenceIntervalWidth, self).__init__(name=name, **kwargs)
        self.conf_level = conf_level
        self.min_samples = min_samples
        self.skip_zeros = skip_zeros
        self.lower = (1 - self.conf_level)/2
        self.upper = (1 + self.conf_level)/2
        self.scale = scale
        
        self.conf_width_sum = tf.Variable(0, dtype=tf.float32)
        self.total_predictions = tf.Variable(0, dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.transpose(y_pred, perm=[0, 2, 1])
        
        if self.skip_zeros:        
            # Keep predictions with real value and enough samples
            mask = tf.logical_and(y_true!=0, 
                                  (tf.math.count_nonzero(y_pred, axis=2)
                                   >=self.min_samples))
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            
            # Sort predictions for nodal confidence interval calculation
            y_pred = tf.sort(y_pred)
            
            # Keep only non-zero samples
            y_pred = tf.ragged.boolean_mask(y_pred, y_pred!=0)
            
            # Count samples in each prediction
            sample_count = tf.cast(y_pred.row_lengths(), tf.float32)
            
        else:
            # Sort predictions for nodal confidence interval calculation
            y_pred = tf.sort(y_pred)
            
            # Flatten into prediction and sample dimensions
            samples = tf.cast(tf.shape(y_pred)[2], tf.float32)
            y_pred = tf.reshape(y_pred, shape=[-1, samples])
            y_true = tf.reshape(y_true, shape=[-1])
            
            # Count samples in each prediction
            sample_count = tf.ones_like(y_true)*samples
            
        # Calculate lower and upper confidence interval indices
        lower_indices = self.lower*(sample_count-1)
        lower_indices = tf.cast(tf.round(lower_indices), tf.int32)
        upper_indices = self.upper*(sample_count-1)
        upper_indices = tf.cast(tf.round(upper_indices), tf.int32)
            
        # Obtain lower and upper confidence interval values
        lower = tf.gather(y_pred, lower_indices, batch_dims=1)
        upper = tf.gather(y_pred, upper_indices, batch_dims=1)
            
        # Calculate confidence interval width
        conf_width = upper - lower
            
        # Count covered predictions
        self.conf_width_sum.assign_add(tf.reduce_sum(conf_width))
            
        # Count total predictions 
        self.total_predictions.assign_add(tf.size(conf_width))
            
    def result(self):
        total = tf.cast(self.total_predictions, tf.float32)
        return self.conf_width_sum/total*self.scale
    
    def reset_state(self):
        self.conf_width_sum.assign(0)
        self.total_predictions.assign(0)

class TotalConfidenceIntervalWidth(keras.metrics.Metric):
    def __init__(self, name='total_conf_width', conf_level=0.8, 
                 scale=1, **kwargs):
        super(TotalConfidenceIntervalWidth, self).__init__(name=name, **kwargs)
        self.conf_level = conf_level
        self.lower = (1 - self.conf_level)/2
        self.upper = (1 + self.conf_level)/2
        self.scale = scale
        
        self.conf_width_sum = tf.Variable(0, dtype=tf.float32)
        self.total_predictions = tf.Variable(0, dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Sum predictions and real values over nodes to get totals
        y_pred = tf.reduce_sum(y_pred, axis=-1)
        y_true = tf.reduce_sum(y_true, axis=-1)
        
        # Sort predictions for nodal confidence interval calculation
        y_pred = tf.sort(y_pred)
        
        # Count samples in each prediction
        samples = tf.cast(tf.shape(y_pred)[1], tf.float32)
        sample_count = tf.ones_like(y_true)*samples
            
        # Calculate lower and upper confidence interval indices
        lower_indices = self.lower*(sample_count-1)
        lower_indices = tf.cast(tf.round(lower_indices), tf.int32)
        upper_indices = self.upper*(sample_count-1)
        upper_indices = tf.cast(tf.round(upper_indices), tf.int32)
            
        # Obtain lower and upper confidence interval values
        lower = tf.gather(y_pred, lower_indices, batch_dims=1)
        upper = tf.gather(y_pred, upper_indices, batch_dims=1)
        
        # Calculate confidence interval width
        conf_width = upper - lower
            
        # Count covered predictions
        self.conf_width_sum.assign_add(tf.reduce_sum(conf_width))
            
        # Count total predictions 
        self.total_predictions.assign_add(tf.size(conf_width))
        
    def result(self):
        total = tf.cast(self.total_predictions, tf.float32)
        return self.conf_width_sum/total*self.scale
    
    def reset_state(self):
        self.conf_width_sum.assign(0)
        self.total_predictions.assign(0)
        
class NodalCRPSSum(keras.metrics.Metric):
    def __init__(self, name='nodal_crps_sum', scale=1, **kwargs):
        super(NodalCRPSSum, self).__init__(name=name, **kwargs)
        self.scale = scale
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Error between sample and real data point
        abs_error = tf.abs(tf.expand_dims(y_true, axis=1) - y_pred)
        
        # Error between two samples
        y_pred_shifted = tf.roll(y_pred, 1, axis=1)
        abs_error_samples = tf.abs(y_pred - y_pred_shifted)
        
        # Estimation of CRPS
        crps = abs_error - abs_error_samples/2
        
        # Average over sample dimension
        crps = tf.reduce_mean(crps, axis=1)
        
        # Sum over nodal dimension
        crps = tf.reduce_sum(crps, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(crps, axis=0))
        self.batch_count.assign_add(tf.size(crps))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class NodalCRPSMin(keras.metrics.Metric):
    def __init__(self, name='nodal_crps_min', scale=1, **kwargs):
        super(NodalCRPSMin, self).__init__(name=name, **kwargs)
        self.scale = scale
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Error between sample and real data point
        abs_error = tf.abs(tf.expand_dims(y_true, axis=1) - y_pred)
        
        # Error between two samples
        y_pred_shifted = tf.roll(y_pred, 1, axis=1)
        abs_error_samples = tf.abs(y_pred - y_pred_shifted)
        
        # Estimation of CRPS
        crps = abs_error - abs_error_samples/2
        
        # Average over sample dimension
        crps = tf.reduce_mean(crps, axis=1)
        
        # Minimum over nodal dimension
        crps = tf.reduce_min(crps, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(crps, axis=0))
        self.batch_count.assign_add(tf.size(crps))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)

class NodalCRPSMax(keras.metrics.Metric):
    def __init__(self, name='nodal_crps_max', scale=1, **kwargs):
        super(NodalCRPSMax, self).__init__(name=name, **kwargs)
        self.scale = scale
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Error between sample and real data point
        abs_error = tf.abs(tf.expand_dims(y_true, axis=1) - y_pred)
        
        # Error between two samples
        y_pred_shifted = tf.roll(y_pred, 1, axis=1)
        abs_error_samples = tf.abs(y_pred - y_pred_shifted)
        
        # Estimation of CRPS
        crps = abs_error - abs_error_samples/2
        
        # Average over sample dimension
        crps = tf.reduce_mean(crps, axis=1)
        
        # Maximum over nodal dimension
        crps = tf.reduce_max(crps, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(crps, axis=0))
        self.batch_count.assign_add(tf.size(crps))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)

class NodalCRPSAverage(keras.metrics.Metric):
    def __init__(self, name='nodal_crps_average', scale=1, **kwargs):
        super(NodalCRPSAverage, self).__init__(name=name, **kwargs)
        self.scale = scale
        
        self.batch_sum = tf.Variable(0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Error between sample and real data point
        abs_error = tf.abs(tf.expand_dims(y_true, axis=1) - y_pred)
        
        # Error between two samples
        y_pred_shifted = tf.roll(y_pred, 1, axis=1)
        abs_error_samples = tf.abs(y_pred - y_pred_shifted)
        
        # Estimation of CRPS
        crps = abs_error - abs_error_samples/2
        
        # Average over sample dimension
        crps = tf.reduce_mean(crps, axis=1)
        
        # Sum over nodal dimension
        crps = tf.reduce_sum(crps, axis=-1)
        
        # Average over batch dimension when querying the metric
        self.batch_sum.assign_add(tf.reduce_sum(crps, axis=0))
        self.batch_count.assign_add(tf.size(crps))
    
    def result(self):
        # Average over batch dimension
        return self.batch_sum/tf.cast(self.batch_count, tf.float32)*self.scale
    
    def reset_state(self):
        self.batch_sum.assign(0)
        self.batch_count.assign(0)
        
class TotalCRPS(keras.metrics.Metric):
    def __init__(self, name='total_crps', scale=1, **kwargs):
        super(TotalCRPS, self).__init__(name=name, **kwargs)
        self.scale = scale
        
        self.summed_crps = tf.Variable(0, dtype=tf.float32)
        self.count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        totals_pred = tf.reduce_sum(y_pred, axis=-1)
        totals_true = tf.reduce_sum(y_true, axis=-1)
        
        # Error between sample and real data point
        abs_error = tf.abs(tf.expand_dims(totals_true, axis=1) - totals_pred)
        
        # Error between two samples
        totals_pred_shifted = tf.roll(totals_pred, 1, axis=1)
        abs_error_samples = tf.abs(totals_pred - totals_pred_shifted)
        
        # Estimation of CRPS
        crps = abs_error - abs_error_samples/2

        crps = tf.reduce_mean(crps, axis=1)
        
        self.summed_crps.assign_add(tf.reduce_sum(crps))
        self.count.assign_add(tf.size(crps))
    
    def result(self):
        return self.summed_crps/tf.cast(self.count, tf.float32)*self.scale
    
    def reset_state(self):
        self.summed_crps.assign(0)
        self.count.assign(0)
        
class MAEMedian(keras.metrics.Metric):
    def __init__(self, name='nodal_mae_median', scale=1,
                 median_index=1, **kwargs):
        super(MAEMedian, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.median_index = median_index
        
        self.summed_abs_error = tf.Variable(0, dtype=tf.float32)
        self.count = tf.Variable(0, dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        median = y_pred[:, self.median_index, ...]

        abs_error = tf.abs(y_true - median)
        
        self.summed_abs_error.assign_add(tf.reduce_sum(abs_error))
        self.count.assign_add(tf.size(abs_error))
    
    def result(self):
        return self.summed_abs_error/tf.cast(self.count, tf.float32)*self.scale
    
    def reset_state(self):
        self.summed_abs_error.assign(0)
        self.count.assign(0)

class QuantileCoverageProbability(keras.metrics.Metric):
    def __init__(self, name='nodal_cov_prob', 
                 lower_quantile_index=0, upper_quantile_index=2, **kwargs):
        super(QuantileCoverageProbability, self).__init__(name=name, 
                                                          **kwargs)
        self.lower_quantile_index = lower_quantile_index
        self.upper_quantile_index = upper_quantile_index
        
        self.covered_predictions = tf.Variable(0, dtype=tf.int32)
        self.total_predictions = tf.Variable(0, dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):            
        # Obtain lower and upper confidence interval values
        lower = y_pred[:, self.lower_quantile_index, ...]
        upper = y_pred[:, self.upper_quantile_index, ...]
            
        # Check which confidence intervals cover the real values
        covered = tf.logical_and(lower<=y_true, y_true<=upper)
        covered = tf.cast(covered, tf.int32)
            
        # Count covered predictions
        self.covered_predictions.assign_add(tf.reduce_sum(covered))
            
        # Count total predictions 
        self.total_predictions.assign_add(tf.size(covered))
            
    def result(self):
        covered = tf.cast(self.covered_predictions, tf.float32)
        total = tf.cast(self.total_predictions, tf.float32)
        return covered/total
    
    def reset_state(self):
        self.covered_predictions.assign(0)
        self.total_predictions.assign(0)
        
class QuantileConfidenceIntervalWidth(keras.metrics.Metric):
    def __init__(self, name='nodal_conf_width', scale=1,
                 lower_quantile_index=0, upper_quantile_index=2, **kwargs):
        super(QuantileConfidenceIntervalWidth, self).__init__(name=name, 
                                                              **kwargs)
        self.scale = scale
        self.lower_quantile_index = lower_quantile_index
        self.upper_quantile_index = upper_quantile_index
        
        self.conf_width_sum = tf.Variable(0, dtype=tf.float32)
        self.total_predictions = tf.Variable(0, dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Obtain lower and upper confidence interval values
        lower = y_pred[:, self.lower_quantile_index, ...]
        upper = y_pred[:, self.upper_quantile_index, ...]
            
        # Calculate confidence interval width
        conf_width = upper - lower
            
        # Count covered predictions
        self.conf_width_sum.assign_add(tf.reduce_sum(conf_width))
            
        # Count total predictions 
        self.total_predictions.assign_add(tf.size(conf_width))
            
    def result(self):
        total = tf.cast(self.total_predictions, tf.float32)
        return self.conf_width_sum/total*self.scale
    
    def reset_state(self):
        self.conf_width_sum.assign(0)
        self.total_predictions.assign(0) 
