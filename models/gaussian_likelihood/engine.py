from operator import invert
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
    
class SamplingModel(tf.keras.Model):
    def __init__(self, architecture, llh_loss=None,
                 inference_samples_train=20,
                 inference_samples_test=20, 
                 inference_samples_predict=1000,
                 **kwargs):

        super(SamplingModel, self).__init__(**kwargs)
        self.architecture = architecture
        self.llh_loss = llh_loss
        self.inference_samples_train = inference_samples_train
        self.inference_samples_test = inference_samples_test
        self.inference_samples_predict = inference_samples_predict

    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        
        # Run in training mode to calculate loss
        with backprop.GradientTape() as tape:
            y_pred = self((x,y), training=True, 
                          samples=1, verbose=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        
        # Run in inference mode for other metrics
        if self.compiled_metrics._metrics is not None:
            y_pred_inference = self(x, training=False, 
                                    samples=self.inference_samples_train,
                                    verbose=True)
            self.compiled_metrics.update_state(y, y_pred_inference, 
                                               sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """The logic for one evaluation step.
        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.
        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Run in training mode to calculate loss
        y_pred = self((x,y), training=True,
                      samples=1)
        
        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, 
                           regularization_losses=self.losses)
        
        # Run in inference mode for other metrics
        if self.compiled_metrics._metrics is not None:
            y_pred_inference = self(x, training=False, 
                                    samples=self.inference_samples_test, 
                                    verbose=True)
            self.compiled_metrics.update_state(y, y_pred_inference, 
                                               sample_weight)
        
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        """The logic for one inference step.
        This method can be overridden to support custom inference logic.
        his method is called by `Model.make_predict_function`.
        his method should contain the mathematical logic for one step of inference.
        This typically includes the forward pass.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_predict_function`, which can also be overridden.
        Args:
            data: A nested structure of `Tensor`s.
        Returns:
            The result of one inference step, typically the output of calling the
            `Model` on data.
        """
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        
        return self(x, training=False, samples=self.inference_samples_predict)

    def build(self, input_shape):
        # Instantiate architecture if passed as subclass
        if isinstance(self.architecture, type):
            assert isinstance(input_shape, (list, tuple)), (
                '''Cannot infer network output shape from x data only.
                Please call model on (x,y) in training mode to build'''
                )
            if type(input_shape[-1]) == dict:
                self.output_dim = list(input_shape[-1].values())[0][-1]
            else:
                self.output_dim = input_shape[-1][-1]

            self.architecture = self.architecture(output_dim=self.output_dim)
            
        # Instantiate losses if passed as subclasses
        if isinstance(self.llh_loss, type):
            self.llh_loss = self.llh_loss(normalize=self.output_dim)

    def call(self, data, training=False, samples=1, verbose=False):
        samples = tf.cast(samples, tf.int32)
        
        if training:
            data_x, data_y = data
        else:
            data_x = data

        y_params, y_sampled = self.architecture([data_x, samples], training)
        
        # Add reconstruction loss and metric if using add_loss API
        if training and self.llh_loss:
            llh_loss = self.llh_loss(data_y, y_params)
            
            self.add_loss(llh_loss)
            self.add_metric(llh_loss, aggregation='mean', name='llh_loss')
        
        if verbose == True:
            result = {}
            if type(y_params) == dict:
                result.update(y_params)
            else:
                result['y_params'] = y_params
            
            if type(y_sampled) == dict:
                result.update(y_sampled)
            else:
                result['y_sampled'] = y_sampled

            return result
        
        else:
            return y_sampled
        

