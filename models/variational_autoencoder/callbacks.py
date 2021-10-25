import tensorflow.keras as keras

class BetaSchedule(keras.callbacks.Callback):
    def __init__(self, total_epochs=50):
        super(BetaSchedule, self).__init__()
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, *args):
        if epoch < self.total_epochs:
            weight = epoch/self.total_epochs
        else:
            weight = 1
        
        self.model.beta.assign(weight)