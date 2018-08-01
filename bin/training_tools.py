#!/bin/python

# custom functions for keras dl model training

from keras.callbacks import Callback
from keras import backend as K


class ChangeLR(Callback):
    def __init__(self, epochs, new_lrs, *args, **kwargs):
        self.epochs = epochs
        self.new_lrs = new_lrs
        super().__init__(*args, **kwargs)
    def on_epoch_begin(self, epoch, logs):
        if epoch in self.epochs:
            K.set_value(self.model.optimizer.lr, self.new_lrs[self.epochs.index(epoch)])
            print("\n\nLR CHANGE TO {}\n\n".format(self.new_lrs[self.epochs.index(epoch)]))

