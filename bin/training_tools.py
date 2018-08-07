#!/bin/python

# custom functions for keras dl model training
import change_path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import tf_memory_limit
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sequence
import ucscgenome
from tqdm import tqdm
import time
import datagen
import sequence
import helper
import viz_sequence
import pickle
from keras.models import Model, load_model
from keras.layers import Input, Dense, SpatialDropout1D, Conv1D, Lambda 
from keras.layers import Dropout, Activation, Concatenate, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from colour import Color
from livelossplot import PlotLossesKeras
import lasso
import training_tools
import shutil

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from keras.callbacks import Callback


class ChangeLR(Callback):
    def __init__(self, epochs, new_lrs, *args, **kwargs):
        self.epochs = epochs
        self.new_lrs = new_lrs
        super().__init__(*args, **kwargs)
    def on_epoch_begin(self, epoch, logs):
        if epoch in self.epochs:
            K.set_value(self.model.optimizer.lr, self.new_lrs[self.epochs.index(epoch)])
            print("\n\nLR CHANGE TO {}\n\n".format(self.new_lrs[self.epochs.index(epoch)]))

def train_model(out_dir, model_function, peaks, model_string, vpe=3, opt=Adam(lr=1e-04), num_epochs=10, batch_size=32, lr_sched=([0,5], [1e-03, 1e-05]), use_lasso=False, atac_only=False, seq_only=False, patience=1, args=[]):

    # facts about the data
    num_training_samples = len(peaks[(peaks.chr != 'chr8')])
    print('{} training samples'.format(num_training_samples))
    num_testing_samples = len(peaks[(peaks.chr == 'chr8') & (peaks.index%2 == 0)])
    print('{} testing samples'.format(num_testing_samples))
    num_validaiton_samples = len(peaks[(peaks.chr == 'chr8') & (peaks.index%2 == 1)])
    print('{} validation samples'.format(num_validaiton_samples))    

    # directory creation
    temp_path = os.path.join(out_dir, 'model_in_training')
    try:
        os.makedirs(temp_path)
    except FileExistsError as e:
        print('Warning, overwriting current model in training')
        shutil.rmtree(temp_path)
        os.makedirs(temp_path)
    # make a file system
    weights_path = os.path.join(temp_path, 'intermediate_weights')
    os.makedirs(weights_path)
    history_path = os.path.join(temp_path, 'history')
    os.makedirs(history_path)
    
    # get the model
    if use_lasso:
        model, grads = model_function(get_grads=True, atac_only=atac_only, seq_only=seq_only, batch_size=batch_size, args=args)
    else:
        model = model_function(atac_only=atac_only, seq_only=seq_only, batch_size=batch_size, args=args)
        
    # compile model
    if use_lasso:
        model.compile(loss=lasso.get_lwg(grads), metrics=['mean_absolute_error', lasso.get_gp(grads)], optimizer=opt)
    else:
        model.compile(loss='mean_absolute_error', metrics=['mean_absolute_error'], optimizer=opt)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience*vpe)
    filepath = os.path.join(weights_path, 'weights-{epoch:02d}-{val_loss:.3f}.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # training params
    train_gen = datagen.batch_gen(peaks, mode='train', atac_only=atac_only, seq_only=seq_only, batch_size=batch_size)
    val_gen = datagen.batch_gen(peaks, mode='val', atac_only=atac_only, seq_only=seq_only, batch_size=batch_size)
    spe = num_training_samples//(batch_size*vpe)
    spve = num_validaiton_samples//(batch_size*(vpe//5+1))
    callbacks=[early_stop, checkpoint, PlotLossesKeras(), ChangeLR(lr_sched[0], lr_sched[1])]
    
    # train the model
    losses = model.fit_generator(train_gen, steps_per_epoch=spe, epochs=num_epochs*vpe, 
                                 callbacks=callbacks, validation_data=val_gen, validation_steps=spve, verbose=2)

    # final save
    model.save(os.path.join(temp_path, 'final_model.h5'))
    val_hist = losses.history['val_loss']
    train_hist = losses.history['loss']
    losses=dict()
    losses['val_loss'] = val_hist
    losses['loss'] = train_hist
    pickle.dump(losses, open(os.path.join(history_path, 'history.pk'), 'wb'))    
        
    # final directory change
    timestr = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, timestr + '_' + model_string)
    os.rename(temp_path, out_path)
    print('Model moved to ' + out_path)

    return out_path
