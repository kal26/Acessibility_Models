#!/bin/python

import numpy as np
import pandas as pd
import sys
sys.path.append('/home/kal/TF_models/bin/')
import sequence
import h5py
import ucscgenome

#load in the DNA
genome19 = ucscgenome.Genome('/home/kal/.ucscgenome/hg19.2bit')

# load in ATAC data
atac_path = '/home/kal/K27act_models/GM_data/ATAC/atac_average.hdf5'
atac_data = h5py.File(atac_path, 'r')

# make a generator
def datagen(peaks, num_groups=5, mode='train', split_column='score', shuffle=True, log=False, atac_only=False, seq_only=False):
    assert len(peaks) > 1
    
    # stratify the data
    quant_idx=dict()
    for i in range(num_groups):
        # get quintile values
        min_fold, max_fold = peaks.quantile([i/num_groups, (i+1)/num_groups])[split_column]
        # split text, treain, val
        if mode == 'test':
            subset_peaks = peaks[(peaks.chr == 'chr8') & (peaks.index%2 == 0)]
        elif mode =='val':
            subset_peaks = peaks[(peaks.chr == 'chr8') & (peaks.index%2 == 1)]
        else:
            subset_peaks = peaks[(peaks.chr != 'chr8')]
            
        quant_idx[i] = subset_peaks[(subset_peaks[split_column] > min_fold) & (subset_peaks[split_column] < max_fold)].index.values  
      
    # yield the values
    while True:
        # shuffle the samples
        if shuffle:
            for key in quant_idx:
                np.random.shuffle(quant_idx[key])     
        # groups the smaples into a dataframe
        cutoff=min([len(quant_idx[k]) for k in quant_idx])
        d = pd.DataFrame([zip(quant_idx[k][:cutoff]) for k in quant_idx]).transpose()
        # get each sample
        for i, row in d.iterrows():
            for j in row:
                idx=j[0]
                yield get_sample(peaks.iloc[idx], log=log, atac_only=atac_only, seq_only=seq_only)
                
def get_sample(row, genome=None, log=False, verb=False, atac_only=False, seq_only=False):
    if genome==None:
        genome=genome19
    try:
        nucs = sequence.encode_to_onehot(row['nucs'])
    except KeyError:
        if verb:
            print('passed sample without nucs')
        nucs = sequence.encode_to_onehot(genome[row['chr']][row['start']:row['end']])
    if not seq_only:
        atac_counts = atac_data[row['chr']][row['start']:row['end']]
        if not atac_only:
            # we will train on [atac, one_hot DNA]
            try:
                out = np.insert(nucs.astype(np.float32), 0, atac_counts, axis=1)
            except ValueError as e:
                print(row)
                raise(e)
        else:
            out = np.expand_dims(atac_counts, axis=2)
    else:
        out = nucs
    # return relevant information
    if log:
        return out, np.log2(row['score'] + 1)
    else:
        return out, row['score']

def batch_gen(peaks, num_groups=5, batch_size=32, mode='train', shuffle=True, log=False, atac_only=False, seq_only=False):
    d = datagen(peaks, num_groups=num_groups, mode=mode, shuffle=shuffle, log=log, atac_only=atac_only, seq_only=seq_only)
    test_data, test_score = next(d)
    while True:
        X = np.empty((batch_size, test_data.shape[0], test_data.shape[1]))
        y = np.empty((batch_size))
        for i in range(batch_size):
            inputs, score = next(d)
            X[i]=inputs
            y[i]=score
        yield X, y


def simple_gen(peaks, atac_only=False, seq_only=False):
    for index, row in peaks.iterrows():
        yield datagen.get_sample(row, atac_only=atac_only, seq_only=seq_only)
        
def simple_batch(peaks, batch_size=32. atac_only=False, seq_only=False):
    d = simple_gen(peaks, atac_only=atac_only, seq_only=seq_only)
    while True:
        if atac_only:
            X = np.empty((batch_size, 1024, 1))
        elif seq_only:
            X = np.empty((batch_size, 1024, 4))
        else:
            X = np.empty((batch_size, 1024, 5))
        y = np.empty((batch_size))
        for i in range(batch_size):
            inputs, score = next(d)
            X[i]=inputs
            y[i]=score
        yield X, y
