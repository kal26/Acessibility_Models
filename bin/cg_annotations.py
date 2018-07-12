#!/bin/python

import pandas as pd
import numpy as np
import h5py

lowercase_map = np.arange(256, dtype=np.uint8)
lowercase_map[np.fromstring("ACGT", dtype=np.uint8)] = np.fromstring("acgt", dtype=np.uint8)

#load in the DNA
DNA = h5py.File('/home/kal/K27act_models/reference/hg19.h5', 'r')
dnamap = {}
for k in DNA.keys():
    print("loading", k)
    dnamap[k] = DNA[k][...]
DNA.close()
DNA = dnamap

def gc_frac(row):
    seq = lowercase_map[DNA[row.chr][row.start:row.end]]
    return ((seq == ord('g')).sum() + (seq == ord('c')).sum()) / len(seq)

def cpg_frac(row):
    seq = lowercase_map[DNA[row.chr][row.start:row.end]]
    gs = (seq == ord('g'))
    cs = (seq == ord('c'))
    count = (cs[:-1] & gs[1:]).sum()
    return count / len(seq)
