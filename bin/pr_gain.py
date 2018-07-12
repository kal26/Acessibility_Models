#!/bin/python
import numpy as np

def pr_aoc(precision, recall):
    pos_p, pos_r = get_pos(precision, recall, 0, 1)
    #extrapolate a point for recall zero
    if len(pos_p)>0:
        pos_r = np.append(pos_r, 0)
        pos_p = np.append(pos_p, pos_p[-1])    
    return -trapz(pos_p, pos_r)
    
def get_pos(listy, listz, min_value, max_value):
    try:
        newy=list()
        newz=list()
        for y, z in zip(listy, listz):
            if min([y, z]) > min_value and max([y, z]) < max_value:
                newy.append(y)
                newz.append(z)
        return newy, newz 
    except ValueError:
        print('The sequence was empty')
        return [], [] 
    
def get_between(r, p):
        """listy=r, listz=p"""
        start_index = np.argmax([(rv<1) and (pv>0) for rv, pv in zip(r, p)])
        stop_index = np.argmax([(pv>=1) or (rv<=0) for rv, pv in zip(r, p)])
        return start_index, stop_index

def get_gain(p, r, prop_pos):
    start_index, stop_index = get_between(r, p) 
    pgain = [(x-prop_pos)/((1-prop_pos)*x) for x in p[start_index:stop_index]]
    rgain = [(x-prop_pos)/((1-prop_pos)*x) for x in r[start_index:stop_index]]
    return pgain, rgain
