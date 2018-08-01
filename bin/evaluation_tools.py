#!/bin/python
import datagen
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# evaluation and comparison tools for ml models

def compare_models(peaks, graph_list=['hexbin', 'bars', 'pr'], fbaselines={}, kbaselines={}, logkbaselines={}, 
                   fmls={}, kmls={}, logkmls={}, batch_size=32):
    fpreds={}
    kpreds={}
    logkpreds={}

    # predict baseline models
    for key in fbaselines:
        try:
            fpreds[key] = fbaselines[key].predict(peaks[['gc_frac','cpg_frac', 'atac']])
        except ValueError:
            fpreds[key] = fbaselines[key].predict(peaks[['gc_frac','cpg_frac']])
        kpreds[key] = (2**(fpreds[key]))*(peaks['atac']+1)-1
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
    for key in kbaselines:
        try:
           kfpreds[key] = kbaselines[key].predict(peaks[['gc_frac','cpg_frac', 'atac']])
        except ValueError:
            kpreds[key] = kbaselines[key].predict(peaks[['gc_frac','cpg_frac']])
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
        fpreds[key] = np.log2((kpreds[key]+1)/(peaks['atac']+1))
    for key in logkbaselines:
        try:
           temppreds = logkbaselines[key].predict(peaks[['gc_frac','cpg_frac', 'atac']])
        except ValueError:
            temppreds = logkbaselines[key].predict(peaks[['gc_frac','cpg_frac']])
        kpreds[key] = np.e ** temppreds
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
        fpreds[key] = np.log2((kpreds[key]+1)/(peaks['atac']+1))

    # predict ml models
    for key in fmls:
        try:
            fpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks), steps=len(peaks)//batch_size).flatten()
        except ValueError:
            try:
                fpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks, seq_only=True), steps=len(peaks)//batch_size).flatten()
            except ValueError:
                fpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks, atac_only=True), steps=len(peaks)//batch_size).flatten()
        kpreds[key] = (2**(fpreds[key]))*(peaks['atac']+1)-1
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
    for key in kmls:
        try:
            kpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks), steps=len(peaks)//batch_size).flatten()
        except ValueError:
            try:
                kpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks, seq_only=True), steps=len(peaks)//batch_size).flatten()
            except ValueError:
                kpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks, atac_only=True), steps=len(peaks)//batch_size).flatten()
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
        fpreds[key] = np.log2((kpreds[key]+1)/(peaks['atac']+1))
    for key in logkmls:
        try:
            logkpreds[key] = logkmls[key].predict_generator(datagen.simple_batch(peaks), steps=len(peaks)//batch_size).flatten()
        except ValueError:
            try:
                logkpreds[key] = logkmls[key].predict_generator(datagen.simple_batch(peaks, seq_only=True), steps=len(peaks)//batch_size).flatten()
            except ValueError:
                logkpreds[key] = logkmls[key].predict_generator(datagen.simple_batch(peaks, atac_only=True), steps=len(peaks)//batch_size).flatten()
        kpreds[key] = np.e ** temppreds
        fpreds[key] = np.log2((kpreds[key]+1)/(peaks['atac']+1))

# combine into a dataframe
predframe = pd.DataFrame(data={'fold_change':fpreds, 'k27act':kpreds, 'logk27act':logkpreds}
peaks['logk27act'] = np.log2(peaks['k27act'].clip(0) +1)

# make hexbin plots
if 'hexbin' in graph_list:
    for name, row in predframe.iterrows():
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        f.suptitle('Predicted vs Actual Values by {} Model'.format((name))
        ax1.set_xlabel('Log of Normalized K27act')
        ax1.set_ylabel('Model Prediciton for K27act')
        ax2.set_xlabel('Log Fold Change K27act over ATAC')
        ax2.set_ylabel('Model Prediciton for Fold Change')
        ax1.hexbin(peaks['logk27act'], row['logk27act'], bins='log', extent=(0, 8, 0, 8))
        ax2.hexbin(peaks['fold_change'], row['fold_change'], bins='log', extent=(-6, 6, -6, 6))
        f.subplots_adjust(left=0.15, top=0.95)
        plt.show()

# color stuff
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_iter = iter(colors)
for name, row in predframe.iterrows():
    predframe.at[name, 'color'] = next(color_iter)

# make mse plots
if 'bars' in graph_list:
    f, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    f.suptitle('Error for Model Predictions')
    axes[0][0].set_ylabel('K27 Acetylation')
    axes[1][0].set_ylabel('Fold Change')
    axes[0][0].set_title('Mean Squared Error')
    axes[0][1].set_title('Mean Absolute Error')

    for name, row in predframe.iterrows():
        predframe.at[name, 'fmse'] = np.mean((peaks['fold_change'] - row['fold_change']) **2)
        predframe.at[name, 'fmae'] = np.mean(abs(peaks['fold_change'] - row['fold_change']))
        predframe.at[name, 'kmse'] = np.mean((peaks['k27act'] - row['k27act']) **2)
        predframe.at[name, 'kmae'] = np.mean(abs(peaks['k27act'] - row['k27act']))

    barlist = axes[0][0].bar(range(len(predframe['kmse'])), predframe['kmse'])
    for i in range(len(barlist)):
        barlist[i].set_color(colors[i])
    axes[0][0].legend(barlist, c.keys(), loc=4) 
    barlist = axes[1][0].bar(range(len(predframe['fmse'])), predframe['fmse'])
    for i in range(len(barlist)):
        barlist[i].set_color(colors[i])
    barlist = axes[0][1].bar(range(len(predframe['kmae'])), predframe['kmae'])
    for i in range(len(barlist)):
        barlist[i].set_color(colors[i])
    barlist = axes[1][1].bar(range(len(predframe['fmae'])), predframe['fmae'])
    for i in range(len(barlist)):
        barlist[i].set_color(colors[i])

if 'pr' in graph_list:
    
