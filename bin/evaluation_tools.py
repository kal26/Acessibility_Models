#!/bin/python
import datagen
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import pr_gain
import os

# evaluation and comparison tools for ml models

def compare_models(peaks, graph_list=['hexbin', 'bars', 'pr', 'roc'], fbaselines={}, kbaselines={}, logkbaselines={}, 
                   fmls={}, kmls={}, logkmls={}, batch_size=32, save_dir={}):
    fpreds={}
    kpreds={}
    logkpreds={}

    # predict baseline models
    for key in fbaselines:
        try:
            fpreds[key] = fbaselines[key].predict(peaks[['gc_frac','cpg_frac', 'atac']])
        except ValueError as e:
            print(e) 
            fpreds[key] = fbaselines[key].predict(peaks[['gc_frac','cpg_frac']])
        kpreds[key] = (2**(fpreds[key]))*(peaks['atac']+1)-1
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
    for key in kbaselines:
        try:
           kpreds[key] = kbaselines[key].predict(peaks[['gc_frac','cpg_frac', 'atac']])
        except ValueError as e:
            print(e)
            kpreds[key] = kbaselines[key].predict(peaks[['gc_frac','cpg_frac']])
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
        fpreds[key] = np.log2((kpreds[key]+1)/(peaks['atac']+1))
    for key in logkbaselines:
        try:
           temppreds = logkbaselines[key].predict(peaks[['gc_frac','cpg_frac', 'atac']])
        except ValueError as e:
            print(e)
            temppreds = logkbaselines[key].predict(peaks[['gc_frac','cpg_frac']])
        kpreds[key] = np.e ** temppreds
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
        fpreds[key] = np.log2((kpreds[key]+1)/(peaks['atac']+1))

    # predict ml models
    for key in fmls:
        try:
            fpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks), steps=len(peaks)//batch_size).flatten()
        except ValueError as e:
            print(e) 
            try:
                fpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks, seq_only=True), steps=len(peaks)//batch_size).flatten()
            except ValueError as e:
                print(e) 
                fpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks, atac_only=True), steps=len(peaks)//batch_size).flatten()
        kpreds[key] = (2**(fpreds[key]))*(peaks['atac']+1)-1
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
    for key in kmls:
        try:
            kpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks), steps=len(peaks)//batch_size).flatten()
        except ValueError as e:
            print(e) 
            try:
                kpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks, seq_only=True), steps=len(peaks)//batch_size).flatten()
            except ValueError as e:
                print(e) 
                kpreds[key] = fmls[key].predict_generator(datagen.simple_batch(peaks, atac_only=True), steps=len(peaks)//batch_size).flatten()
        logkpreds[key] = np.log2(kpreds[key].clip(0) + 1)
        fpreds[key] = np.log2((kpreds[key]+1)/(peaks['atac']+1))
    for key in logkmls:
        try:
            logkpreds[key] = logkmls[key].predict_generator(datagen.simple_batch(peaks), steps=len(peaks)//batch_size).flatten()
        except ValueError as e:
            print(e) 
            try:
                logkpreds[key] = logkmls[key].predict_generator(datagen.simple_batch(peaks, seq_only=True), steps=len(peaks)//batch_size).flatten()
            except ValueError as e:
                print(e) 
                logkpreds[key] = logkmls[key].predict_generator(datagen.simple_batch(peaks, atac_only=True), steps=len(peaks)//batch_size).flatten()
        kpreds[key] = np.e ** logkpreds[key] 
        fpreds[key] = np.log2((kpreds[key]+1)/(peaks['atac']+1))

    # combine into a dataframe
    predframe = pd.DataFrame(data={'fold_change':fpreds, 'k27act':kpreds, 'logk27act':logkpreds})
    peaks['logk27act'] = np.log2(peaks['k27act'].clip(0) +1)

    # make hexbin plots
    if 'hexbin' in graph_list:
        for name, row in predframe.iterrows():
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            f.suptitle('Predicted vs Actual Values by {} Model'.format(name))
            ax1.set_xlabel('Log of Normalized K27act')
            ax1.set_ylabel('Model Prediciton for K27act')
            ax2.set_xlabel('Log Fold Change K27act over ATAC')
            ax2.set_ylabel('Model Prediciton for Fold Change')
            ax1.hexbin(peaks['logk27act'], row['logk27act'], bins='log', extent=(0, 8, 0, 8))
            ax2.hexbin(peaks['fold_change'], row['fold_change'], bins='log', extent=(-6, 6, -6, 6))
        #    f.subplots_adjust(left=0.15, top=0.95)
            plt.show()
            try:
                f.savefig(os.path.join(save_dir[name], 'hexbin.png'))
            except KeyError as e:
                print(e)

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
             predframe.at[name, 'kmse'] = np.mean((peaks['logk27act'] - row['logk27act']) **2)
             predframe.at[name, 'kmae'] = np.mean(abs(peaks['logk27act'] - row['logk27act']))

        barlist = axes[0][0].bar(range(len(predframe['kmse'])), predframe['kmse'])
        for i in range(len(barlist)):
            barlist[i].set_color(colors[i])
        axes[0][0].legend(barlist, predframe.index, loc=4) 
        barlist = axes[1][0].bar(range(len(predframe['fmse'])), predframe['fmse'])
        for i in range(len(barlist)):
             barlist[i].set_color(colors[i])
        barlist = axes[0][1].bar(range(len(predframe['kmae'])), predframe['kmae'])
        for i in range(len(barlist)):
            barlist[i].set_color(colors[i])
        barlist = axes[1][1].bar(range(len(predframe['fmae'])), predframe['fmae'])
        for i in range(len(barlist)):
            barlist[i].set_color(colors[i])
        plt.show()
        for name in predframe.index:
            try:
                f.savefig(os.path.join(save_dir[name], 'errors.png'))
            except KeyError as e:
                print(e)


    if 'pr' in graph_list:
        prop_pos=sum(peaks['fold_change']>0)/len(peaks)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        f.suptitle('Fold-Change Direction Prediction')
        ax1.set_title('Precision-Recall Curve')
        ax2.set_title('Precision-Recall Gain Curve')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax2.set_xlabel('Recall Gain')
        ax2.set_ylabel('Precision Gain')
        for name, row in predframe.iterrows():
            p, r, t = precision_recall_curve(peaks['fold_change']>0, row['fold_change'])
            ax1.plot(r, p, label=name, color=row['color'])
            pgain, rgain = pr_gain.get_gain(p, r, prop_pos)
            ax2.plot(rgain, pgain, label=name, color=row['color'])    
        ax1.legend()
        for a in ax1, ax2:
            a.set_ylim([0.0, 1.05])
            a.set_xlim([0.0, 1.0])
        plt.show()
        for name in predframe.index:
            try:
                f.savefig(os.path.join(save_dir[name], 'prcurve.png'))
            except KeyError as e:
                print(e)


    if 'roc' in graph_list:
        for name, row in predframe.iterrows():
            fpr, tpr, t = roc_curve(peaks['fold_change']>0, row['fold_change'])
            plt.plot(fpr, tpr, label=name, color=row['color'])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend()
        plt.title('Fold-Change Direction ROC Curve')
        plt.show()
        for name in predframe.index:
            try:
                f.savefig(os.path.join(save_dir[name], 'roc.png'))
            except KeyError as e:
                print(e)

    return predframe
