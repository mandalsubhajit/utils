#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:12:54 2018

@author: subhajit
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_partial_dependence(clf, X, features, n_cols=3, figsize=(10, 10)):
    X_temp = X.copy().values
    
    fig = plt.figure(figsize=figsize)
    nrows=int(np.ceil(len(features)/float(n_cols)))
    ncols=min(n_cols, len(features))
    axs = []
    for i, f_id in enumerate(features):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        
        x_scan = np.linspace(np.percentile(X_temp[:, f_id], 0.1), np.percentile(X_temp[:, f_id], 99.5), 10)
        y_partial = []
        
        for point in x_scan:
            X_temp[:, f_id] = point
            y_partial.append(np.average(clf.predict(X_temp)))
        
        y_partial = np.array(y_partial)
        
        # Plot partial dependence
        ax.plot(x_scan, y_partial, '-', color = 'green', linewidth = 1)
        ax.set_xlim(min(x_scan)-0.1*(max(x_scan)-min(x_scan)), max(x_scan)+0.1*(max(x_scan)-min(x_scan)))
        ax.set_ylim(min(y_partial)-0.1*(max(y_partial)-min(y_partial)), max(y_partial)+0.1*(max(y_partial)-min(y_partial)))
        ax.set_xlabel(X.columns[f_id])
    axs.append(ax)
    fig.subplots_adjust(bottom=0.15, top=0.7, left=0.1, right=0.95, wspace=0.4,
                        hspace=0.3)
    return fig, axs

#fig, axs = plot_partial_dependence(clf, X_train, range(10), figsize=(10, 15))