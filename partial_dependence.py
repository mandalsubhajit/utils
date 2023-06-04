#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:12:54 2018

@author: subhajit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_partial_dependence(clf, X, features, n_cols=3, figsize=(10, 10)):    
    fig = plt.figure(figsize=figsize)
    nrows=int(np.ceil(len(features)/float(n_cols)))
    ncols=min(n_cols, len(features))
    axs = []
    for i, f_id in enumerate(features):
        X_temp = X.values
        ax = fig.add_subplot(nrows, ncols, i + 1)
        
        x_scan = np.linspace(np.percentile(X_temp[:, f_id], 0.1), np.percentile(X_temp[:, f_id], 99.5), 10)
        y_partial = []
        
        for point in x_scan:
            X_temp[:, f_id] = point
            y_partial.append(np.average(clf.predict(pd.DataFrame(data=X_temp, columns=X.columns))))
        
        y_partial = np.array(y_partial)
        
        # Plot partial dependence
        ax.plot(x_scan, y_partial, '-', color = 'green', linewidth = 1)
        ax.set_xlim(min(x_scan)-0.1*(max(x_scan)-min(x_scan)), max(x_scan)+0.1*(max(x_scan)-min(x_scan)))
        ax.set_ylim(min(y_partial)-0.1*(max(y_partial)-min(y_partial)), max(y_partial)+0.1*(max(y_partial)-min(y_partial)))
        ax.set_xlabel(X.columns[f_id])
    axs.append(ax)
    fig.subplots_adjust(bottom=0.15, top=0.7, left=0.1, right=0.95, wspace=0.4,
                        hspace=0.3)
    fig.tight_layout()
    return fig, axs

#fig, axs = plot_partial_dependence(clf, X_train, range(10), figsize=(10, 15))

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    from xgboost import XGBClassifier
    # load data
    data = load_breast_cancer(return_X_y=False)
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='response')
    clf = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.01, objective='binary:logistic')
    # fit model
    clf.fit(X, y)
    # generate partial dependence plots
    for i in range(0, X.shape[1], 9):
        feature_ids = range(i, min(i+9, X.shape[1]))
        fig, axs = plot_partial_dependence(clf, X, feature_ids, figsize=(15, 10))
        fig.savefig(f'./partial_dependence_{min(feature_ids)}_{max(feature_ids)}.png')
