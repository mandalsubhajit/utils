#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 00:42:40 2018

@author: subhajit
"""

from sklearn.tree import _tree
import numpy as np

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]
    print('def tree({}):'.format(', '.join(feature_names)))

    def recurse(node, depth):
        indent = '  ' * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print('{}if {} <= {}:'.format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print('{}else:  # if {} > {}'.format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print('{}return {}'.format(indent, np.argmax(tree_.value[node][0])))

    recurse(0, 1)

# Usage:
#tree_to_code(clf.estimators_[9], X_train.columns)