#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 00:45:09 2018

@author: subhajit
"""

from sklearn.tree import _tree
import itertools
import operator
import numpy as np
import pandas as pd

def tree_to_rules(tree, feature_names, simplify=False):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]
    
    rules = []
    numobs = np.sum(tree.tree_.value[0])
    
    def simplify_rules(l):
        simple_rule = []
        l_left = [r for r in l if r[1]=='<=']
        l_right = [r for r in l if r[1]=='>']
        it_left = itertools.groupby(sorted(l_left), operator.itemgetter(0))
        for key, subiter in it_left:
           simple_rule.append((key, '<=', min(item[2] for item in subiter)))
        it_right = itertools.groupby(sorted(l_right), operator.itemgetter(0))
        for key, subiter in it_right:
           simple_rule.append((key, '>', max(item[2] for item in subiter)))
        simple_rule = sorted(simple_rule, key=lambda t: (t[0], t[1]))
        return simple_rule
    
    def recurse(node, depth, prev_rule=[]):
        if simplify:
            curr_rule = simplify_rules(prev_rule)
        else:
            curr_rule = prev_rule
        #print(curr_rule)
        rules.append((', '.join([' '.join([t[0], t[1], str(t[2])]) for t in curr_rule]),
                     np.sum(tree.tree_.value[node])/numobs,
                     tree.tree_.value[node][0][1]/np.sum(tree.tree_.value[node])))
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            curr_rule_l = curr_rule + [(name, '<=', threshold)]
            recurse(tree_.children_left[node], depth + 1, curr_rule_l)
            curr_rule_r = curr_rule + [(name, '>', threshold)]
            recurse(tree_.children_right[node], depth + 1, curr_rule_r)
    
    recurse(0, 1)
    rules = pd.DataFrame(rules)
    rules.columns = ['decision_rule', 'support', 'confidence']
        
    return rules

# Usage:
# 1. Decision Tree:
# rules = tree_to_rules(clf, X_train.columns, simplify=True)
# 2. Random Forest:
# rules = pd.concat([tree_to_rules(est, X_train.columns, simplify=False) for est in clf.estimators_], ignore_index=True)