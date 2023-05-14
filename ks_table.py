import pandas as pd
import numpy as np
from typing import List

def find_rank_order_break(sorted_array: List[float]):
    for i in range(1, len(sorted_array)):
        if sorted_array[i] > sorted_array[i-1]:
            yield i
            return
        if i == len(sorted_array) - 1:
            yield None

def kscalc(target: List[int], prob: List[float], max_bins: int=10):
    assert np.array_equal(np.unique(target), [0, 1]), "target variable should include *only* and *both* 0, 1 values"
    data = pd.DataFrame({'target': target, 'prob': prob})
    data['target0'] = 1 - data['target']
    
    #Ranked Partitioning
    bins = np.unique(np.quantile(data['prob'], np.linspace(0, 1, max_bins+1)))
    data['bucket'] = pd.cut(data['prob'], bins, include_lowest=True)
    
    #Calculation
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()['prob']
    kstable['max_prob'] = grouped.max()['prob']
    kstable['total']   = grouped.count()['target']
    kstable['events']   = grouped.sum()['target']
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_pct'] = (kstable.events / data['target'].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_pct'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventpct']=(kstable.events / data['target'].sum()).cumsum()
    kstable['cum_noneventpct']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(np.abs(kstable['cum_eventpct']-kstable['cum_noneventpct']), 3) * 100
    
    #Formating
    kstable['cum_eventpct']= kstable['cum_eventpct'].apply('{0:.2%}'.format)
    kstable['cum_noneventpct']= kstable['cum_noneventpct'].apply('{0:.2%}'.format)
    kstable.index = range(1, len(bins))
    kstable.index.rename('Ranked_Partition', inplace=True)
    
    #Final KS
    ks = max(kstable['KS'])
    kspartition = kstable.index[kstable['KS']==ks][0]
    
    #Check for rank order break
    event_prob = (kstable['events']/(kstable['total'])).values
    rob = list(find_rank_order_break(event_prob))[0]
    robpartition = kstable.index[rob] if rob else None
    
    
    return ks, kspartition, robpartition, kstable


if __name__ == '__main__':
    prob = np.random.uniform(0, 1, 10000)
    target = [np.random.binomial(1, p**10/10) for p in prob]
    ks, kspartition, robpartition, kstable = kscalc(target, prob)
    pd.set_option('display.max_columns', 10)
    print(kstable)
