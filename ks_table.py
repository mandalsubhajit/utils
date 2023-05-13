import pandas as pd
import numpy as np

def kscalc(target, prob, max_bins=10):
    data = pd.DataFrame({'target': target, 'prob': prob})
    data['target0'] = 1 - data['target']
    bins = np.unique(np.quantile(data['prob'], np.linspace(0, 1, max_bins+1)))
    data['bucket'] = pd.cut(data['prob'], bins)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()['prob']
    kstable['max_prob'] = grouped.max()['prob']
    kstable['events']   = grouped.sum()['target']
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data['target'].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data['target'].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1, len(bins))
    kstable.index.rename('Ranked_Partition', inplace=True)
    
    ks = max(kstable['KS'])
    kspartition = kstable.index[kstable['KS']==ks][0]
    
    return ks, kspartition, kstable


if __name__ == '__main__':
    prob = np.random.uniform(0, 1, 10000)
    target = [np.random.binomial(1, p**10/10) for p in prob]
    ks, kspartition, kstable = kscalc(target, prob)
    print(kstable)
