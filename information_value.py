import pandas as pd
import numpy as np
import pandas.core.algorithms as algos

def num_iv(Y, X, xname, max_bins=20):
 df = pd.DataFrame({'X': X, 'Y': Y})
 justmiss = df[['X', 'Y']][df.X.isnull()]
 notmiss = df[['X', 'Y']][df.X.notnull()]
 
 bins = algos.quantile(notmiss.X, np.linspace(0, 1, max_bins+1))
 d1 = pd.DataFrame({'X': notmiss.X, 'Y': notmiss.Y, 'Bucket': pd.cut(notmiss.X, np.unique(bins), include_lowest=True)})
 d2 = d1.groupby('Bucket', as_index=True)
 
 d3 = pd.DataFrame({}, index=[])
 d3['MIN_VALUE'] = d2.X.min()
 d3['MAX_VALUE'] = d2.X.max()
 d3['GROUP'] = ''
 d3['COUNT'] = d2.Y.count()
 d3['EVENT'] = d2.Y.sum()
 d3['NONEVENT'] = d2.Y.count() - d2.Y.sum()
 
 if justmiss.shape[0] > 0:
  d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
  d4['MAX_VALUE'] = np.nan
  d4['GROUP'] = ''
  d4['COUNT'] = justmiss.Y.count()
  d4['EVENT'] = justmiss.Y.sum()
  d4['NONEVENT'] = justmiss.Y.count() - justmiss.Y.sum()
  d3 = d3.append(d4, ignore_index=True)
 
 d3['EVENT_DIST'] = d3.EVENT/d3.EVENT.sum()
 d3['NON_EVENT_DIST'] = d3.NONEVENT/d3.NONEVENT.sum()
 d3['WOE'] = np.log(d3.EVENT_DIST/d3.NON_EVENT_DIST)
 d3['IV'] = (d3.EVENT_DIST-d3.NON_EVENT_DIST)*d3['WOE']
 d3['VAR_NAME'] = xname
 d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'GROUP', 'COUNT', 'EVENT', 'EVENT_DIST', 'NONEVENT', 'NON_EVENT_DIST', 'WOE', 'IV']]
 d3.IV = d3.IV.replace([np.inf, -np.inf], 0)
 
 return d3

def char_iv(Y, X, xname):
 df = pd.DataFrame({'X': X, 'Y': Y})
 justmiss = df[['X', 'Y']][df.X.isnull()]
 notmiss = df[['X', 'Y']][df.X.notnull()]
 
 d2 = notmiss.groupby('X', as_index=True)
 
 d3 = pd.DataFrame({}, index=[])
 d3['COUNT'] = d2.Y.count()
 d3['MIN_VALUE'] = np.nan
 d3['MAX_VALUE'] = np.nan
 d3['GROUP'] = d3.index
 d3['EVENT'] = d2.Y.sum()
 d3['NONEVENT'] = d2.Y.count() - d2.Y.sum()
 
 if justmiss.shape[0] > 0:
  d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
  d4['MAX_VALUE'] = np.nan
  d4['GROUP'] = ''
  d4['COUNT'] = justmiss.Y.count()
  d4['EVENT'] = justmiss.Y.sum()
  d4['NONEVENT'] = justmiss.Y.count() - justmiss.Y.sum()
  d3 = d3.append(d4, ignore_index=True)
 
 d3['EVENT_DIST'] = d3.EVENT/d3.EVENT.sum()
 d3['NON_EVENT_DIST'] = d3.NONEVENT/d3.NONEVENT.sum()
 d3['WOE'] = np.log(d3.EVENT_DIST/d3.NON_EVENT_DIST)
 d3['IV'] = (d3.EVENT_DIST-d3.NON_EVENT_DIST)*d3['WOE']
 d3['VAR_NAME'] = xname
 d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'GROUP', 'COUNT', 'EVENT', 'EVENT_DIST', 'NONEVENT', 'NON_EVENT_DIST', 'WOE', 'IV']]
 d3.IV = d3.IV.replace([np.inf, -np.inf], 0)
 
 return d3

def info_value(df, target, max_bins=20):
 var_list = df.columns
 count = -1
 
 for var in var_list:
  print('Processing (%d/%d)  %s                    ' % (count+1, len(var_list), var), end='\r')
  try:
   if np.issubdtype(df[var], np.number) and df[var].nunique() > 5:
    var_iv = num_iv(target, df[var], var, max_bins)
    var_iv['VAR_NAME'] = var
    count = count + 1
   else:
    var_iv = char_iv(target, df[var], var)
    var_iv['VAR_NAME'] = var
    count = count + 1
  except:
   pass
  
  if count == 0:
   iv_df = var_iv
  else:
   iv_df = iv_df.append(var_iv, ignore_index=True)
 
 print()
 iv = pd.DataFrame(iv_df.groupby('VAR_NAME').IV.sum())
 return iv_df, iv

if __name__ == '__main__':
 from sklearn.datasets import load_breast_cancer
 data = load_breast_cancer(return_X_y=False)
 X = pd.DataFrame(data.data, columns=data.feature_names)
 y = pd.Series(data.target, name='response')
 iv_calc, IV = info_value(X, y, max_bins=5)