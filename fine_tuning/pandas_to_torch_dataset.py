import pandas as pd
import numpy as np
import datasets
import torch

# LOAD PANDAS DATAFRAME 
datapath = './bbc-text.csv'
df = pd.read_csv(datapath)
df.head()

'''
        category                                               text
0           tech  tv future in the hands of viewers with home th...
1       business  worldcom boss  left books alone  former worldc...
2          sport  tigers wary of farrell  gamble  leicester say ...
3          sport  yeading face newcastle in fa cup premiership s...
4  entertainment  ocean s twelve raids box office ocean s twelve...
'''

# USING CUSTOM DATASET CLASS
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        labels = {'business':0,
                  'entertainment':1,
                  'sport':2,
                  'tech':3,
                  'politics':4
                  }
        self.labels = [labels[label] for label in df['category']]
        self.texts = df['text']

    def __len__(self):
        # returns the length of the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # returns item at index=idx
        return {'text': self.texts[idx], 'label': self.labels[idx]}

df_dataset = Dataset(df)



# USING from_pandas
df_train, df_val = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df))])
df_dataset_dict = datasets.DatasetDict({'train': datasets.Dataset.from_pandas(df_train), 
                                        'test': datasets.Dataset.from_pandas(df_val)})
df_dataset_dict = df_dataset_dict.remove_columns(["__index_level_0__"])
