import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm

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

np.random.seed(112)
df_train, df_val = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df))])

print(len(df_train),len(df_val))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("D:\work\distilbert-base-uncased", model_max_length=512)
model = AutoModel.from_pretrained("D:\work\distilbert-base-uncased").to(device)



train_text = df_train["text"].values.tolist()
val_text = df_val["text"].values.tolist()

def get_cls_embedding(batch):
    tokenized_batch = tokenizer(batch, padding = True, truncation = True, return_tensors="pt")
    tokenized_batch = {k:torch.tensor(v).to(device) for k,v in tokenized_batch.items()}
    with torch.no_grad():
      hidden_batch = model(**tokenized_batch) #dim : [batch_size(nr_sentences), tokens, emb_dim]
    cls_batch = hidden_batch.last_hidden_state[:,0,:].cpu().numpy()
    
    return cls_batch



batch_size = 10

x_train = np.vstack([get_cls_embedding(train_text[i:i+batch_size]) for i in tqdm(range(0, len(train_text), batch_size))])
y_train = df_train["category"]

x_val = np.vstack([get_cls_embedding(val_text[i:i+batch_size]) for i in tqdm(range(0, len(val_text), batch_size))])
y_val = df_val["category"]

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf.score(x_val,y_val) 