import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

# 1. LOAD DATA AND TRAIN TEST SPLIT
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


# 2. LOAD BERT MODEL AND TOKENIZER
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("D:\work\distilbert-base-uncased", model_max_length=512)
model = AutoModel.from_pretrained("D:\work\distilbert-base-uncased").to(device)


# 3. EXTRACT BERT [CLS] FEATURE FROM TEXT
def get_cls_embedding(batch: List[str], model, tokenizer):
    tokenized_batch = tokenizer(batch, padding = True, truncation = True, return_tensors="pt")
    tokenized_batch = {k:torch.tensor(v).to(device) for k,v in tokenized_batch.items()}
    with torch.no_grad():
      hidden_batch = model(**tokenized_batch) #dim : [batch_size(nr_sentences), tokens, emb_dim]
    # free gpu memory by moving tensors to cpu
    cls_batch = hidden_batch.last_hidden_state[:,0,:].cpu().numpy()
    
    return cls_batch

train_text = df_train["text"].values.tolist()
val_text = df_val["text"].values.tolist()

# process in batches to avoid gpu memory overflow
batch_size = 10

x_train = np.vstack([get_cls_embedding(train_text[i:i+batch_size], model, tokenizer) for i in tqdm(range(0, len(train_text), batch_size))])
y_train = df_train["category"]

x_val = np.vstack([get_cls_embedding(val_text[i:i+batch_size], model, tokenizer) for i in tqdm(range(0, len(val_text), batch_size))])
y_val = df_val["category"]

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)


# 4. CLASSIFICATION (TRAINING AND TESTING)
# random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf.score(x_val,y_val) 


# neural network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout

num_features = x_train.shape[1]
num_classes = 5

network = Sequential()
network.add(Input(shape=(num_features,)))
network.add(Dropout(0.5))
network.add(Dense(num_features//2, activation='relu'))
network.add(Dropout(0.5))
network.add(Dense(num_classes, activation='softmax'))

network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(network.summary())

label2id = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }
y_tf_train = tf.keras.utils.to_categorical(y_train.map(label2id), num_classes=num_classes)
y_tf_val = tf.keras.utils.to_categorical(y_val.map(label2id), num_classes=num_classes)
network.fit(x_train, y_tf_train, validation_data=(x_val, y_tf_val), verbose=1, epochs=5)

network.predict(get_cls_embedding('India won the world cup', model, tokenizer))
