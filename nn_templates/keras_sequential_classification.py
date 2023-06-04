# 1. Data Preparation
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = load_breast_cancer(as_frame=True)
feature_df = dataset['data']
target = dataset['target']
num_classes = target.nunique()

# scaling data for faster training convergence
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(feature_df)

X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.2, stratify=target)

if num_classes > 2:
    # need to transform target variable into one-hot representation
    label2id = {'business':0,
              'entertainment':1,
              'sport':2,
              'tech':3,
              'politics':4
              }
    y_tf_train = tf.keras.utils.to_categorical(y_train.map(label2id), num_classes=num_classes)
    y_tf_val = tf.keras.utils.to_categorical(y_val.map(label2id), num_classes=num_classes)
else:
    y_tf_train, y_tf_val = y_train, y_val


# 2. Model Definition
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras import metrics

num_features = feature_df.shape[1]

# model architecture
model = Sequential()
model.add(Input(shape=(num_features,)))
model.add(Dropout(0.5))
model.add(Dense(num_features//2, activation='relu'))
model.add(Dropout(0.5))
if num_classes > 2:
    model.add(Dense(num_classes, activation='softmax'))
else:
    model.add(Dense(1, activation='sigmoid'))

# model loss and metrics
if num_classes > 2:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.AUC()])
print(model.summary())


# 3. Model Fitting
model.fit(X_train, y_tf_train, validation_data=(X_val, y_tf_val), verbose=1, epochs=100)
