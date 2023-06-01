from transformers import AutoTokenizer
# from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
import pandas as pd
import datasets

print(torch.cuda.current_device())

# DATA PREPARATION
datapath = './bbc-text.csv'
df = pd.read_csv(datapath)
# df.columns = ['category', 'text']
label2id = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }
id2label = {0:'business',
          1:'entertainment',
          2:'sport',
          3:'tech',
          4:'politics'
          }
df['label'] = df['category'].map(label2id)
df.head()


df_train, df_val = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df))])
df_dataset_dict = datasets.DatasetDict({'train': datasets.Dataset.from_pandas(df_train), 
                                        'test': datasets.Dataset.from_pandas(df_val)})
df_dataset_dict = df_dataset_dict.remove_columns(["__index_level_0__"])




tokenizer = AutoTokenizer.from_pretrained("D:\work\distilbert-base-uncased", model_max_length=512)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

tokenized_dataset = df_dataset_dict.map(preprocess_function, batched=True).remove_columns(['category', 'text'])




# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# MODEL TRAINING
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)




model = AutoModelForSequenceClassification.from_pretrained(
    "D:\work\distilbert-base-uncased", num_labels=5, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="D:\work\distilbert-base-uncased-imdb",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print(training_args.device)
trainer.train()



# MODEL INFERENCE
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("D:\work\distilbert-base-uncased-imdb\checkpoint-224")
model = AutoModelForSequenceClassification.from_pretrained(
    "D:\work\distilbert-base-uncased-imdb\checkpoint-224"
)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
classifier(text)
