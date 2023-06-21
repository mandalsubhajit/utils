from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, models

## Step 1: use an existing language model
word_embedding_model = models.Transformer('D:\\work\\distilbert-base-uncased')

## Step 2: use a pool function over the token embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

## Join steps 1 and 2 using the modules argument
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

dataset_id = "trec"
dataset = load_dataset(dataset_id)

'''
DatasetDict({
    train: Dataset({
        features: ['text', 'coarse_label', 'fine_label'],
        num_rows: 5452
    })
    test: Dataset({
        features: ['text', 'coarse_label', 'fine_label'],
        num_rows: 500
    })
})
'''




train_examples = []
train_data = dataset['train']
# For agility we only 1/2 of our available data
n_examples = dataset['train'].num_rows // 10

for i in range(n_examples):
  example = train_data[i]
  train_examples.append(InputExample(texts=[example['text']], label=example['fine_label']))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.BatchAllTripletLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=10,
          output_path='D:\\work\\distilbert-base-uncased-trec')

# model.save('D:\\work\\distilbert-base-uncased-trec')
