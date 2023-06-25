from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import evaluation

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



# prepare training data: train_examples.append(InputExample(texts=[example['text']], label=example['fine_label']))
train_examples = []
train_data = dataset['train']
# For agility we only 1/2 of our available data
n_examples = dataset['train'].num_rows // 10

for i in range(n_examples):
  example = train_data[i]
  train_examples.append(InputExample(texts=[example['text']], label=example['fine_label']))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.BatchAllTripletLoss(model=model)

##### EVALUATION: OPTIONAL #####
sentences1 = ['The man was having food', 'Solve this puzzle', 'You want your model to evaluate on']
sentences2 = ['He was eating a piece of bread', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
scores = [1, 0, 0]

evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
################################

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=10,
          evaluator=evaluator,
          evaluation_steps=50,
          output_path='D:\\work\\distilbert-base-uncased-trec')

# model.save('D:\\work\\distilbert-base-uncased-trec')

# INFERENCE
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

model = SentenceTransformer('D:\\work\\distilbert-base-uncased-trec')
e1 = model.encode('first sentence')
e2 = model.encode('second sentence')

print('Cosine Similarity: ', dot(e1, e2)/(norm(e1)*norm(e2)))
