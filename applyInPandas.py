import pyspark.sql.functions as F
import pyspark.sql.types as T
import torch

# broadcast the models to worker nodes for distributed processing
sc = spark.sparkContext
broadcasted_clf = sc.broadcast(clf)
broadcasted_model = sc.broadcast(model)
broadcasted_tokenizer = sc.broadcast(tokenizer)

def get_cls_embedding(batch, model, tokenizer):
    tokenized_batch = tokenizer(batch, padding = True, truncation = True, return_tensors="pt")
    tokenized_batch = {k:torch.tensor(v).to(device) for k,v in tokenized_batch.items()}
    with torch.no_grad():
      hidden_batch = model(**tokenized_batch) #dim : [batch_size(nr_sentences), tokens, emb_dim]
    # .cpu().numpy() --> detach values from gpu
    cls_batch = hidden_batch.last_hidden_state[:,0,:].cpu().numpy()
    
    return cls_batch

def get_score(pdf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = broadcasted_clf.value
    model = broadcasted_model.value.to(device)
    tokenizer = broadcasted_tokenizer.value
    
    text = pdf['text']
    # process in batches to avoid gpu memory overflow
    batch_size = 10
    x_val = np.vstack([get_cls_embedding(text[i:i+batch_size], model, tokenizer) for i in range(0, len(text), batch_size)])
    score = clf.predict_proba(x_val)[:,1]
    
    return pdf.assign(score=score)


df = df.repartitionByRange(8, 'date')
df = df \
    .groupBy(F.spark_partition_id().alias('_pid')) \
    .applyInPandas(get_score, T.StructType(df.schema.fields + [T.StructField('score', T.FloatType(), True)]))
