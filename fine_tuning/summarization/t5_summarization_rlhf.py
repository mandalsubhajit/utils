from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
# Define and parse arguments.

config = PPOConfig(
    model_name='D:\work\\t5-small',
    learning_rate=1.41e-5,
    log_with=None,
    mini_batch_size=16,
    batch_size=256,
    gradient_accumulation_steps=1,
    early_stopping=False,
    target_kl=0.1,
)


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
#sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

dataset = load_dataset("billsum", split="ca_test")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# input_size = LengthSampler(2, 8)

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["text"])
    #sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

dataset = dataset.map(tokenize, batched=False)
dataset = dataset.rename_columns({'text': 'query'})
dataset.set_format(type="torch")


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
reward_tokenizer = AutoTokenizer.from_pretrained('D:\work\distilbert-base-uncased-rm\checkpoint-75')
reward_model = AutoModelForSequenceClassification.from_pretrained('D:\work\distilbert-base-uncased-rm\checkpoint-75', num_labels=1)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
#tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
#device = ppo_trainer.accelerator.device
#if ppo_trainer.accelerator.num_processes == 1:
#    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
#sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
# output_min_length = 4
# output_max_length = 16
#output_length_sampler = LengthSampler(output_min_length, output_max_length)


print('Device: ', ppo_trainer.accelerator.device)
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, length_sampler=None, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    reward_inputs = reward_tokenizer(batch["response"], batch["query"], max_length=384, truncation="only_second", return_tensors="pt")
    with torch.no_grad():
        outputs = reward_model(**reward_inputs)[0]
    rewards = [torch.Tensor(output) for output in outputs]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    
    model.save_pretrained("D:\work\distilbert-base-uncased-rlhf", push_to_hub=False)
    tokenizer.save_pretrained("D:\work\distilbert-base-uncased-rlhf", push_to_hub=False)
