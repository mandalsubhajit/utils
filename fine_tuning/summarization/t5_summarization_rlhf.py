from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
# from trl.core import LengthSampler


tqdm.pandas()

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
# Define and parse arguments.

config = PPOConfig(
    model_name='D:\work\\t5-small-billsum',
    learning_rate=1.41e-5,
    log_with=None,
    mini_batch_size=16,
    batch_size=256,
    gradient_accumulation_steps=1,
    early_stopping=False,
    target_kl=0.1,
    ppo_epochs=4,
)


# We then load and prepare the data.

dataset = load_dataset("billsum", split="ca_test")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

def tokenize(sample):
    prefix = "summarize: "
    sample["input_ids"] = tokenizer.encode(prefix + sample["text"])
    sample["query"] = prefix + sample["text"]
    return sample

dataset = dataset.map(tokenize, batched=False)
dataset = dataset.remove_columns(["text"])
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
for epoch in tqdm(range(config.ppo_epochs)):
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from t5
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, length_sampler=None, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # Compute rewards
        reward_inputs = reward_tokenizer(batch["response"], batch["query"], max_length=384, truncation="only_second", return_tensors="pt")
        with torch.no_grad():
            outputs = reward_model(**reward_inputs)[0]
        rewards = [torch.Tensor(output) for output in outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    
    model.save_pretrained("D:\work\distilbert-base-uncased-rlhf", push_to_hub=False)
    tokenizer.save_pretrained("D:\work\distilbert-base-uncased-rlhf", push_to_hub=False)
