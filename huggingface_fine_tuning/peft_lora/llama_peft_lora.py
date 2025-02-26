'''
https://medium.com/@rajatsharma_33357/fine-tuning-llama-using-lora-fb3f48a557d5
requirements:
  transformers
  bitsandbytes
  sentencepiece
  transformers[sentencepiece]
  accelerate
  bitsandbytes
  datasets
  peft
  trl
  py7zr # for the samsum data
'''

import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer



model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )
tokenizer = AutoTokenizer.from_pretrained(model_name)



data = load_dataset("samsum")
data_train, data_test, data_val = data["train"], data["test"], data["validation"]

print(data_train, data_test, data_val)

# example
data_train[0]
# output
#{'id': '13818513',
# 'dialogue': "Amanda: I baked  cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)",
# 'summary': 'Amanda baked cookies and will bring Jerry some tomorrow.'}



def generate_prompt(dialogue, summary=None, eos_token="</s>"):
  instruction = "Summarize the following:\n"
  input = f"{dialogue}\n"
  summary = f"Summary: {summary + ' ' + eos_token if summary else ''} "
  prompt = (" ").join([instruction, input, summary])
  return prompt

print(generate_prompt(data_train[0]["dialogue"], data_train[0]["summary"]))
# Summarize the following:
# Amanda: I baked  cookies. Do you want some?
# Jerry: Sure!
# Amanda: I'll bring you tomorrow :-)
#  Summary: Amanda baked cookies and will bring Jerry some tomorrow.



lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )



print(model)
# LlamaForCausalLM(
#  (model): LlamaModel(
#     (embed_tokens): Embedding(32000, 4096)
#     (layers): ModuleList(
#      (0-31): 32 x LlamaDecoderLayer(
#        (self_attn): LlamaAttention(
#          (q_proj): Linear8bitLt(in_features=4096, out_features=4096, bias=False)
#          (k_proj): Linear8bitLt(in_features=4096, out_features=4096, bias=False)
#          (v_proj): Linear8bitLt(in_features=4096, out_features=4096, bias=False)
#          (o_proj): Linear8bitLt(in_features=4096, out_features=4096, bias=False)
#          (rotary_emb): LlamaRotaryEmbedding()
#        )
#        (mlp): LlamaMLP(
#          (gate_proj): Linear8bitLt(in_features=4096, out_features=11008, bias=False)
#          (up_proj): Linear8bitLt(in_features=4096, out_features=11008, bias=False)
#          (down_proj): Linear8bitLt(in_features=11008, out_features=4096, bias=False)
#          (act_fn): SiLUActivation()
#        )
#        (input_layernorm): LlamaRMSNorm()
#        (post_attention_layernorm): LlamaRMSNorm()
#      )
#    )
#    (norm): LlamaRMSNorm()
#  )
#  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
#)



# this should be set for finutning and batched inference
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))



output_dir = "cp"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
per_device_eval_batch_size = 4
eval_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 5e-4
max_grad_norm = 0.3
max_steps = 50
warmup_ratio = 0.03
evaluation_strategy="steps"
lr_scheduler_type = "constant"

training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            evaluation_strategy=evaluation_strategy,
            save_steps=save_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            ddp_find_unused_parameters=False,
            eval_accumulation_steps=eval_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
        )



def formatting_func(prompt):
  output = []

  for d, s in zip(prompt["dialogue"], prompt["summary"]):
    op = generate_prompt(d, s)
    output.append(op)

  return output


trainer = SFTTrainer(
    model=model,
    train_dataset=data_train,
    eval_dataset=data_val,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args
)

# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.save_model(f"{output_dir}/final")

# Step Training Loss Validation Loss
# 10 1.848200 1.746341
# 20 1.688300 1.696681
# 30 1.654500 1.698127
# 40 1.579400 1.652010
# 50 1.492600 1.701877



# Inference: Load trained model from checkpoint
from peft import PeftModel

peft_model_id = "cp/checkpoint-40"
peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")

input_prompt = generate_prompt(data_train[50]["dialogue"])
input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")
with torch.cuda.amp.autocast():
    generation_output = peft_model.generate(
        input_ids=input_tokens,
        max_new_tokens=100,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature=0.3,
        repetition_penalty=1.15,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
      )
op = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(op)
# Summarize the following:
#  Pitt: Hey Teddy! Have you received my message?
# Teddy: No. An email?
# Pitt: No. On the FB messenger.
# Teddy: Let me check.
# Teddy: Yeah. Ta!
#  Summary:   Pitt sent a message to Teddy on Facebook Messenger, but he didn't receive it yet. 
