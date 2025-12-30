import random

import pandas as pd
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

random.seed(42)

print("prepping data")
raw_data = pd.read_csv('data/syntheticdata.csv')
raw_data = raw_data.sample(frac=1).reset_index(drop=True)
num_eval_samples = 200
eval_data = raw_data[:num_eval_samples]
train_data = raw_data[num_eval_samples:]

def format_data(data):
    formatted_data = []
    # for en_text, tt_text in data:
    for en_text, tt_text in zip(data['english_text'].tolist(), data['tatar_text'].tolist()):
        formatted_data.append({"conversations": [
            {"role": "system", "content": "You are a helpful assistant that translates english text to tatar."},
            {"role": "user", "content": f"Translate this english text to tatar (no explanation, only output the translation in tatar): {en_text}"},
            {"role": "assistant", "content": tt_text},
        ]})
    return formatted_data

formatted_train_data = format_data(train_data)
formatted_eval_data = format_data(eval_data)

train_dataset = Dataset.from_list(formatted_train_data)
eval_dataset = Dataset.from_list(formatted_eval_data)

print(f"training samples: {len(train_dataset)}")
print(f"evaluation samples: {len(eval_dataset)}")
print("example training data point:")
print(train_dataset[0])
print("data preparation complete.\n")

print("loading model and setting up LoRA")
max_seq_length = 2048
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, remove_columns=eval_dataset.column_names)
print("exmples after applying chat template:")
print(train_dataset[0])
print(eval_dataset[0])

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=0,
    use_rslora=False,
    loftq_config=None,
)
print("Model setup complete.\n")

print("defining training arguments")
training_args = SFTConfig(
    output_dir="./llama_en_tt_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=20,
    optim="adamw_8bit",
    weight_decay=0.001,
    lr_scheduler_type="linear",
    seed=42,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    dataset_text_field = "text",
)
print("training arguments defined.\n")

print("setting up and starting SFTTrainer")
trainer = SFTTrainer(
    model=model,
    tokenize=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_num_proc=2,
    packing=False,
    max_seq_length=max_seq_length,
    args=training_args,
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer.train()

print("training complete.")

print("saving model")
model.save_pretrained("final_llama_en_tt_model")
tokenizer.save_pretrained("final_llama_en_tt_model")
print("Model saved to final_llama_en_tt_model")

