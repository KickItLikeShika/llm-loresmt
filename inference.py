from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import pandas as pd
from tqdm import tqdm


df = pd.read_csv('test.csv')
model_path = "llama_en_tt_model/checkpoint-63"
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)
FastLanguageModel.for_inference(model) # enable native 2x faster inference

submission_data = []
for index, row in tqdm(df.iterrows()):
    text_id = row['id']
    source_en_text = row['source_en']
    print(f"Processing ID {text_id}: '{source_en_text}'")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that translates english text to tatar."},
        {"role": "user", "content": f"Translate this english text to tatar (no explanation, only output the translation in tatar): {source_en_text}"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # must add for generation
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True, temperature=0, do_sample=False)
    translation = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()
    print(f"Translation: {translation}")
    submission_data.append({
        "id": text_id,
        "submission": translation
    })

submission_data = pd.DataFrame(submission_data)
submission_data.to_csv('submission.csv', index=False)