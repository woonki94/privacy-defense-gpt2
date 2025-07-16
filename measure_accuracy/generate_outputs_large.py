import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from bert_score import score
from tqdm import tqdm
import os
# === CONFIG ===
MODEL_NAME = "gpt2-large"
BASE_DIR = base_dir = os.path.dirname(os.path.abspath(__file__))
CKPT_DP_PATH = os.path.join(base_dir,'..', 'chkpt','large','dp-sgd')
CKPT_PLAIN_PATH = os.path.join(base_dir,'..', 'chkpt','large','plain')
OUTPUT_DP = os.path.join(base_dir, 'outputs_from_model','large','output_dp.txt')
OUTPUT_PLAIN = os.path.join(base_dir, 'outputs_from_model','large','output_plain.txt')
PROMPT_FILE = os.path.join(base_dir,"ref","prompts_pii.txt")
REF_FILE =  os.path.join(base_dir,"ref","references_pii.txt")
MAX_NEW_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === LOAD PROMPTS ===
def load_prompts():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    with open(REF_FILE, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    return prompts, references

# === LOAD TOKENIZER + MODEL ===
def load_model(ckpt_path):
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.to(DEVICE).eval()
    return model

# === GENERATE ===
def generate_outputs(model, prompts):
    outputs = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        outputs.append(decoded)
    return outputs

# === SAVE OUTPUT ===
def save_ouputs(filepath, outputs):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(out + "\n" for out in outputs)
    print(f"Saved generations to {filepath}")



if __name__ == '__main__':
    CKPT_DP_PATH = os.path.join(CKPT_DP_PATH,'checkpoint-245000')
    CKPT_PLAIN_PATH = os.path.join(CKPT_PLAIN_PATH,'checkpoint-245000')
    gpt_large_dp = load_model(CKPT_DP_PATH)
    gpt_large_plain = load_model (CKPT_PLAIN_PATH)
    prompts, references = load_prompts();
    outputs_dp = generate_outputs(gpt_large_dp, prompts)
    outputs_plain = generate_outputs(gpt_large_plain, prompts)

    save_ouputs(OUTPUT_DP, outputs_dp)
    save_ouputs(OUTPUT_PLAIN, outputs_plain)

