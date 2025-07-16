import logging
logging.basicConfig(level='ERROR')

import argparse
import numpy as np
from pprint import pprint
import sys
import os

import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from peft import PeftModel, PeftConfig
# For NanoGPT
gpt2_nano_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GPT2-Nano"))
sys.path.append(gpt2_nano_path)
from model import GPTConfig, GPT
import tiktoken
from bert_score import score
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
BASE_DIR = base_dir = os.path.dirname(os.path.abspath(__file__))
CKPT_DP_PATH = os.path.join(base_dir,'..', 'chkpt','nano','dp_sgd')
CKPT_PLAIN_PATH = os.path.join(base_dir,'..', 'chkpt','nano','plain')
OUTPUT_DP = os.path.join(base_dir, 'outputs_from_model','nano','output_dp.txt')
OUTPUT_PLAIN = os.path.join(base_dir, 'outputs_from_model','nano','output_plain.txt')
PROMPT_FILE = os.path.join(base_dir,"ref","prompts_pii.txt")
REF_FILE =  os.path.join(base_dir,"ref","references_pii.txt")


torch.set_default_dtype(DTYPE)
enc = tiktoken.get_encoding("gpt2")
def encode(s):
    return enc.encode(s, allowed_special={"<|endoftext|>"})

def decode(toks):
    return enc.decode(toks)

def save_ouputs(filepath, outputs):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(out + "\n" for out in outputs)
    print(f"Saved generations to {filepath}")

def load_prompts():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    with open(REF_FILE, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    return prompts, references

def load_nanoGPT_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model_args = ckpt['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)

    state_dict = ckpt['model']
    cleaned_state_dict = {
        k.replace("_orig_mod.", "")
        .replace("_module.",""): v
        for k, v in state_dict.items()
        }

    model.load_state_dict(cleaned_state_dict)
    model.to(DEVICE)
    return model


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    # parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    return parser.parse_args(argv)

def generate_outputs(ckpt_path,prompts):
    args = parse_arguments(sys.argv[1:])
    # number of tokens to generate
    seq_len = 256
    # sample from the top_k tokens output by the model
    top_k = 40

    nano_model = load_nanoGPT_model(ckpt_path)
    nano_model.eval()
    outputs = []

    for prompt in tqdm(prompts, desc="Generating"):

        prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)
        model_input = prompt_ids  # [1, T]

        with torch.no_grad():
            for _ in range(seq_len):
                logits, _ = nano_model(model_input)
                logits = logits[:, -1, :]  # [1, vocab_size]
                probs = torch.softmax(logits / 0.8, dim=-1)  # temperature = 0.8
                next_token = torch.multinomial(probs, num_samples=1)
                model_input = torch.cat((model_input, next_token), dim=1)

        generated_text = decode(model_input[0].tolist())
        outputs.append(generated_text)
    return outputs



if __name__ == '__main__':
    prompts, references = load_prompts()

    CKPT_DP_PATH = os.path.join(CKPT_DP_PATH,'checkpoint-60000.pt')
    CKPT_PLAIN_PATH = os.path.join(CKPT_PLAIN_PATH,'checkpoint-60000.pt')
    outputs_plain = generate_outputs(CKPT_PLAIN_PATH,prompts)
    outputs_dp = generate_outputs(CKPT_DP_PATH,prompts)

    save_ouputs(OUTPUT_PLAIN, outputs_plain)
    save_ouputs(OUTPUT_DP, outputs_dp)




