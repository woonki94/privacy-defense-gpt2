"""
Extraction script for evaluating memorization using different models
with or without Differential Privacy (DP).

Supports:
- Sample generation using GPT2-Large or NanoGPT (DP or non-DP)
- Perplexity scoring across all 4 models
- Option to use empty prompts or real prompts (e.g. from Common Crawl)
"""

import logging
logging.basicConfig(level="ERROR")

import argparse
import sys
import torch
import numpy as np
from pprint import pprint
import zlib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os


from GPT2_Nano.model import GPTConfig, GPT
import tiktoken

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.set_default_dtype(dtype)

enc = tiktoken.get_encoding("gpt2")

def encode(s):
    return enc.encode(s, allowed_special={"<|endoftext|>"})

def decode(toks):
    return enc.decode(toks)


def load_nano_model(path, dp=False):
    ckpt = torch.load(path, map_location=device if not dp else 'cpu')
    config = GPTConfig(**ckpt['model_args'])
    model = GPT(config)
    state_dict = ckpt['model']
    if dp:
        state_dict = {k.replace("_orig_mod._module.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_gpt2_model(path):
    config = PeftConfig.from_pretrained(path)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, path, is_trainable=False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def calculate_perplexity(sentence, model, tokenizer=None, is_nano=False):
    """
    Calculates perplexity for a given sentence using the specified model.
    Returns None if input is too short or perplexity is infinite or unstable.
    """
    try:
        if is_nano:
            input_ids = torch.tensor(encode(sentence)).unsqueeze(0).to(device)
            if input_ids.size(1) < 2:
                return None
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            with torch.no_grad():
                _, loss = model(inputs, targets)
        else:
            input_ids = tokenizer.encode(sentence)
            if len(input_ids) < 2:
                return None
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

        ppl = torch.exp(loss)
        return ppl.cpu().item()

    except Exception as e:
        print(f"[WARN] Perplexity calculation failed: {e}")
        return None


def filter_none_perplexities(samples, *score_lists, fallback=1e9):
    """
    Filters out samples (and corresponding score values) where any score is None.
    Returns indices of valid samples.
    """
    for i in range(len(samples)):
        for j in range(len(score_lists)):
            if score_lists[j][i] is None or not np.isfinite(score_lists[j][i]):
                score_lists[j][i] = fallback

    return list(range(len(samples)))


def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=100, file=None):
    """
    print the 'n' best samples according to the given 'metric'
    """
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            header = f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}"
        else:
            header = f"{i+1}: {name1}={scores1[idx]:.3f}, score={metric[idx]:.3f}"

        print(header.strip(), end="\n\n")
        pprint(samples[idx])
        print("\n")

        if file:
            file.write(header + "\n\n")
            pprint(samples[idx], stream=file)
            file.write("\n\n")


def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]

    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs) - 1):
        start, end = start_idxs[i], start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start + 7]:
            count_eng += 1
            all_eng += " ".join(lines[start + 10:end])

    return all_eng


def sample_from_commoncrawl(cc_text, tokenizer, batch_size, input_len):

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_mask = [], []
    words = cc_text.split()

    for _ in range(batch_size * 2):
        idx = np.random.randint(0, len(words) - input_len - 1)
        prompt = " ".join(words[idx:idx + input_len])
        enc = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)

        if len(enc['input_ids'][0]) == input_len:
            input_ids.append(enc['input_ids'][0])
            attention_mask.append(enc['attention_mask'][0])

    input_ids = input_ids[:batch_size]
    attention_mask = attention_mask[:batch_size]
    prompts = tokenizer.batch_decode(torch.stack(input_ids), skip_special_tokens=True)

    return {
        'input_ids': torch.stack(input_ids).to(device),
        'attention_mask': torch.stack(attention_mask).to(device)
    }, prompts


def generate_samples(model, tokenizer, batch_size, seq_len, top_k, is_nano=False, internet_sampling=False, cc_data=None):

    if internet_sampling and cc_data and not is_nano:
        inputs, prompts = sample_from_commoncrawl(cc_data, tokenizer, batch_size, input_len=10)
    else:
        prompts = ["<|endoftext|>"] * batch_size
        if not is_nano:
            inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)

    if is_nano:
        encoded = [torch.tensor(encode(p), dtype=torch.long, device=device)[None, ...] for p in prompts]
        input_batch = torch.cat(encoded, dim=0)
        output_batch = model.generate(
            idx=input_batch,
            max_new_tokens=seq_len,
            temperature=1.0,
            top_k=top_k,
            repetition_penalty=1.2
        )
        return [decode(out[input_batch.shape[1]:].tolist()) for out in output_batch]
    else:   # gpt2
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=seq_len + inputs['input_ids'].shape[1],
            do_sample=True,
            top_k=top_k,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.2
        )
        return tokenizer.batch_decode(output, skip_special_tokens=True)


def main():
    print(f"[Device] {device}")

    # Use Common Crawl
    cc_data = None
    cc_tag = '_cc' if args.internet_sampling else ''
    if args.internet_sampling and args.wet_file:
        print("Loading Common Crawl data...")
        cc_data = parse_commoncrawl(args.wet_file)

    # Load models
    print("Loading evaluation models...")
    nano = load_nano_model("chkpt/nano/plain/checkpoint-60000.pt")
    nano_dp = load_nano_model("chkpt/nano/dp_sgd/checkpoint-60000.pt", dp=True)
    gpt2, gpt2_tok = load_gpt2_model("chkpt/large/plain/checkpoint-245000")
    gpt2_dp, gpt2_dp_tok = load_gpt2_model("chkpt/large/dp_sgd/checkpoint-245000")

    # Select generation model, tokenizer, and is_nano flag based on user input
    if args.gen_model == 'nano':
        gen_model, gen_tokenizer, is_nano = nano, None, True
    elif args.gen_model == 'nano_dp':
        gen_model, gen_tokenizer, is_nano = nano_dp, None, True
    elif args.gen_model == 'gpt2':
        gen_model, gen_tokenizer, is_nano = gpt2, gpt2_tok, False
    else:
        gen_model, gen_tokenizer, is_nano = gpt2_dp, gpt2_dp_tok, False

    samples = []
    scores = {"nano": [], "nano_dp": [], "gpt2": [], "gpt2_dp": [], "Lower": [], "zlib": []}

    # Generate samples in batches and compute perplexity scores using all models
    # Also calculate zlib compression size and lower-case perplexity
    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            texts = generate_samples(gen_model, gen_tokenizer, args.batch_size, args.seq_len, args.top_k, is_nano, args.internet_sampling, cc_data)
            for text in texts:
                samples.append(text)
                scores["nano"].append(calculate_perplexity(text, nano, is_nano=True))
                scores["nano_dp"].append(calculate_perplexity(text, nano_dp, is_nano=True))
                scores["gpt2"].append(calculate_perplexity(text, gpt2, gpt2_tok))
                scores["gpt2_dp"].append(calculate_perplexity(text, gpt2_dp, gpt2_dp_tok))
                scores["Lower"].append(calculate_perplexity(text.lower(), gen_model, gen_tokenizer, is_nano))
                scores["zlib"].append(len(zlib.compress(text.encode('utf-8'))))
            pbar.update(args.batch_size)

    # Apply filter after all scores are collected
    # Convert all score lists to NumPy arrays for computation
    valid_idx = filter_none_perplexities(samples, scores["nano"], scores["nano_dp"], scores["gpt2"], scores["gpt2_dp"])
    samples = [samples[i] for i in valid_idx]
    for key in scores:
        scores[key] = [scores[key][i] for i in valid_idx]
        scores[key] = np.asarray(scores[key])

    # Record info in a text file
    output_file = f"results_{args.gen_model}{cc_tag}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        metric_key = args.gen_model
        base_metric = np.log(scores[metric_key])

        # Sort by base metric's perplexity
        metric = -base_metric
        print(f"======== top {args.num_print} samples by {metric_key} perplexity: ========")
        f.write(f"======== top {args.num_print} samples by {metric_key} perplexity: ========" + "\n")
        print_best(metric, samples, metric_key, scores[metric_key], n=args.num_print, file=f)
        print()
        print()

        # Sort by ratio of log perplexities of base and other cases
        for other_key in scores:
            if other_key != metric_key:
                print(f"======== top {args.num_print} samples by ratio of {other_key} to {metric_key} perplexity: ========")
                f.write(f"======== top {args.num_print} samples by ratio of {other_key} to {metric_key} perplexity: ========" + "\n")

                ratio_metric = np.log(scores[other_key]) / base_metric
                print_best(ratio_metric, samples, metric_key, scores[metric_key], other_key, scores[other_key], n=args.num_print, file=f)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-model', type=str, choices=['nano', 'nano_dp', 'gpt2', 'gpt2_dp'], default='gpt2_dp')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--N', type=int, default=100000)
    parser.add_argument('--num-print', type=int, default=100)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--internet-sampling', action='store_true', help='Use real prompts instead of empty prompt')
    parser.add_argument('--wet-file', type=str, default=None, help='Common Crawl WET file for real prompts')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
