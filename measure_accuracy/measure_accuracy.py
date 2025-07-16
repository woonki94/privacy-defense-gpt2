import os
import argparse
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import numpy as np

# === Load files ===
def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# === BLEU ===
def compute_bleu(hyps, refs):
    hyps_tok = [h.split() for h in hyps]
    refs_tok = [[r.split()] for r in refs]
    return corpus_bleu(refs_tok, hyps_tok, smoothing_function=SmoothingFunction().method2)

# === ROUGE ===
def compute_rouge(hyps, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for h, r in zip(hyps, refs):
        s = scorer.score(r, h)
        for k in scores:
            scores[k].append(s[k].fmeasure)
    return {k: np.mean(v) for k, v in scores.items()}

# === BERTScore ===
def compute_bertscore(hyps, refs):
    P, R, F1 = bertscore(hyps, refs, lang="en", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["nano", "large"], required=True,
                        help="Model size: 'nano' or 'large'")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    output_base = os.path.join(base_dir, "outputs_from_model", args.size)
    dp_file = os.path.join(output_base, "output_dp.txt")
    plain_file = os.path.join(output_base, "output_plain.txt")
    ref_file = os.path.join(base_dir, "ref", "references.txt")

    references = load_lines(ref_file)
    plain_outputs = load_lines(plain_file)
    dp_outputs = load_lines(dp_file)

    min_len = min(len(references), len(plain_outputs), len(dp_outputs))
    references = references[:min_len]
    plain_outputs = plain_outputs[:min_len]
    dp_outputs = dp_outputs[:min_len]

    bleu_plain = compute_bleu(plain_outputs, references)
    bleu_dp = compute_bleu(dp_outputs, references)

    rouge_plain = compute_rouge(plain_outputs, references)
    rouge_dp = compute_rouge(dp_outputs, references)

    bert_plain = compute_bertscore(plain_outputs, references)
    bert_dp = compute_bertscore(dp_outputs, references)

    # === PRINT ===
    print("\n=== BLEU & ROUGE Comparison ===")
    print(f"BLEU (Plain):   {bleu_plain * 100:.2f}")
    print(f"BLEU (DP):      {bleu_dp * 100:.2f}")

    print("\nROUGE-F1 Scores:")
    for metric in ["rouge1", "rouge2", "rougeL"]:
        print(f"{metric.upper()} (Plain): {rouge_plain[metric]:.4f}")
        print(f"{metric.upper()} (DP):    {rouge_dp[metric]:.4f}")

    print("\nBERT-F1 Scores:")
    for metric in ["precision", "recall", "f1"]:
        print(f"{metric.upper()} (Plain): {bert_plain[metric]:.4f}")
        print(f"{metric.upper()} (DP):    {bert_dp[metric]:.4f}")

