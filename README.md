# Data Extraction and Defense in LLMs

# 📚 Table of Contents

1. [Overview](#-Overview)
2. [Resources](#-resources)
3. [What We Do](#-what-we-do)
4. [Experimental Setup](#-experimental-setup)
5. [Key Findings](#-key-findings)
6. [Environment Setup](#-environment-setup)
7. [Step 1: Dataset Preparation](#-step-1-dataset-preparation)

   * [1.1 Download and Place Raw Data](#11-download-and-place-raw-data)
   * [1.2 Model-Specific Dataset Use](#12-model-specific-dataset-use)
   * [1.3 Clean and Merge Datasets](#-step-13-clean-and-merge-datasets)
8. [Step 2: Train Models](#-step-2-train-models)

   * [2.1 GPT-2 Large (LoRA Fine-Tune)](#-21-gpt-2-large-lora-fine-tune)
   * [2.2 GPT-2 Nano (Train from Scratch)](#-22-gpt-2-nano-train-from-scratch)
   * [Use Pretrained Checkpoints](#-optional-use-pretrained-checkpoints)
9. [Step 3: Evaluate Accuracy & Leakage](#-step-3-evaluate-accuracy--leakage)

   * [3.1 Generate Prompt–Reference Pairs](#31-generate-promptreference-pairs)
   * [3.2 Generate Model Outputs](#32-generate-outputs-from-models)
   * [3.3 Evaluate Outputs](#33-evaluate-outputs)
10. [Step 4: Run Extraction and Categorization](#step-4-run-extraction-and-categorization)

    * [4.1 Run Extraction Script](#run-the-script)
    * [4.2 Run Categorization Script](#run-categorization)
    * [4.3 Limitations of Categorization](#limitations-of-categorization)


# 🧠 Overview

This project investigates **memorization and privacy risks** in large language models (LLMs), focusing on the leakage of **personally identifiable information (PII)**. We replicate and extend **data extraction attacks**, and implement a scalable, memory-efficient **differential privacy (DP)** defense using **DP-SGD**.

We apply our pipeline to two model setups:

* 🧠 **NanoGPT** (trained from scratch)
* 🛠️ **GPT-2 Large** (fine-tuned with LoRA adapters)
---

## 📄 Resources

* 🎞️ **[Presentation Slides (PDF)](./report/Initial_Presentation.pdf)**
* 📘 **[Final Report (PDF)](./report/Data_Extraction_and_Defense.pdf)**


---


## 🔍 What We Do

1. **Train and fine-tune LLMs** on mixed datasets (e.g., Enron emails, synthetic PII, and public corpora).
2. **Evaluate privacy risk** using **Carlini-style black-box extraction** to detect memorization.
3. **Defend against memorization** using a custom **DP-SGD** pipeline featuring:

   * ✅ **Ghost Clipping** – Efficient gradient clipping for linear (LoRA) layers.
   * ⚡ **Fast Gradient Clipping** – Optimized clipping for non-linear layers.
   * 🧪 **Virtual Batching** – Train with large batch sizes under GPU constraints.

---

## 🧪 Experimental Setup

| Model           | Training Method | Privacy Level | Dataset                        |
| --------------- | --------------- | ------------- | ------------------------------ |
| **NanoGPT**     | From scratch    | DP / Non-DP   | Public + Enron + Synthetic PII |
| **GPT-2 Large** | LoRA Fine-tune  | DP / Non-DP   | Enron + Synthetic PII          |

**Evaluation Metrics**:

* 🔢 **Utility**: BLEU, ROUGE, BERTScore
* 🔍 **Privacy Leakage**: Perplexity filtering + manual review of PII
* 📉 **Memorization**: Compression-based filters, Carlini-style extraction

---

## 📊 Key Results

Our experiments demonstrate the effectiveness of DP-SGD in mitigating privacy risks while maintaining model utility:

* 🔐 **Privacy Protection**:
  Differentially Private SGD (DP-SGD) reduces **PII leakage by over 90%** in both NanoGPT and GPT-2 Large.

* ✅ **Model Utility Preserved**:
  Despite the privacy enhancements, generation quality suffers only **minor degradation**—with a maximum drop of **≤4 BERTScore points** compared to non-private models.

* 🧠 **Memory Optimization**:
  The introduction of **ghost clipping** reduces GPU memory usage during training (GPT-2 Large) from **>40 GB to \~30 GB**, enabling more scalable private training.

---

### 📉 Perplexity vs. Entropy of Extracted Samples

<p align="center"> <img src="https://github.com/user-attachments/assets/add64790-f507-483f-98ad-6fe44e82d189" alt="Perplexity vs. Entropy" width="600"/> </p>

> **Observation**: Higher perplexity tends to align with higher entropy—indicating lower confidence and less structured outputs.
> **Impact**: DP-SGD reduces both perplexity and entropy, producing outputs less likely to contain memorized, identifiable content.

---

### 🔍 Extraction by Sensitive Category

<p align="center"> <img src="https://github.com/user-attachments/assets/e08973d5-6785-47b5-a24c-e760f20fed26" alt="Sensitive Categories Barplot" width="400"/> </p>

> **Analysis**: The DP-trained models generate significantly fewer samples containing sensitive content such as names, email addresses, URLs, and license text.
> **Conclusion**: DP-SGD is effective at mitigating memorization of identifiable or proprietary content.


---

# ⚙️ End-to-End Usage Guide

---

## 💻 Environment Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Hardware Requirements**

* Recommended: **NVIDIA H100 GPU**
* Minimum: **40 GB GPU memory** for GPT-2 Large + DP-SGD

---

## 📁 Step 1: Dataset Preparation

### 1.1 Download and Place Raw Data

Place the following in `./preprocess/datasets/`:

* **Enron Emails**
  [https://www.cs.cmu.edu/\~enron/](https://www.cs.cmu.edu/~enron/)
  → `preprocess/datasets/maildir/`

* **PII-Masked Dataset**
  [Kaggle: AI4Privacy PII](https://www.kaggle.com/datasets/verracodeguacas/ai4privacy-pii)
  → `preprocess/datasets/PII43k.csv`

* **WikiText-103** & **OpenWebText**
  Automatically downloaded via Hugging Face.

### 1.2 Model-Specific Dataset Use

| Model       | Dataset                               |
| ----------- | ------------------------------------- |
| GPT-2 Nano  | WikiText-103, OpenWebText, Enron, PII |
| GPT-2 Large | Enron, PII                            |

---

### 🧹 Step 1.3: Clean and Merge Datasets

1. **Create output directories**

```bash
mkdir -p preprocess/cleaned_dataset preprocess/merged_dataset
```

2. **Run preprocessing**

```bash
sh scripts/run_preprocess.sh
```

3. **Merge datasets**

```bash
sh scripts/run_merge_dataset.sh
```

Output:

```
preprocess/merged_dataset/
├── merged_large.txt
└── merged_nano.txt
```

---

## 🧠 Step 2: Train Models

### \[Optional] Enable WandB Logging

Add your API key in `wandbkey.env`:

```env
WANDB_API_KEY=yourkey
```

---

### 🚀 2.1 GPT-2 Large (LoRA Fine-Tune)

1. **Create directories**

```bash
mkdir -p chkpt/large GPT2-Large/data/tokenized_data
```

2. **Train Non-DP Model**

```bash
python GPT2-Large/GPT2_finetune.py
```

📍 Checkpoints saved to: `chkpt/large/plain/`

3. **Train with DP-SGD**

```bash
python GPT2-Large/GPT2_opacus_finetune.py
```

📍 Checkpoints saved to: `chkpt/large/dp_sgd/`

---

### 🧪 2.2 GPT-2 Nano (Train from Scratch)

1. **Prepare Data Directory**

```bash
mkdir -p GPT2-Nano/data
```

2. **Tokenize Dataset**

```bash
python GPT2-Nano/prepare.py
```

Output: `GPT2-Nano/data/customtext/`

3. **Train Non-DP**

```bash
python GPT2-Nano/train.py
```

4. **Train with DP-SGD**

```bash
python GPT2-Nano/train_DP_SGD.py
```

📍 Checkpoints saved to:

```
chkpt/nano/plain/
chkpt/nano/dp-sgd/
```

---

### 🧾 Optional: Use Pretrained Checkpoints

📦 [Download (Google Drive)](https://drive.google.com/drive/folders/1i2-KrwdKyX9Ufemq3hpoqJRkKZnemLJ9?usp=sharing)

Place into:

```
chkpt/
├── large/
│   ├── plain/
│   └── dp_sgd/
└── nano/
    ├── plain/
    └── dp-sgd/
```

---

## 📐 Step 3: Evaluate Accuracy & Leakage

### 3.1 Generate Prompt–Reference Pairs

```bash
python measure_accuracy/generate_prset.py
```

📂 Output:

```
measure_accuracy/ref/
├── prompts.txt
├── references.txt
├── prompts_pii.txt
└── references_pii.txt
```

---

### 3.2 Generate Outputs from Models

```bash
sh scripts/generate_output.sh
```

📂 Output:

```
measure_accuracy/outputs_from_model/
├── large/
│   ├── output_plain.txt
│   └── output_dp.txt
└── nano/
    ├── output_plain.txt
    └── output_dp.txt
```

---

### 3.3 Evaluate Outputs

* **NanoGPT**

```bash
python measure_accuracy/measure_accuracy.py --size nano
```

* **GPT-2 Large**

```bash
python measure_accuracy/measure_accuracy.py --size large
```

---

## 🧪 Step 4: Run Extraction and Categorization

This step measures **how much memorized or sensitive content** each model reproduces during generation. We evaluate memorization by:

* Generating samples from the model
* Ranking them using **perplexity** and **compression-based metrics**
* Categorizing extracted content (e.g., names, emails, URLs, code, etc.)

---

### 🧰 4.1 Run Extraction

#### ✅ Prerequisites

* Trained model checkpoints:

```
chkpt/
├── large/
│   ├── plain/checkpoint-245000/
│   └── dp_sgd/checkpoint-245000/
└── nano/
    ├── plain/checkpoint-60000.pt
    └── dp-sgd/checkpoint-60000.pt
```

* Required libraries: `tiktoken`, `transformers`, `peft`, `torch`, `zlib`, etc. (install via `requirements.txt`)
* (Optional) For real-world prompts: download a [Common Crawl `.wet` file](https://github.com/ftramer/LM_Memorization)

#### ▶️ Run the Extraction Script

```bash
python extraction_LMs.py
```

By default, this generates **10,000 samples**, ranks them by perplexity, and prints the top 100.

You can customize the run with arguments:

| Argument              | Default   | Description                                  |
| --------------------- | --------- | -------------------------------------------- |
| `--gen-model`         | `gpt2_dp` | Choose: `nano`, `nano_dp`, `gpt2`, `gpt2_dp` |
| `--batch-size`        | `100`     | Samples per generation batch                 |
| `--N`                 | `10000`   | Total number of samples to generate          |
| `--num-print`         | `100`     | Top samples to save and print                |
| `--seq-len`           | `256`     | Max token length per sample                  |
| `--top-k`             | `40`      | Use top-K sampling                           |
| `--internet-sampling` | *(flag)*  | Use real prompts from web data               |
| `--wet-file`          | `None`    | Required if using internet sampling          |

Example with custom options:

```bash
python extraction_LMs.py --gen-model gpt2_dp --N 10000 --num-print 20 --internet-sampling --wet-file commoncrawl.warc.wet
```

📄 **Output**:

* `results_model.txt` — top extracted samples
* Console output with perplexity scores

---

### 🧾 4.2 Run Categorization

This step **automatically labels memorized content** by matching against categories like names, contact info, licenses, and more.

#### ✅ Prerequisites

* Merged datasets:

```
preprocess/
├── cleaned_dataset/
├── merged_dataset/
│   ├── merged_large.txt
│   └── merged_nano.txt
```

* Output file from Step 4.1 (e.g., `results_gpt2_dp.txt`)

#### ▶️ Run the Categorization Script

```bash
python categorization.py --sample-file results_gpt2_dp.txt
```

📄 **Output**:

* `final_summary_results_gpt2_dp.csv` — includes matched categories, counts, and examples

#### 🧾 Sample CSV Output

| Category            | Unique Samples | Total Matches | Examples                                   |
| ------------------- | -------------- | ------------- | ------------------------------------------ |
| Named Individuals   | 46             | 119           | Aiden, Arlene, Bass, Amy...                |
| Contact Info        | 4              | 4             | 10 Downing Street, ...                     |
| Valid URLs          | 4              | 4             | [www.cisco.com](http://www.cisco.com), ... |
| Copyright & Terms   | 5              | 5             | "All rights reserved", ...                 |
| Promotional Content | 4              | 4             | "Buy Now", "Subscribe", ...                |
| Religious Texts     | 1              | 1             | "Jesus"                                    |
| Trump Quotes        | 2              | 2             | "Donald Trump"                             |

---

### ⚠️ 4.3 Limitations of Categorization

* Uses rule-based matching + spaCy NER
* May **miss ambiguous or uncommon PII**
* For best accuracy, combine with **manual review**

---



