# Data Extraction and Defense in LLMs

This project explores **memorization and privacy risks** in large language models (LLMs), focusing on the potential leakage of **personally identifiable information (PII)**. We replicate and extend **data extraction attacks** and propose a **differentially private training pipeline** using **DP-SGD**, optimized for both:

* **NanoGPT** (trained from scratch)
* **GPT-2 Large** (fine-tuned with LoRA adapters)

## What We Do

1. **Train and fine-tune LLMs** on datasets containing sensitive content (e.g., Enron emails + synthetic PII).
2. **Evaluate privacy risk** via black-box data extraction (Carlini-style) by generating and filtering outputs.
3. **Defend** against memorization with a scalable, memory-efficient version of **DP-SGD**:

   * **Ghost Clipping**: Efficient clipping for linear layers like LoRA.
   * **Fast Gradient Clipping**: General fallback for non-linear layers.
   * **Virtual Batching**: Allows large batch sizes under GPU constraints.

## Experiments

| Model           | Training Method | Privacy Level | Training Goal              |
| --------------- | --------------- | ------------- | -------------------------- |
| **NanoGPT**     | From scratch    | DP / Non-DP   | Public + synthetic PII mix |
| **GPT-2 Large** | LoRA fine-tune  | DP / Non-DP   | Enron + PII                |

We compare model outputs using:

* **BLEU**, **ROUGE**, **BERTScore** (for utility)
* **Perplexity & Compression**-based filters (for memorization)
* **Manual annotation** of extracted PII

## Key Results

* **DP-SGD reduces PII leakage by >90%** in both NanoGPT and GPT-2 Large
* **Only minor drops in generation quality** (≤4 BERTScore points)
* **Memory usage reduced** with ghost clipping: from >40GB to \~30GB (GPT-2 Large)

---

## End-to-End Usage Guide

We provide step-by-step instructions to:
- Prepare datasets from **Enron**, **PII-masked corpora**, and general text
- Train **GPT-2 Nano** from scratch and fine-tune **GPT-2 Large** using **LoRA**
- Apply **DP-SGD** using memory-efficient mechanisms like ghost clipping
- Evaluate model utility and data leakage using BLEU, ROUGE, BERTScore, and extraction metrics

### Environment Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Recommended hardware**

- NVIDIA H100 GPU
- At least **40 GB of GPU memory** is required to train GPT-2 Large with DP-SGD

---

### Step 1: Preprocess the Dataset

#### 1.1 Download and Place Raw Data

Place the following datasets in `./preprocess/datasets/`:

- **Enron Emails**
  - Download from: [https://www.cs.cmu.edu/~enron/](https://www.cs.cmu.edu/~enron/)
  - After unzipping, the directory should look like:
    ```
    preprocess/datasets/maildir/
    ```

- **PII-Masked Dataset**
  - Download from: [https://www.kaggle.com/datasets/verracodeguacas/ai4privacy-pii](https://www.kaggle.com/datasets/verracodeguacas/ai4privacy-pii?select=pii-masking-43k)
  - Place the CSV file as:
    ```
    preprocess/datasets/PII43k.csv
    ```

- **WikiText-103** and **OpenWebText**
  - These will be automatically downloaded via the Hugging Face `datasets` library during preprocessing

#### 1.2 Dataset Usage by Model

- **GPT-2 Nano (trained from scratch)** uses:
  - WikiText-103 (general)
  - OpenWebText 5GB subset (general)
  - Enron Emails (private)
  - PII-Masked Data (private)

- **GPT-2 Large (fine-tuned)** uses:
  - Enron Emails (private)
  - PII-Masked Data (private)

---

### Step 2: Clean and Merge the Dataset

1. **Create output directories**

```bash
mkdir -p preprocess/cleaned_dataset
mkdir -p preprocess/merged_dataset
```

2. **Run preprocessing script**

```bash
sh scripts/run_preprocess.sh
```

This will create cleaned files in:

```
preprocess/cleaned_dataset/
├── cleaned_email.txt
├── cleaned_pii.txt
├── cleaned_openwebtext_5gb.txt
└── cleaned_wikitext103.txt
```

3. **Merge datasets**

```bash
sh scripts/run_merge_dataset.sh
```

This script calls:

- `merge_dataset_large.py` for GPT-2 Large (Enron + PII only)
- `merge_dataset_nano.py` for GPT-2 Nano (Enron + PII + general corpora)

You will get:

```
preprocess/merged_dataset/
├── merged_large.txt
└── merged_nano.txt
```

At this point, preprocessing is complete.



## Step 2: Train Models

If you want to enable logging with Weights & Biases (WandB), add your API key to:

```
root/wandbkey.env
```

Inside that file, include the line:

```
WANDB_API_KEY=yourkey
```

---

### 2.1 GPT-2 Large (Fine-Tuning)

1. **Create directories**

```bash
mkdir -p chkpt/large
mkdir -p GPT2-Large/data/tokenized_data
```

2. **Train without DP (plain model)**

```bash
python GPT2-Large/GPT2_finetune.py
```

Checkpoints will be saved to:

```
chkpt/large/plain/
```

3. **Train with DP-SGD**

```bash
python GPT2-Large/GPT2_opacus_finetune.py
```

Checkpoints will be saved to:

```
chkpt/large/dp_sgd/
```

---

### 2.2 GPT-2 Nano (Train from Scratch)

1. **Create directory**

```bash
mkdir -p GPT2-Nano/data
```

2. **Prepare the dataset**

```bash
python GPT2-Nano/prepare.py
```

This will create:

```
GPT2-Nano/data/customtext/
├── train.bin
├── val.bin
└── meta.pkl
```

3. **Train without DP (plain model)**

```bash
python GPT2-Nano/train.py
```

4. **Train with DP-SGD**

```bash
python GPT2-Nano/train_DP_SGD.py
```

Checkpoints will be saved to:

```
chkpt/nano/plain/
chkpt/nano/dp-sgd/
```

---

### Optional: Use Pretrained Checkpoints

If you do not wish to retrain the models from scratch, you may download our pretrained checkpoints from the following link:

**[Download Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1i2-KrwdKyX9Ufemq3hpoqJRkKZnemLJ9?usp=sharing)**

After downloading, place the contents into the appropriate checkpoint directories:

```
chkpt/
├── large/
│   ├── plain/
│   └── dp_sgd/
└── nano/
    ├── plain/
    └── dp-sgd/
```



## Step 3: Measure Accuracy

We evaluate the generation quality of each model using prompt–reference pairs built from datasets used during training.

- For **GPT-2 Nano**, we use prompts from **WikiText-103** (used in training).
- For **GPT-2 Large**, we use prompts from **Enron emails** and **PII data** (used during fine-tuning).

---

### 3.1 Build Prompt–Reference Sets

Run the script to generate the prompt–reference files:

```bash
python measure_accuracy/generate_prset.py
```

This will create:

```
measure_accuracy/ref/
├── prompts.txt             # for GPT-2 Nano
├── references.txt          # for GPT-2 Nano
├── prompts_pii.txt         # for GPT-2 Large
└── references_pii.txt      # for GPT-2 Large
```

---

### 3.2 Generate Model Outputs

Use each model to generate responses to the prompt sets:

```bash
sh scripts/generate_output.sh
```

This will generate output files:

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

### 3.3 Run Evaluation Script

Measure BLEU, ROUGE, and BERTScore against the references:

- **For GPT-2 Nano:**

```bash
python measure_accuracy/measure_accuracy.py --size nano
```

- **For GPT-2 Large:**

```bash
python measure_accuracy/measure_accuracy.py --size large
```

This will print evaluation metrics comparing the outputs of the plain and DP-trained models.

