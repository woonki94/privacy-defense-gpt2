import os
from transformers import AutoModelForCausalLM
import sys
from data_file import get_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
import torch
import wandb
from dotenv import load_dotenv

load_dotenv(dotenv_path = "wandbkey.env")
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    report_to= "wandb"
    wandb.login(key = wandb_api_key)
    wandb.init(project="lora-gpt2", config={
        "model": "gpt2-large",
        "dp_noise_multiplier": 0.5,
        "dp_max_grad_norm": 1.0,
        "batch_size": 8,
        "epochs": 3,
        "lr": 1e-4
    })
else:
    report_to = "none"


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
chkpt_dir = os.path.join(data_dir, 'chkpt')

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

model_name = "gpt2-large"

finetune_data = get_dataset()

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token



if os.path.exists(os.path.join(data_dir,"tokenized_data","tokenized_gpt_large")):
    tokenized_dataset = load_from_disk(os.path.join(data_dir,"tokenized_data","tokenized_gpt_large"))

    # Use only the first half of the dataset
    tokenized_dataset = tokenized_dataset.shuffle(seed=42)

    # Use only the first quarter of the dataset
    quarter_len = len(tokenized_dataset) // 20
    tokenized_dataset = tokenized_dataset.select(range(quarter_len))

    print(f"Using only {quarter_len} examples out of {len(tokenized_dataset)}")
else:
    print("Tokenizing raw dataset...")
    tokenized_dataset = finetune_data.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=8)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_dataset.save_to_disk(os.path.join(data_dir,"tokenized_data","tokenized_gpt_large"))
    print("Saved tokenized dataset.")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is causal LM
)

model = AutoModelForCausalLM.from_pretrained("gpt2-large")

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False

training_args = TrainingArguments(
    overwrite_output_dir=True,
    gradient_checkpointing=False,
    output_dir=os.path.join(chkpt_dir,"plain"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=5000,
    save_steps=5000,
    logging_steps=50000,
    save_total_limit=24,
    learning_rate=1e-7,
    # dataloader_num_workers=8,
    dataloader_pin_memory=True,
    bf16=True,
    report_to=report_to,
    disable_tqdm=False
)
# if training_args.gradient_checkpointing:
#     model.enable_input_require_grads()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(100)),  # Small eval split
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
