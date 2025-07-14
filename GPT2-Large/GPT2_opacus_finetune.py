import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

# === Monkey-patch torch.zeros to fix Opacus dtype issue ===
_orig_zeros = torch.zeros
def safe_zeros(*args, **kwargs):
    dtype = kwargs.get("dtype", None)
    if isinstance(dtype, type):
        if dtype == int:
            kwargs["dtype"] = torch.int32
        elif dtype == float:
            kwargs["dtype"] = torch.float32
        else:
            kwargs["dtype"] = torch.float32
    return _orig_zeros(*args, **kwargs)
torch.zeros = safe_zeros

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from opacus import PrivacyEngine

from datasets import load_from_disk
from data_file import get_dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import gc
import logging
import wandb
from dotenv import load_dotenv

load_dotenv(dotenv_path = "wandbkey.env")
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    report_to= "wandb"
    wandb.login(key = wandb_api_key)
    wandb.init(project="dp-lora-gpt2", config={
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

logging.basicConfig(filename="./log/training_errors.log", level=logging.ERROR,
                    format="%(asctime)s %(levelname)s: %(message)s")


class CustomDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        input_ids = []
        attention_masks = []

        for e in examples:
            ids = e["input_ids"][:self.max_length]
            mask = e["attention_mask"][:self.max_length]

            # Enforce long dtype explicitly
            input_ids.append(ids.clone().detach())
            attention_masks.append(mask.clone().detach())

        # Pad all sequences
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        labels = input_ids_padded.clone()


        return {
            "input_ids": input_ids_padded.to(dtype=torch.long),
            "attention_mask": attention_mask_padded.to(dtype=torch.float32),
            "labels": labels.to(dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2-large")


def tokenize_fn(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

finetune_data = get_dataset()

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

#tokenized_dataset = load_from_disk("./train_dataset/tokenized_gpt_mixed")
custom_collator = CustomDataCollator(tokenizer,max_length=512)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn = custom_collator,
)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)#5e-5
criterion = nn.CrossEntropyLoss()

# Attach PrivacyEngine
'''
privacy_engine = PrivacyEngine()
model, optimizer,criterion, _ = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_dataloader,
    noise_multiplier=0.5,#2.0
    max_grad_norm=1.0,
    grad_sample_mode="ghost"
)
'''

privacy_engine = PrivacyEngine()
model, optimizer, criterion, _ = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_dataloader,
    criterion=criterion,
    target_epsilon=7.5,
    target_delta=1e-5,  # same as before
    epochs=3,
    max_grad_norm=1.0,  # or lower if you want less clipping
    grad_sample_mode="ghost",  # keep this for ghost clipping
)




model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 3
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        try:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()


        except Exception as e:
            logging.error(f"Epoch {epoch} Step {step} - Exception occurred: {str(e)}", exc_info=True)
            print(f"[ERROR] Step {step} failed. Logged to file.")


        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        if wandb_api_key:
            wandb.log({
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
                "epsilon": epsilon
            })

        print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, ε: {epsilon:.2f}")


        if step % 100 == 0:
            for k, v in batch.items():
                 print(f"{k}: {v.dtype}, shape: {v.shape}")

        if step % 5000 ==0:
            lora_model = model._module
            save_path = os.path.join(chkpt_dir, "dp_sgd", f"epochs_{epoch}", f"iterations_{step}")
            os.makedirs(save_path, exist_ok=True)
            lora_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            torch.save(privacy_engine.accountant.state_dict(), os.path.join(save_path, "privacy_state.pt"))
            torch.save({"epoch": epoch, "step": step}, os.path.join(save_path, "trainer_state.pt"))
            print(f"Saved LoRA model checkpoint to {save_path}")


        del input_ids, attention_mask, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()

    lora_model = model._module
    save_path = os.path.join(chkpt_dir,"dp_sgd","epochs_",f"{epoch}","_fin")
    os.makedirs(save_path, exist_ok=True)
    lora_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    torch.save(privacy_engine.accountant.state_dict(), os.path.join(save_path, "privacy_state.pt"))
    torch.save({"epoch": epoch, "step": 0}, os.path.join(save_path, "trainer_state.pt"))
    print(f"Saved LoRA model checkpoint to {save_path}")





