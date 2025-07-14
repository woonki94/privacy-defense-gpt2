import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set paths and generation params
BASE_MODEL_NAME = "gpt2-large"
#ADAPTER_CHECKPOINT_PATH = "./gpt2_cp_private_lora_only"
ADAPTER_CHECKPOINT_PATH = "./checkpoints/epoch_1"
PROMPT = "what is computer science"
MAX_LENGTH = 100
USE_CUDA = torch.cuda.is_available()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

# Load LoRA adapter on top of base model
model = PeftModel.from_pretrained(base_model, ADAPTER_CHECKPOINT_PATH)

# Move model to device
device = torch.device("cuda" if USE_CUDA else "cpu")
model.to(device)
model.eval()

# Tokenize prompt
inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n" + "-"*60)
print(generated_text)
