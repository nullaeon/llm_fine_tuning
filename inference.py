from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# === Load base model and LoRA adapter ===
BASE_MODEL = "tiiuae/falcon-rw-1b"
ADAPTER_PATH = "./falcon-lora/checkpoint"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# === Test prompt ===
prompt = "### Prompt:\nHow do I train a neural network?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )

print("\n=== Output ===")
print(tokenizer.decode(output[0], skip_special_tokens=True))