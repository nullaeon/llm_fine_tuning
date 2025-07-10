from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.utils as nn_utils
import torch

# === Config ===
MODEL_NAME = "tiiuae/falcon-rw-1b"
MAX_SEQ_LENGTH = 1024

# === Toy dataset === (Replace with your own!)
data = {
    "prompt": [
        "Explain the difference between AI and machine learning.",
        "How do I train a neural network?",
        "What is gradient descent?"
    ],
    "completion": [
        "AI is the broader concept of machines being able to carry out tasks in a way that we would consider smart, while ML is a specific subset of AI that involves learning from data.",
        "Training a neural network involves feeding it data, computing loss, and using backpropagation to update weights.",
        "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent."
    ]
}
# dataset = Dataset.from_dict(data)
dataset = load_dataset("tatsu-lab/alpaca")["train"]

# === Tokenizer & Formatting ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def formatting(example):
    prompt = f"### Instruction:\n{example['instruction']}"
    if example.get("input"):
        prompt += f"\n### Input:\n{example['input']}"
    prompt += f"\n### Response:\n{example['output']}"
    return [prompt]

# === Load model and apply LoRA ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value"]
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

# === Training config ===
training_args = TrainingArguments(
    output_dir="./falcon-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    logging_steps=1,
    num_train_epochs=5,
    max_steps=-1,
    bf16=False,
    fp16=False,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    max_grad_norm=10.0,
    warmup_steps=10,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
)

# == Monkey patch for trl > 0.8.0
class ClippingSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            nn_utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        return loss


# === Trainer ===
trainer = ClippingSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=formatting,
    max_seq_length=MAX_SEQ_LENGTH,
)

trainer.train()
trainer.model.save_pretrained("./falcon-lora/checkpoint")
