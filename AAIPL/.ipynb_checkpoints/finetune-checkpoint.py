# ============================================================
# FIX READ-ONLY CACHE (MUST BE FIRST)
# ============================================================
import os
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache/transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hf_cache/hub"

os.makedirs("/workspace/hf_cache/datasets", exist_ok=True)
os.makedirs("/workspace/hf_cache/transformers", exist_ok=True)
os.makedirs("/workspace/hf_cache/hub", exist_ok=True)


# ============================================================
# IMPORTS
# ============================================================
import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, TrainerCallback
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTTrainer, SFTConfig
import subprocess, threading, time


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "qwen14b_lora"
DATA_FILE = "data/train2.json"
OUTPUT_DIR = "qwen14b_lora"
MAX_SEQ_LENGTH = 4096


# ============================================================
# LOAD MODEL (QLORA)
# ============================================================
print("\nLoading Qwen2.5-14B...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# IMPORTANT: auto detect Qwen template
tokenizer = get_chat_template(tokenizer)


# ============================================================
# MODEL INFO
# ============================================================
print("\n===== MODEL INFO =====")
print("Device:", next(model.parameters()).device)
print("Trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Total params:",
      sum(p.numel() for p in model.parameters()))
print("======================\n")


# ============================================================
# LOAD DATASET
# ============================================================
print("Loading dataset...")

dataset = load_dataset("json", data_files=DATA_FILE, split="train")

print("Columns:", dataset.column_names)
print("Sample raw:", dataset[0])

# Standardize ShareGPT format
dataset = standardize_sharegpt(dataset)


# ============================================================
# APPLY CHAT TEMPLATE
# ============================================================
def formatting_prompts_func(examples):
    texts = []
    for convo in examples["conversations"]:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset.column_names
)

dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

print("\nFormatted sample:\n")
print(dataset[0]["text"][:700])


# ============================================================
# TOKEN STATS
# ============================================================
lengths = [len(tokenizer(x["text"]).input_ids) for x in dataset.select(range(min(200,len(dataset))))]
print("\n===== TOKEN STATS =====")
print("Avg tokens:", sum(lengths)/len(lengths))
print("Max tokens:", max(lengths))
print("Min tokens:", min(lengths))
print("=======================\n")


# ============================================================
# TRAIN HEARTBEAT CALLBACK
# ============================================================
class HeartbeatCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            loss = state.log_history[-1].get("loss", None)
            print(f"[STEP {state.global_step}] loss={loss}")


# ============================================================
# GPU MONITOR
# ============================================================
def gpu_monitor():
    while True:
        try:
            out = subprocess.check_output("rocm-smi --showuse --showmemuse", shell=True).decode()
            print("\n[GPU STATUS]\n", "\n".join(out.split("\n")[:6]))
        except:
            pass
        time.sleep(60)

threading.Thread(target=gpu_monitor, daemon=True).start()


# ============================================================
# TRAINER
# ============================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    ),
)

trainer.add_callback(HeartbeatCallback)

# IMPORTANT: automatic masking for Qwen
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

FastLanguageModel.for_training(model)

print("\n===== TRAINING START =====\n")

trainer.train()


# ============================================================
# SAVE ADAPTER
# ============================================================
print("\nSaving LoRA adapters...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete! Adapter saved to:", OUTPUT_DIR)
