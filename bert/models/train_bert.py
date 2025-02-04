import os
import time
from datetime import datetime

import wandb
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from transformers import DistilBertForMaskedLM
from peft import LoraConfig, get_peft_model
from data.setup_corpus_data import get_retain_dataloader
from tqdm.auto import tqdm

def compute_metrics(logits, labels):
    """
    Computes accuracy for masked language modeling,
    considering only positions where label != -100.
    """
    predictions = logits.argmax(dim=-1)
    mask = labels != -100
    total = mask.sum().item()
    if total == 0:
        return 0.0
    correct = (predictions[mask] == labels[mask]).sum().item()
    return correct / total


wandb.login()
wandb.init(project="bert_finetune_project", name="custom_bert_finetune_run")


model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS",  # Token-level predictions
    target_modules=["q_lin", "v_lin"]
)
model = get_peft_model(model, lora_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


train_dataloader = get_retain_dataloader(batch_size=128, train=True)


optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()

num_epochs = 30
logging_steps = 100
gradient_accumulation_steps = 2
output_dir = "/media/jagadeesh/New Volume/Jagadeesh/GS-LoRA/bert/results"
os.makedirs(output_dir, exist_ok=True)

global_step = 0
running_loss = 0.0

start_time = time.time()

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0.0
    epoch_steps = 0
    epoch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=False)
    
    for step, batch in enumerate(epoch_iterator):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device_type='cuda'):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        epoch_steps += 1
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                wandb.log({"train_loss": avg_loss, "global_step": global_step})
                epoch_iterator.set_postfix(train_loss=avg_loss)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"bestmodel_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path} after Epoch {epoch+1}")

total_time = time.time() - start_time
print(f"Training completed in {total_time/60:.2f} minutes.")
