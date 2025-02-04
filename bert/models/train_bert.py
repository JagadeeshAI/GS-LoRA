import os
import pickle
import torch
import numpy as np
from transformers import (
    DistilBertForMaskedLM, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback
)

try:
    from transformers.trainer_utils import TrainerControl
except ImportError:
    class TrainerControl:
        def __init__(self):
            self.should_save = False
            self.should_evaluate = False
            self.should_log = False
            self.should_epoch_stop = False
            self.should_training_stop = False

from peft import LoraConfig, get_peft_model
from bert.data.setup_corpus_data import get_retain_dataloader
from sklearn.metrics import accuracy_score

model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

# Enable gradient checkpointing to reduce memory usage.
model.gradient_checkpointing_enable()

# Configure LoRA.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS",  # Allowed task type for token-level predictions.
    target_modules=["q_lin", "v_lin"]
)
model = get_peft_model(model, lora_config)

# Prepare training and validation datasets.
train_dataloader = get_retain_dataloader(batch_size=8, train=True)
train_dataset = train_dataloader.dataset

validation_dataloader = get_retain_dataloader(batch_size=8, train=False)
validation_dataset = validation_dataloader.dataset


def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    if np.sum(mask) == 0:
        accuracy = 0.0
    else:
        accuracy = (predictions[mask] == labels[mask]).mean()
    return {'eval_accuracy': accuracy}


class EmptyCacheCallback(TrainerCallback):
    
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        return control


class DiskFlushTrainer(Trainer):
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Create a temporary directory for flush files.
        temp_dir = "./temp_eval_results"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get evaluation DataLoader.
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Determine GPU memory threshold.
        device = self.args.device
        device_props = torch.cuda.get_device_properties(device)
        total_mem = device_props.total_memory
        flush_threshold = 0.95  # 95% usage threshold
        
        # Lists for accumulating predictions and labels.
        all_preds = []
        all_labels = []
        flush_files = []
        flush_count = 0
        
        self.model.eval()
        for step, inputs in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = self.prediction_step(
                    self.model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
                )
            # outputs is a tuple: (loss, logits, labels)
            logits = outputs[1]
            labels = outputs[2]
            
            if logits is not None:
                logits = logits.detach().cpu()
            if labels is not None:
                labels = labels.detach().cpu()
                
            if logits is not None:
                all_preds.append(logits)
            if labels is not None:
                all_labels.append(labels)
            
            # Check current GPU memory usage.
            current_mem = torch.cuda.memory_allocated(device)
            if current_mem / total_mem >= flush_threshold:
                # Flush the accumulated predictions and labels to disk.
                file_path = os.path.join(temp_dir, f"eval_flush_{flush_count}.pkl")
                with open(file_path, "wb") as f:
                    pickle.dump((all_preds, all_labels), f)
                flush_files.append(file_path)
                flush_count += 1
                # Clear the in-memory lists.
                all_preds = []
                all_labels = []
                torch.cuda.empty_cache()
        
        # Flush any remaining results.
        if all_preds or all_labels:
            file_path = os.path.join(temp_dir, f"eval_flush_{flush_count}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump((all_preds, all_labels), f)
            flush_files.append(file_path)
        
        # Read and combine all flushed files.
        combined_preds = []
        combined_labels = []
        for file_path in flush_files:
            with open(file_path, "rb") as f:
                preds_list, labels_list = pickle.load(f)
            if preds_list:
                combined_preds.extend(preds_list)
            if labels_list:
                combined_labels.extend(labels_list)
            os.remove(file_path)
        # Clean up the temporary directory if empty.
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
        
        # Concatenate all predictions and labels.
        if combined_preds:
            combined_preds = torch.cat(combined_preds, dim=0)
        else:
            combined_preds = None
        if combined_labels:
            combined_labels = torch.cat(combined_labels, dim=0)
        else:
            combined_labels = None
        
        # Compute metrics using the provided compute_metrics function.
        metrics = {}
        if self.compute_metrics is not None and combined_preds is not None and combined_labels is not None:
            metrics = self.compute_metrics((combined_preds.numpy(), combined_labels.numpy()))
        # Set self.control to a new instance of TrainerControl to avoid errors.
        self.control = TrainerControl()
        return metrics


training_args = TrainingArguments(
    output_dir="./outputs/bert_model",
    per_device_train_batch_size=128,  # Adjust based on your hardware.
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",  # Note: evaluation_strategy is deprecated; using eval_strategy.
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    report_to="none",
    gradient_accumulation_steps=2,   # Accumulate gradients over 2 steps.
    fp16=True,                       # Enable mixed precision training.
    eval_accumulation_steps=2         # This parameter is optional with disk flushing.
)

trainer = DiskFlushTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EmptyCacheCallback()]  # Use custom callback to free GPU memory.
)

trainer.train()
