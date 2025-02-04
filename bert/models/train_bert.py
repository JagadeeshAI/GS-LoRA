import torch
from transformers import DistilBertForMaskedLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from bert.data.setup_corpus_data import get_retain_dataloader
from utils.util import compute_metrics, EmptyCacheCallback, DiskFlushTrainer

# Load the base model for masked language modeling.
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

# Define training arguments.
training_args = TrainingArguments(
    output_dir="./outputs/bert_model",
    per_device_train_batch_size=128,  # Adjust based on your hardware.
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",  # Using evaluation every eval_steps.
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    report_to="none",
    gradient_accumulation_steps=2,   # Accumulate gradients over 2 steps.
    fp16=True,                       # Enable mixed precision training.
    eval_accumulation_steps=2         # Optional when using disk flushing.
)

# Create the custom trainer instance.
trainer = DiskFlushTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EmptyCacheCallback()]  # Use custom callback to free GPU memory.
)

# Start training.
trainer.train()
