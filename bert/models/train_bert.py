import torch
from transformers import DistilBertForMaskedLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from bert.data.setup_corpus_data import get_retain_dataloader  # ✅ Load dataset properly

# Load base model
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

# Apply LoRA modifications
lora_config = LoraConfig(
    r=16,  # Rank of the LoRA matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    bias="none",
    task_type="CAUSAL_LM",  # Hugging Face's PEFT expects CAUSAL_LM for MLM tasks
    target_modules=["q_lin", "v_lin"]  # Specify target attention layers
)
model = get_peft_model(model, lora_config)

# Load dataset (50k sentences)
train_dataloader = get_retain_dataloader(batch_size=8, train=True)  # ✅ Load training dataset
train_dataset = train_dataloader.dataset  # ✅ Extract dataset from DataLoader

# Training arguments (Epoch-based training)
training_args = TrainingArguments(
    output_dir="./outputs/bert_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=128,
    num_train_epochs=30,  # ✅ Train for 3 full epochs
    save_steps=5000,
    logging_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
)

# Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset  # ✅ Pass dataset instead of DataLoader
)

# Start Training
trainer.train()
