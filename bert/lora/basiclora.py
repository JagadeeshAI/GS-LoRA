import torch
from transformers import DistilBertForMaskedLM
from peft import LoraConfig, get_peft_model

def get_lora_config():
    """
    Returns a predefined LoRA configuration.
    """
    return LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout probability
        bias="none",
        task_type="TOKEN_CLS",  # Token-level classification for MLM
        target_modules=["q_lin", "v_lin"]  # LoRA applied to attention layers
    )

def load_lora_model():
    """
    Loads DistilBERT with LoRA applied.
    """
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    model.gradient_checkpointing_enable()  # Reduce memory usage

    # Apply LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device
