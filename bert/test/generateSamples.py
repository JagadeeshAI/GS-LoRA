import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from peft import LoraConfig, get_peft_model

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load model and move to GPU
model_path = "/media/jagadeesh/New Volume/Jagadeesh/GS-LoRA/bert/results/bestmodel_20250205_002103.pth"
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS",  # Token-level predictions
    target_modules=["q_lin", "v_lin"]
)
model = get_peft_model(model, lora_config)

# Load model weights and move to GPU
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load test data
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

test_data = load_jsonl("/media/jagadeesh/New Volume/Jagadeesh/GS-LoRA/bert/data/test.jsonl")

# Function to mask a word in a sentence
def mask_sentence(sentence):
    words = sentence.split()
    if len(words) < 3:  # Skip very short sentences
        return None, None

    mask_index = torch.randint(0, len(words), (1,)).item()
    masked_word = words[mask_index]
    words[mask_index] = "[MASK]"
    masked_sentence = " ".join(words)

    return masked_sentence, masked_word

# Custom dataset class for batching
class MaskedDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for sample in data:
            sentence = sample["text"]
            masked_sentence, masked_word = mask_sentence(sentence)
            if masked_sentence is not None and masked_word is not None:
                self.data.append((masked_sentence, masked_word, sentence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create dataset & dataloader
batch_size = 128
dataset = MaskedDataset(test_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Function to predict masked words for a batch
def predict_batch(batch_sentences):
    inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU

    with torch.no_grad():
        outputs = model(**inputs)

    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

    predictions = []
    for i in range(len(batch_sentences)):
        idx = mask_token_index[1][i].item()
        top_pred = torch.topk(outputs.logits[i, idx], k=1).indices.squeeze().item()
        predicted_word = tokenizer.convert_ids_to_tokens(top_pred)
        predictions.append(predicted_word)

    return predictions

# Process data in batches
generated_samples = []

for batch in dataloader:
    batch_sentences, batch_masked_words, original_sentences = batch

    predicted_words = predict_batch(batch_sentences)

    for i in range(len(batch_sentences)):
        generated_samples.append({
            "text": original_sentences[i],
            "masked_word": batch_masked_words[i],
            "predicted_word": predicted_words[i],
            "type": "retain"
        })

# Save generated samples to JSONL
output_file = "/media/jagadeesh/New Volume/Jagadeesh/GS-LoRA/bert/results/predicted.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in generated_samples:
        f.write(json.dumps(entry) + "\n")

print(f"Generated samples saved to {output_file}")
