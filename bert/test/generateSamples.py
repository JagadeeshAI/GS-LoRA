import torch
import json
from transformers import DistilBertTokenizer, DistilBertForMaskedLM

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the trained model
model_path = "bestmodel.pth"
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Load test data
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

test_data = load_jsonl("retain_test.jsonl") + load_jsonl("forget_test.jsonl")

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

# Function to predict masked word
def predict_masked_word(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.topk(outputs.logits[0, mask_index], k=1, dim=-1).indices.squeeze().tolist()
    predicted_word = tokenizer.convert_ids_to_tokens(predictions)

    return predicted_word[0] if isinstance(predicted_word, list) else predicted_word

# Generate masked samples and predictions
generated_samples = []

for sample in test_data:
    sentence = sample["text"]
    masked_sentence, masked_word = mask_sentence(sentence)

    if masked_sentence is not None and masked_word is not None:
        predicted_word = predict_masked_word(masked_sentence)
        generated_samples.append({
            "text": sentence,
            "masked_word": masked_word,
            "predicted_word": predicted_word
        })

# Save generated samples to JSONL
output_file = "generatedsamples.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in generated_samples:
        f.write(json.dumps(entry) + "\n")

print(f"Generated samples saved to {output_file}")
