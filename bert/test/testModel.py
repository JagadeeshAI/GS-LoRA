import torch
import json
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForMaskedLM

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model_path = "bestmodel.pth"
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

FORGET_KEYS = ["red", "blue", "white", "black", "green"]

# Load datasets
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

retain_data = load_jsonl("retain_test.jsonl")
forget_data = load_jsonl("forget_test.jsonl")

def mlm_forget_test(data, forget_keys):
    forget_count = 0
    total = 0

    for sample in data:
        sentence = sample["text"]  
        for key in forget_keys:
            if key in sentence:
                masked_sentence = sentence.replace(key, "[MASK]")
                inputs = tokenizer(masked_sentence, return_tensors="pt")
                mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

                with torch.no_grad():
                    outputs = model(**inputs)
                
                predictions = torch.topk(outputs.logits[0, mask_index], k=5, dim=-1).indices.squeeze().tolist()
                predicted_tokens = tokenizer.convert_ids_to_tokens(predictions)

                if key in predicted_tokens:
                    forget_count += 1  
                total += 1

    return 1 - (forget_count / total) if total > 0 else 1.0  

def log_prob_forget_test(data, forget_keys):
    log_probs = []

    for sample in data:
        sentence = sample["text"]
        for key in forget_keys:
            if key in sentence:
                masked_sentence = sentence.replace(key, "[MASK]")
                inputs = tokenizer(masked_sentence, return_tensors="pt")
                mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs.logits[0, mask_index].squeeze()
                prob = torch.nn.functional.softmax(logits, dim=-1)
                key_id = tokenizer.convert_tokens_to_ids(key)

                log_probs.append(torch.log(prob[key_id]).item())

    avg_log_prob = np.mean(log_probs) if log_probs else -float('inf')
    return avg_log_prob

def mlm_accuracy_test(data):
    correct = 0
    total = 0

    for sample in data:
        sentence = sample["text"]
        words = sentence.split()
        
        masked_sentence = sentence
        for word in words:
            if word not in FORGET_KEYS and np.random.rand() < 0.15:
                masked_sentence = masked_sentence.replace(word, "[MASK]", 1)
                break  
        
        inputs = tokenizer(masked_sentence, return_tensors="pt")
        mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.topk(outputs.logits[0, mask_index], k=1, dim=-1).indices.squeeze().tolist()
        predicted_token = tokenizer.convert_ids_to_tokens(predictions)

        if predicted_token in words:
            correct += 1
        total += 1

    return correct / total if total > 0 else 1.0 

forget_accuracy = mlm_forget_test(forget_data, FORGET_KEYS)
retain_accuracy = mlm_accuracy_test(retain_data)
log_prob_forget = log_prob_forget_test(forget_data, FORGET_KEYS)

if (forget_accuracy + retain_accuracy) > 0:
    h_mean = (2 * forget_accuracy * retain_accuracy) / (forget_accuracy + retain_accuracy)
else:
    h_mean = 0.0

print("=== Forgetting Test Results ===")
print(f"MLM Completion Forget Accuracy: {forget_accuracy:.4f}")
print(f"MLM Accuracy on Retained Data: {retain_accuracy:.4f}")
print(f"Log Probability of Forgotten Words: {log_prob_forget:.4f}")
print(f"H-Mean Metric: {h_mean:.4f}")
