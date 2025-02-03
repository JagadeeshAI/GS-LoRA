import os
import jsonlines
import re
import torch
import spacy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizer

# File paths
DATA_FOLDER = "bert/data/"
TRAIN_FILE = os.path.join(DATA_FOLDER, "retain_train.jsonl")
TEST_FILE = os.path.join(DATA_FOLDER, "retain_test.jsonl")
TRAIN_SIZE = 50000  # Number of sentences for training
TEST_SIZE = 7500  # Number of sentences for testing

# Words to filter out
FILTER_WORDS = ["red", "black", "white", "blue", "green"]
FILTER_PATTERNS = {word: re.compile(rf"\b{word}\b", re.IGNORECASE) for word in FILTER_WORDS}

# Load spaCy tokenizer
nlp = spacy.load("en_core_web_sm")


def is_valid_sentence(sentence):
    """Checks if a sentence is valid (not too short, no unwanted words)."""
    MIN_WORDS = 6  # Adjust as needed
    words = sentence.strip().split()

    # Ensure sentence has enough words and does not contain forbidden terms
    if len(words) < MIN_WORDS:
        return False

    for word, pattern in FILTER_PATTERNS.items():
        if pattern.search(sentence):
            return False  # Contains an unwanted word

    return True


def process_bookcorpus():
    """
    Downloads BookCorpus (train and test splits), filters sentences, and saves:
    - 50,000 clean sentences for `retain_train.jsonl`
    - 7,500 clean sentences for `retain_test.jsonl`
    Saves in JSONL format using streaming.
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Process training dataset (50,000 sentences)
    print("🔄 Downloading & Processing BookCorpus (Train Split)...")
    train_dataset = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)
    train_count = 0

    with jsonlines.open(TRAIN_FILE, mode="w") as train_writer:
        for example in tqdm(train_dataset, desc="Processing Train Data"):
            text = example.get("text", "").strip()
            doc = nlp(text)  # Tokenize into sentences

            for sentence in doc.sents:
                sentence_text = sentence.text.strip()

                if is_valid_sentence(sentence_text):
                    train_writer.write({"text": sentence_text})
                    train_count += 1

                if train_count >= TRAIN_SIZE:
                    break
            if train_count >= TRAIN_SIZE:
                break
    print("train set is about", train_count)            
    # Process testing dataset (7,500 sentences)
    print("🔄 Downloading & Processing BookCorpus (Test Split)...")
    # test_dataset = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)
    test_count = 0

    with jsonlines.open(TEST_FILE, mode="w") as test_writer:
        for example in tqdm(train_dataset, desc="Processing Test Data"):
            text = example.get("text", "").strip()
            doc = nlp(text)  

            for sentence in doc.sents:
                sentence_text = sentence.text.strip()

                if is_valid_sentence(sentence_text):
                    test_writer.write({"text": sentence_text})
                    test_count += 1

                if test_count >= TEST_SIZE:
                    break
            if test_count >= TEST_SIZE:
                break
                            
    print(f"✅ Saved {train_count} training sentences in {TRAIN_FILE}")
    print(f"✅ Saved {test_count} testing sentences in {TEST_FILE}")


class RetainDataset(Dataset):
    """
    Loads BookCorpus dataset (train or test) from JSONL.
    """
    def __init__(self, dataset_path, tokenizer_name="distilbert-base-uncased"):
        # Check if dataset exists, otherwise download
        if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
            print("⚠️ Dataset not found. Extracting from BookCorpus...")
            process_bookcorpus()

        # Load dataset from JSONL
        self.dataset = []
        with jsonlines.open(dataset_path, "r") as reader:
            for obj in reader:
                self.dataset.append(obj)

        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

    def tokenize_and_mask(self, example):
        """
        Tokenizes text and applies masked language modeling (MLM).
        """
        inputs = self.tokenizer(
            example["text"],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inputs["labels"] = inputs["input_ids"].clone()

        # Apply masking (MLM with 15% probability)
        probability_matrix = torch.full(inputs["input_ids"].shape, 0.15)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            inputs["input_ids"][0].tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).unsqueeze(0)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs["input_ids"][masked_indices] = self.tokenizer.mask_token_id

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.tokenize_and_mask(self.dataset[idx])


def get_retain_dataloader(batch_size=8, train=True):
    """
    Returns a PyTorch DataLoader for train or test dataset.
    """
    dataset_path = TRAIN_FILE if train else TEST_FILE
    dataset = RetainDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Test the DataLoader
if __name__ == "__main__":
    train_dataloader = get_retain_dataloader(batch_size=8, train=True)
    test_dataloader = get_retain_dataloader(batch_size=8, train=False)

    print("🔹 Sample Training Batch:")
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i+1}: {batch}")
        if i == 2:  # Stop after 3 batches
            break

    print("\n🔹 Sample Testing Batch:")
    for i, batch in enumerate(test_dataloader):
        print(f"Batch {i+1}: {batch}")
        if i == 2:  # Stop after 3 batches
            break
