import os
import jsonlines
import re
import torch
import spacy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizer

# File paths and constants
DATA_FOLDER = "/media/jagadeesh/New Volume/Jagadeesh/GS-LoRA/bert/data/"


TRAIN_FILE = os.path.join(DATA_FOLDER, "retain_trian.jsonl")
VALID_FILE = os.path.join(DATA_FOLDER, "retain_valid.jsonl")
TEST_FILE = os.path.join(DATA_FOLDER, "retain_test.jsonl")
TRAIN_SIZE = 50000
VALID_SIZE = 7500

# Filtering configuration: sentences containing these words will be filtered out.
FILTER_WORDS = ["red", "black", "white", "blue", "green"]
FILTER_PATTERNS = {word: re.compile(rf"\b{word}\b", re.IGNORECASE) for word in FILTER_WORDS}

# Load spaCy English model for sentence segmentation.
nlp = spacy.load("en_core_web_sm")


def is_valid_sentence(sentence):
    """
    Check if a sentence is valid:
      - It must contain at least 6 words.
      - It must not contain any filtered words.
    """
    if len(sentence.strip().split()) < 6:
        return False
    for pattern in FILTER_PATTERNS.values():
        if pattern.search(sentence):
            return False
    return True


def process_bookcorpus():
    """
    Process the BookCorpus dataset in streaming mode:
      - Extract and filter sentences.
      - Save the first TRAIN_SIZE sentences for training and the next VALID_SIZE sentences for validation.
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)
    train_dataset = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)
    train_count, valid_count = 0, 0

    with jsonlines.open(TRAIN_FILE, mode="w") as train_writer, jsonlines.open(VALID_FILE, mode="w") as valid_writer:
        for example in tqdm(train_dataset, desc="Processing Data"):
            # Stop processing once both targets have been reached.
            if train_count >= TRAIN_SIZE and valid_count >= VALID_SIZE:
                break

            text = example.get("text", "").strip()
            if not text:
                continue

            doc = nlp(text)
            for sentence in doc.sents:
                # Check in the inner loop as well for early exit.
                if train_count >= TRAIN_SIZE and valid_count >= VALID_SIZE:
                    break

                sentence_text = sentence.text.strip()
                if is_valid_sentence(sentence_text):
                    if train_count < TRAIN_SIZE:
                        train_writer.write({"text": sentence_text})
                        train_count += 1
                    elif valid_count < VALID_SIZE:
                        valid_writer.write({"text": sentence_text})
                        valid_count += 1

    print(f" Saved {train_count} training and {valid_count} validation sentences")


class RetainDataset(Dataset):
    """
    PyTorch Dataset for masked language modeling.
    Reads a JSON Lines file and tokenizes & applies masking on the fly.
    """
    def __init__(self, dataset_path, tokenizer_name="distilbert-base-uncased"):
        self.dataset = []
        with jsonlines.open(dataset_path, "r") as reader:
            for obj in reader:
                self.dataset.append(obj)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

    def tokenize_and_mask(self, example):
        """
        Tokenize the input text and apply random masking:
          - Tokenize to a fixed length.
          - Use a probability matrix to mask tokens with 15% probability.
          - For tokens not selected for masking, set their label to -100 so they are ignored in loss computation.
        """
        inputs = self.tokenizer(
            example["text"],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # Clone input_ids for labels.
        labels = inputs["input_ids"].clone()

        # Create a probability matrix for masking with a probability of 15%.
        probability_matrix = torch.full(inputs["input_ids"].shape, 0.15)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            inputs["input_ids"][0].tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool).unsqueeze(0), value=0.0
        )
        mask_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels for unmasked tokens to -100 to ignore them during loss computation.
        labels[~mask_indices] = -100

        # Replace masked tokens in input_ids with the mask token id.
        inputs["input_ids"][mask_indices] = self.tokenizer.mask_token_id
        inputs["labels"] = labels

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["labels"].squeeze(0)
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.tokenize_and_mask(self.dataset[idx])


def get_retain_dataloader(batch_size=8, train=True):
    """
    Returns a DataLoader for the training or validation dataset.
    Shuffling is enabled only for training data.
    """
    dataset_path = TRAIN_FILE if train else VALID_FILE
    dataset = RetainDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
