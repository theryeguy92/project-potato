# src/data/dataset.py
import requests
from torch.utils.data import Dataset
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()


def download_shakespeare_text():
    """
    Download Shakespeare text from raw source.
    Returns the text content as a string.
    """
    # Use Andrej Karpathy's tiny shakespeare dataset (raw text)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    # Create cache directory
    cache_dir = Path("data/raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "shakespeare.txt"
    
    # Check if already cached
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Download
    print(f"  Downloading Shakespeare text from {url}")
    response = requests.get(url)
    response.raise_for_status()
    text = response.text
    
    # Cache it
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  Saved to cache: {cache_file}")
    
    return text


def load_shakespeare(tokenizer, split="train", chunk_size=512):
    """
    Load Shakespeare text and split into chunks.
    
    Args:
        tokenizer: Tokenizer to use
        split: "train" or "test" (we'll do a simple split)
        chunk_size: Size of text chunks in characters
    
    Returns:
        TextDataset instance
    """
    # Download full text
    full_text = download_shakespeare_text()
    
    # Simple split: 90% train, 10% test
    split_idx = int(len(full_text) * 0.9)
    
    if split == "train":
        text = full_text[:split_idx]
    else:
        text = full_text[split_idx:]
    
    # Split into chunks (by characters, not optimal but simple)
    # Each chunk will be around chunk_size characters
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50:  # Only keep chunks with reasonable length
            chunks.append(chunk)
    
    print(f"  Split: {split}, Chunks: {len(chunks)}")
    
    return TextDataset(chunks, tokenizer)