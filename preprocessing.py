import os
import re
import time
import json
import gzip
import tarfile
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def basic_clean_text(text):
    """Basic text cleaning - remove extra spaces, special chars, etc."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-.,;:!?\'"()]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def download_and_extract_samples(num_samples=1000):
    """Download the tar.gz file and extract first N samples"""
    url = "https://huggingface.co/datasets/albertvillanova/legal_contracts/resolve/main/contracts.tar.gz"
    
    logger.info(f"Downloading dataset from {url}...")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Save to temp file
    temp_file = "temp_contracts.tar.gz"
    with open(temp_file, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Extract samples
    texts = []
    logger.info(f"Extracting first {num_samples} contracts...")
    
    with tarfile.open(temp_file, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.name.endswith('.txt')]
        
        for i, member in enumerate(tqdm(members[:num_samples], desc="Extracting")):
            f = tar.extractfile(member)
            if f:
                text = f.read().decode('utf-8', errors='ignore')
                texts.append(text)
    
    # Clean up temp file
    os.remove(temp_file)
    
    return texts


def main():
    start_time = time.time()
    
    # Create directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Load samples
    logger.info("Loading legal contracts dataset...")
    try:
        texts = download_and_extract_samples(num_samples=1000)
        logger.info(f"Successfully loaded {len(texts)} contracts")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Save raw samples
    logger.info("Saving raw data...")
    raw_data = [{"id": i, "text": text} for i, text in enumerate(texts[:100])]
    with gzip.open("data/raw/legal_contracts_sample.jsonl.gz", 'wt') as f:
        for item in raw_data:
            f.write(json.dumps(item) + '\n')
    
    # Show before/after example
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE - First document:")
    logger.info("="*50)
    logger.info(f"BEFORE ({len(texts[0])} chars): {texts[0][:200]}...")
    cleaned_example = basic_clean_text(texts[0])
    logger.info(f"AFTER ({len(cleaned_example)} chars): {cleaned_example[:200]}...")
    logger.info("="*50 + "\n")
    
    # Process all texts
    logger.info("Cleaning texts...")
    processed_data = []
    total_chars_before = 0
    total_chars_after = 0
    
    for i, text in enumerate(tqdm(texts, desc="Processing")):
        chars_before = len(text)
        cleaned = basic_clean_text(text)
        chars_after = len(cleaned)
        
        total_chars_before += chars_before
        total_chars_after += chars_after
        
        processed_data.append({
            "id": i,
            "text": cleaned,
            "original_length": chars_before,
            "cleaned_length": chars_after
        })
    
    # Save processed data
    logger.info("Saving processed data...")
    df = pd.DataFrame(processed_data)
    df.to_parquet("data/processed/legal_contracts_processed.parquet", compression='snappy')
    
    # Also save as compressed jsonl
    with gzip.open("data/processed/legal_contracts_processed.jsonl.gz", 'wt') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    # Calculate and show stats
    processing_time = time.time() - start_time
    avg_reduction = (1 - total_chars_after / total_chars_before) * 100
    
    logger.info("\n" + "="*50)
    logger.info("PROCESSING STATS")
    logger.info("="*50)
    logger.info(f"Documents processed: {len(processed_data)}")
    logger.info(f"Total chars before: {total_chars_before:,}")
    logger.info(f"Total chars after: {total_chars_after:,}")
    logger.info(f"Average reduction: {avg_reduction:.1f}%")
    logger.info(f"Processing time: {processing_time:.1f} seconds")
    logger.info("="*50)
    
    # Save stats
    stats = {
        "total_documents": len(processed_data),
        "total_chars_before": total_chars_before,
        "total_chars_after": total_chars_after,
        "avg_reduction_percent": avg_reduction,
        "processing_time": processing_time
    }
    
    with open("data/processed/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()