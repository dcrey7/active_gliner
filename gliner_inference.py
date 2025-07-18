import time
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import torch
from tqdm import tqdm
from gliner import GLiNER
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Container for processing statistics"""
    total_text_length: int
    total_tokens: int
    num_chunks: int
    chunk_sizes: List[int]
    processing_time: float
    device: str
    model_name: str
    overlap_size: int
    entities_found: int
    avg_confidence: float


class GLiNERLargeTextProcessor:
    """
    Process large texts with GLiNER model using intelligent chunking and overlap.
    Handles texts that exceed model's maximum token limit.
    """
    
    def __init__(self, model_name: str = "urchade/gliner_large-v2.1", device: str = None):
        """
        Initialize the processor with GLiNER model.
        
        Args:
            model_name: Name of the GLiNER model to use
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        logger.info(f"Initializing GLiNER processor with model: {model_name}")
        
        # Load model
        self.model = GLiNER.from_pretrained(model_name)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")
        
        # Initialize tokenizer for accurate token counting
        # GLiNER uses BERT-based tokenizers typically
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Model configuration
        self.max_tokens = 512  # Conservative limit for BERT-based models
        self.model_name = model_name
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text"""
        return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def create_chunks_with_overlap(self, text: str, chunk_size: int = 384, overlap: int = 64) -> List[Tuple[str, int]]:
        """
        Create overlapping chunks from text.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in tokens
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of tuples (chunk_text, start_offset)
        """
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        stride = chunk_size - overlap
        
        logger.info(f"Creating chunks: chunk_size={chunk_size}, overlap={overlap}, stride={stride}")
        
        for i in range(0, len(tokens), stride):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + chunk_size]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Calculate character offset in original text
            # This is approximate due to tokenization
            prefix_tokens = tokens[:i]
            prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            start_offset = len(prefix_text)
            
            chunks.append((chunk_text, start_offset))
            
            # Stop if we've processed all tokens
            if i + chunk_size >= len(tokens):
                break
                
        logger.info(f"Created {len(chunks)} chunks from {len(tokens)} total tokens")
        return chunks
    
    def merge_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Merge overlapping entities, keeping the one with highest confidence.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Merged list of entities
        """
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        merged = []
        current = sorted_entities[0]
        
        for entity in sorted_entities[1:]:
            # Check if entities overlap
            if (entity['start'] <= current['end'] and 
                entity['label'] == current['label'] and
                abs(entity['start'] - current['start']) < 50):  # Similar position
                
                # Keep the one with higher confidence
                if entity['score'] > current['score']:
                    current = entity
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = entity
        
        # Don't forget the last entity
        merged.append(current)
        
        logger.info(f"Merged {len(entities)} entities down to {len(merged)}")
        return merged
    
    def process_text(self, text: str, labels: List[str], chunk_size: int = 384, 
                    overlap: int = 64, threshold: float = 0.5) -> Tuple[pd.DataFrame, ProcessingStats]:
        """
        Process large text and extract entities.
        
        Args:
            text: Input text to process
            labels: List of entity labels to extract
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks in tokens
            threshold: Confidence threshold for entities
            
        Returns:
            Tuple of (DataFrame with entities, ProcessingStats)
        """
        start_time = time.time()
        
        logger.info(f"Starting text processing. Text length: {len(text)} characters")
        logger.info(f"Labels to extract: {labels}")
        
        # Count total tokens
        total_tokens = self.count_tokens(text)
        logger.info(f"Total tokens in text: {total_tokens}")
        
        # Create chunks
        chunks = self.create_chunks_with_overlap(text, chunk_size, overlap)
        chunk_sizes = [self.count_tokens(chunk[0]) for chunk in chunks]
        
        # Process chunks with progress bar
        all_entities = []
        
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for i, (chunk_text, offset) in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Predict entities for this chunk
                try:
                    chunk_entities = self.model.predict_entities(
                        chunk_text, 
                        labels, 
                        threshold=threshold
                    )
                    
                    # Adjust positions based on chunk offset
                    for entity in chunk_entities:
                        entity['start'] += offset
                        entity['end'] += offset
                        entity['chunk_id'] = i
                    
                    all_entities.extend(chunk_entities)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
                
                pbar.update(1)
                pbar.set_postfix({'entities_found': len(all_entities)})
        
        # Merge overlapping entities
        logger.info(f"Found {len(all_entities)} total entities before merging")
        merged_entities = self.merge_overlapping_entities(all_entities)
        
        # Extract actual text for each entity
        for entity in merged_entities:
            try:
                entity['text'] = text[entity['start']:entity['end']]
            except:
                entity['text'] = entity.get('text', 'N/A')
        
        # Create DataFrame
        df = pd.DataFrame(merged_entities)
        
        # Select and rename columns for final output
        if not df.empty:
            df = df[['text', 'label', 'score']]
            df.columns = ['Value', 'Entity', 'Probability Score']
            df = df.sort_values('Probability Score', ascending=False).reset_index(drop=True)
        else:
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=['Value', 'Entity', 'Probability Score'])
        
        # Calculate statistics
        processing_time = time.time() - start_time
        avg_confidence = df['Probability Score'].mean() if not df.empty else 0.0
        
        stats = ProcessingStats(
            total_text_length=len(text),
            total_tokens=total_tokens,
            num_chunks=len(chunks),
            chunk_sizes=chunk_sizes,
            processing_time=processing_time,
            device=self.device,
            model_name=self.model_name,
            overlap_size=overlap,
            entities_found=len(df),
            avg_confidence=avg_confidence
        )
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Found {len(df)} unique entities")
        
        return df, stats
    
    def print_stats(self, stats: ProcessingStats):
        """Pretty print processing statistics"""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Model: {stats.model_name}")
        print(f"Device: {stats.device}")
        print(f"Total text length: {stats.total_text_length:,} characters")
        print(f"Total tokens: {stats.total_tokens:,}")
        print(f"Number of chunks: {stats.num_chunks}")
        print(f"Average chunk size: {sum(stats.chunk_sizes)/len(stats.chunk_sizes):.1f} tokens")
        print(f"Overlap size: {stats.overlap_size} tokens")
        print(f"Processing time: {stats.processing_time:.2f} seconds")
        print(f"Entities found: {stats.entities_found}")
        print(f"Average confidence: {stats.avg_confidence:.3f}")
        print("="*50 + "\n")


# Example usage
if __name__ == "__main__":
    from datasets import load_dataset
    
    # Load dataset
    logger.info("Loading dataset...")
    ds = load_dataset("hugsid/legal-contracts", split="train")
    
    # Initialize processor with the large model
    processor = GLiNERLargeTextProcessor(model_name="xomad/gliner-model-merge-large-v1.0")
    
    # Process text
    text = ds[1]['text']
    labels = ["person", "organization", "location", "date", "money", "percentage"]
    
    # Run processing
    results_df, stats = processor.process_text(
        text=text,
        labels=labels,
        chunk_size=384,  # Adjust based on your needs
        overlap=64,      # Overlap to catch entities at boundaries
        threshold=0.5
    )
    
    # Display results
    print("\nENTITY EXTRACTION RESULTS")
    print(results_df.head(20))  # Show top 20 entities
    
    # Display statistics
    processor.print_stats(stats)
    
    # Save results if needed
    # results_df.to_csv("extracted_entities.csv", index=False)