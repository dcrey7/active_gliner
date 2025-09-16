"""
Test preprocessing functions with real data
"""
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess.data_transformation import tokenize_text, convert_synthetic_to_ner_format

def test_tokenization():
    print("=== Testing Text Tokenization ===")
    
    # User can change these test texts
    test_texts = [
        "Tom Hanks starred in Forrest Gump.",
        "The action movie directed by Steven Spielberg was amazing!",
        "What's your favorite sci-fi film from 2023?",
        "Great performance by the lead actor in this thriller."
    ]
    
    for i, text in enumerate(test_texts):
        try:
            tokens = tokenize_text(text)
            print(f"\n📝 Test {i+1}:")
            print(f"   Input: {text}")
            print(f"   Tokens: {tokens}")
            print(f"   Count: {len(tokens)} tokens")
        except Exception as e:
            print(f"❌ Tokenization failed for text {i+1}: {e}")

def test_synthetic_conversion():
    print("\n=== Testing Synthetic Data Conversion ===")
    
    # Sample synthetic data (like what would come from LLM)
    synthetic_data = [
        {
            "text": "Great action movie starring Tom Hanks",
            "entities": [
                {"entity": "action", "types": ["genre"]},
                {"entity": "Tom Hanks", "types": ["actor"]}
            ]
        },
        {
            "text": "Steven Spielberg directed this thriller in 2023", 
            "entities": [
                {"entity": "Steven Spielberg", "types": ["director"]},
                {"entity": "thriller", "types": ["genre"]},
                {"entity": "2023", "types": ["year"]}
            ]
        }
    ]
    
    try:
        ner_data = convert_synthetic_to_ner_format(synthetic_data)
        
        print(f"✅ Conversion successful!")
        print(f"📊 Converted {len(ner_data)} examples")
        
        for i, (synthetic, ner) in enumerate(zip(synthetic_data, ner_data)):
            print(f"\n📝 Example {i+1}:")
            print(f"   Original text: {synthetic['text']}")
            print(f"   Original entities: {synthetic['entities']}")
            print(f"   Tokenized: {ner['tokenized_text']}")
            print(f"   NER format: {ner['ner']}")
            
    except Exception as e:
        print(f"❌ Synthetic conversion failed: {e}")

def test_tokenization_edge_cases():
    print("\n=== Testing Tokenization Edge Cases ===")
    
    # Test edge cases
    edge_cases = [
        "",  # Empty string
        "Single",  # Single word
        "Multi-word entity",  # Hyphenated
        "Movie_Title_2023",  # Underscores
        "What's happening?!",  # Punctuation
        "123 Main St.",  # Numbers and abbreviations
    ]
    
    for text in edge_cases:
        try:
            tokens = tokenize_text(text)
            print(f"📝 '{text}' → {tokens} ({len(tokens)} tokens)")
        except Exception as e:
            print(f"❌ Failed for '{text}': {e}")

if __name__ == "__main__":
    test_tokenization()
    test_synthetic_conversion()
    test_tokenization_edge_cases()