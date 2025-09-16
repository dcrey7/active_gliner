"""
Test data loading functions
"""
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess.data_loading import load_mit_dataset, load_results_from_file

def test_mit_dataset():
    print("=== Testing MIT Dataset Loading ===")
    
    # User can change these paths
    data_path = "../data/mit-movie/train.json"
    labels_path = "../data/mit-movie/labels.json"
    
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        print("💡 Change data_path to your actual data file")
        return
        
    try:
        data, entity_types = load_mit_dataset(data_path, labels_path, "train")
        print(f"✅ Loaded {len(data)} examples")
        print(f"🏷️ Entity types: {entity_types}")
        print(f"📝 First example: {' '.join(data[0]['tokenized_text'][:5])}...")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_results_loading():
    print("\n=== Testing Results Loading ===")
    
    # User can change this path  
    results_path = "../results/results.json"
    
    if not os.path.exists(results_path):
        print(f"❌ File not found: {results_path}")
        print("💡 Change results_path to your actual results file")
        return
        
    try:
        results = load_results_from_file(results_path)
        print(f"✅ Loaded {len(results['all_predictions'])} predictions")
        print(f"📊 Keys: {list(results.keys())}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_mit_dataset()
    test_results_loading()