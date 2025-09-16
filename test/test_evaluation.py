"""
Test evaluation functions with real data
"""
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiment.active_learning import get_lowest_score_examples_sorted
from preprocess.data_loading import load_results_from_file

def test_active_learning_with_real_data():
    print("=== Testing Active Learning with Real Results ===")
    
    # User can change this path
    results_path = "results.json"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        print("💡 Change results_path to your actual results file")
        return
    
    try:
        # Load real results
        results = load_results_from_file(results_path)
        
        # User can change n (number of examples to extract)
        n = 5
        
        examples = get_lowest_score_examples_sorted(results, n=n)
        
        print(f"✅ Loaded results with {len(results['all_predictions'])} predictions")
        print(f"📊 Requested: {n} examples")
        print(f"📊 Got: {len(examples)} examples")
        print("📝 Lowest confidence examples:")
        
        for i, ex in enumerate(examples):
            min_score = min(ex['scores']) if ex['scores'] else 0.0
            text = ' '.join(ex['tokenized_text'][:10])  # First 10 tokens
            print(f"   {i+1}. Score: {min_score:.3f} | Text: {text}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_active_learning_with_real_data()