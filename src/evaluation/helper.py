import torch
import numpy as np
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import matplotlib.pyplot as plt

def apply_gradient_styling(df, title=""):
    """Apply gradient color styling to dataframe"""
    
    def highlight_values(df):
        # Find numeric columns for gradient styling
        numeric_cols = []
        for col in df.columns:
            if col == 'entity_type':
                continue
            try:
                # Check if column has any non-numeric values (like '-')
                numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
                if numeric_mask.all():  # Only if ALL values are numeric
                    numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        # Apply gradient only to fully numeric columns
        styled_df = df.style
        if numeric_cols:
            styled_df = styled_df.background_gradient(
                cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (high=red, low=blue)
                subset=numeric_cols
            )
        
        # Format only numeric values, leave strings as-is
        def safe_format(val, format_str):
            try:
                if pd.isna(val) or val == '-':
                    return val
                return format_str.format(float(val))
            except (ValueError, TypeError):
                return val
        
        # Apply formatting to specific columns
        for col in df.columns:
            if col == 'entity_type':
                continue
            elif col in ['tp', 'fp', 'fn', 'support']:  # Count columns - no decimals
                styled_df = styled_df.format({col: lambda x: safe_format(x, '{:.0f}')})
            elif 'confidence' in col.lower():  # Confidence columns - 2 decimals
                styled_df = styled_df.format({col: lambda x: safe_format(x, '{:.2f}')})
            elif col in ['precision', 'recall', 'f1']:  # Metric columns - 2 decimals
                styled_df = styled_df.format({col: lambda x: safe_format(x, '{:.2f}')})
            else:  # Confidence bin counts - no decimals
                styled_df = styled_df.format({col: lambda x: safe_format(x, '{:.0f}')})
        
        if title:
            styled_df = styled_df.set_caption(title)
            
        return styled_df
    
    return highlight_values(df)


def display_results(results):
    """Display all results with proper formatting"""
    
    print("\n" + "="*80)
    print("ENHANCED EVALUATION RESULTS")
    print("="*80)
    
    # Display overall metrics
    metrics = results['overall_metrics']
    print(f"\nOverall Metrics:")
    print(f"Total Predictions: {metrics['total_predictions']:,}")
    print(f"Overall Confidence: {metrics['overall_confidence']:.4f} ({metrics['overall_confidence_pct']:.2f}%)")
    
    if 'total_examples' in metrics:
        print(f"Total Examples: {metrics['total_examples']:,}")
        print(f"Example-Level Accuracy: {metrics['example_level_accuracy']:.4f} ({metrics['example_level_accuracy_pct']:.2f}%)")
        print(f"Entity-Level Accuracy: {metrics['entity_level_accuracy']:.4f} ({metrics['entity_level_accuracy_pct']:.2f}%)")
        
        # Calculate and display F1 score from classification report
        if 'classification_report_df' in results:
            f1_score = results['classification_report_df'][results['classification_report_df']['entity_type'] == 'micro_avg']['f1'].iloc[0]
            print(f"Overall F1 Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    # Display styled dataframes
    print(f"\nConfidence Distribution:")
    display(results['confidence_bins'])
    
    if 'classification_report' in results:
        print(f"\nClassification Report:")
        display(results['classification_report'])
        
        print(f"\nTrue Positives Confidence Analysis:")
        display(results['tp_confidence_analysis'])
        
        print(f"\nFalse Positives Confidence Analysis:")
        display(results['fp_confidence_analysis'])
        
        print(f"\nIncorrect Examples: {len(results['incorrect_examples'])}")
        print(f"Corrected Labels Available: {len(results['corrected_examples'])}")
    
    if 'high_confidence_examples' in results:
        print(f"\nHigh Confidence Examples: {len(results['high_confidence_examples'])}")
        print(f"Low Confidence Examples: {len(results['low_confidence_examples'])}")
