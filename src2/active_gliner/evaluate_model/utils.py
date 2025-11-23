import pandas as pd
from typing import List, Dict, Tuple, Optional
from ..create_data.gliner_utils import tokenize_text, char_to_word_positions


def convert_char_to_token_format(predictions: List[List[Dict]], data: List[Dict]) -> List[List[List]]:
    """
    Convert GLiNER character-level predictions to token-level format.

    Args:
        predictions: GLiNER predictions (character-level)
                    [[{'start': 17, 'end': 29, 'label': 'actor', 'text': '...', 'score': 0.95}, ...], ...]
        data: Original data with tokenized_text and original text
              [{'tokenized_text': ['what', 'movies', ...], 'text': 'original sentence', ...}, ...]

    Returns:
        Token-level predictions: [[[start, end, label, text, score], ...], ...]
    """
    token_predictions = []

    for example_preds, example_data in zip(predictions, data):
        tokenized_text = example_data['tokenized_text']
        original_text = example_data.get('text', ' '.join(tokenized_text))  # Use original or reconstruct

        example_token_preds = []
        for pred in example_preds:
            char_start = pred['start']
            char_end = pred['end']
            label = pred['label']
            text = pred['text']
            score = pred.get('score', 1.0)  # Default to 1.0 for LLM predictions

            # Convert to token positions using original text (NOT reconstructed from tokens)
            token_start, token_end = char_to_word_positions(original_text, char_start, char_end)

            if token_start is not None and token_end is not None:
                example_token_preds.append([token_start, token_end, label, text, score])

        token_predictions.append(example_token_preds)

    return token_predictions


def apply_gradient_styling(df: pd.DataFrame, title: str = ""):
    """
    Apply gradient color styling to dataframe.

    Args:
        df: DataFrame to style
        title: Optional title for the styled table

    Returns:
        Styled dataframe with gradient colors
    """
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
                cmap='RdYlGn',  # Red-Yellow-Green (low=red, high=green)
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


def display_results(results: Dict):
    """
    Pretty print evaluation results.

    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Display overall metrics
    metrics = results['overall_metrics']
    print(f"\nOverall Metrics:")
    print(f"Total Predictions: {metrics['total_predictions']:,}")

    if 'overall_confidence' in metrics:
        print(f"Overall Confidence: {metrics['overall_confidence']:.4f} ({metrics['overall_confidence_pct']:.2f}%)")

    if 'total_examples' in metrics:
        print(f"Total Examples: {metrics['total_examples']:,}")
        print(f"Correct Examples: {metrics['correct_examples']:,}")
        print(f"Incorrect Examples: {metrics['incorrect_examples']:,}")
        print(f"Example-Level Accuracy: {metrics['example_level_accuracy']:.4f} ({metrics['example_level_accuracy_pct']:.2f}%)")
        print(f"Entity-Level Accuracy: {metrics['entity_level_accuracy']:.4f} ({metrics['entity_level_accuracy_pct']:.2f}%)")

    if 'overall_f1' in metrics:
        print(f"Overall F1 Score: {metrics['overall_f1']:.4f} ({metrics['overall_f1_pct']:.2f}%)")

    # Display styled dataframes
    if 'confidence_bins' in results:
        print(f"\nConfidence Distribution:")
        display(results['confidence_bins'])

    if 'classification_report' in results:
        print(f"\nClassification Report:")
        display(results['classification_report'])

    if 'tp_confidence_analysis' in results:
        print(f"\nTrue Positives Confidence Analysis:")
        display(results['tp_confidence_analysis'])

    if 'fp_confidence_analysis' in results:
        print(f"\nFalse Positives Confidence Analysis:")
        display(results['fp_confidence_analysis'])

    if 'high_confidence_examples' in results:
        print(f"\nHigh Confidence Examples: {len(results['high_confidence_examples'])}")
        print(f"Low Confidence Examples: {len(results['low_confidence_examples'])}")
