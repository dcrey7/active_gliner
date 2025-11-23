def calculate_mse_score(prediction) -> float:
    """
    Calculate MSE (Mean Squared Error) uncertainty score for a single example.

    MSE measures average squared distance from perfect confidence (1.0).
    Higher MSE indicates more uncertain predictions.

    Args:
        prediction: List of entity dicts with 'score' field
                   [{'score': 0.9, 'label': 'actor', ...}, ...]

    Returns:
        MSE score (0.0 to 1.0), where higher values indicate more uncertainty.
        Returns 1.0 if no predictions (maximum uncertainty).

    Example:
        >>> pred = [{'score': 0.9}, {'score': 0.85}, {'score': 0.95}]
        >>> calculate_mse_score(pred)
        0.0058  # Low uncertainty (high confidence)

        >>> pred = [{'score': 0.5}, {'score': 0.6}, {'score': 0.4}]
        >>> calculate_mse_score(pred)
        0.2233  # High uncertainty (low confidence)
    """
    entity_scores = [entity['score'] for entity in prediction]
    if not entity_scores:
        return 1.0  # Maximum uncertainty for no predictions

    # Calculate MSE: mean of squared errors from perfect confidence
    squared_errors = [(1.0 - score) ** 2 for score in entity_scores]
    mse = sum(squared_errors) / len(squared_errors)

    return mse


def calculate_min_score(prediction) -> float:
    """
    Calculate minimum confidence score for a single example.

    Finds the lowest confidence among all predicted entities.
    Lower minimum score indicates higher uncertainty (weakest prediction).

    Args:
        prediction: List of entity dicts with 'score' field
                   [{'score': 0.9, 'label': 'actor', ...}, ...]

    Returns:
        Minimum score (0.0 to 1.0). Returns 0.0 if no predictions.

    Example:
        >>> pred = [{'score': 0.9}, {'score': 0.85}, {'score': 0.95}]
        >>> calculate_min_score(pred)
        0.85  # Lowest confidence entity
    """
    entity_scores = [entity['score'] for entity in prediction]
    return min(entity_scores) if entity_scores else 0.0


def calculate_avg_score(prediction) -> float:
    """
    Calculate average confidence score for a single example.

    Computes mean confidence across all predicted entities.
    Lower average score indicates overall lower confidence.

    Args:
        prediction: List of entity dicts with 'score' field
                   [{'score': 0.9, 'label': 'actor', ...}, ...]

    Returns:
        Average score (0.0 to 1.0). Returns 0.0 if no predictions.

    Example:
        >>> pred = [{'score': 0.9}, {'score': 0.8}, {'score': 0.7}]
        >>> calculate_avg_score(pred)
        0.8  # Average confidence
    """
    entity_scores = [entity['score'] for entity in prediction]
    return sum(entity_scores) / len(entity_scores) if entity_scores else 0.0


import math

def calculate_mnlp_score(prediction) -> float:
    """
    Calculate MNLP (Maximum Normalized Log Probability) uncertainty score.

    MNLP = (-1/n) * sum(log(confidence_i))
    Higher MNLP indicates more uncertain predictions.

    Input:
        prediction: List of entity dicts with 'score' field
                   [{'score': 0.9, 'label': 'actor', ...}, ...]

    Output:
        MNLP score (0.0 to inf), higher = more uncertain.
        Returns float('inf') if no predictions (maximum uncertainty).

    Example:
        >>> pred = [{'score': 0.9}, {'score': 0.85}, {'score': 0.95}]
        >>> calculate_mnlp_score(pred)
        0.1054  # Low uncertainty

        >>> pred = [{'score': 0.5}, {'score': 0.6}, {'score': 0.4}]
        >>> calculate_mnlp_score(pred)
        0.6365  # High uncertainty
    """
    entity_scores = [entity['score'] for entity in prediction]
    if not entity_scores:
        return float('inf')  # Maximum uncertainty for no predictions

    # Clamp scores to avoid log(0)
    eps = 1e-10
    log_probs = [math.log(max(score, eps)) for score in entity_scores]
    mnlp = -sum(log_probs) / len(log_probs)

    return mnlp


# Strategy configuration for sorting
# Maps strategy name to (score_function, reverse_sort)
# reverse=True means higher score = more uncertain (sort descending)
# reverse=False means lower score = more uncertain (sort ascending)
STRATEGY_CONFIG = {
    'mse': (calculate_mse_score, True),      # Higher MSE = more uncertain
    'min': (calculate_min_score, False),     # Lower min = more uncertain
    'avg': (calculate_avg_score, False),     # Lower avg = more uncertain
    'mnlp': (calculate_mnlp_score, True),    # Higher MNLP = more uncertain
}