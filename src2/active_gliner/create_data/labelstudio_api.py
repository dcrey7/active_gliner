"""
Label Studio API operations.

Simple utilities for:
- Fetching tasks from API
- Filtering tasks by predictions
- Splitting train/test with annotation priority
"""

from typing import List, Dict, Tuple, Optional
import random
import gc
from label_studio_sdk import LabelStudio


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def has_prediction_from_model(task: Dict, model_keywords: List[str]) -> bool:
    """
    Check if task has prediction from model matching keywords.

    Args:
        task: Label Studio task
        model_keywords: Keywords to match (case-insensitive)

    Returns:
        bool: True if any prediction matches any keyword

    Example:
        >>> has_prediction_from_model(task, ['abhi-gliner', 'gliner'])
        True
    """
    predictions = task.get('predictions', [])

    for pred in predictions:
        model_version = pred.get('model_version', '').lower()
        # Check if any keyword matches this prediction
        if any(keyword.lower() in model_version for keyword in model_keywords):
            return True

    return False


def has_annotations(task: Dict) -> bool:
    """
    Check if task has annotations.

    Args:
        task: Label Studio task

    Returns:
        bool: True if task has at least one annotation
    """
    annotations = task.get('annotations', [])
    return len(annotations) > 0


def get_prediction_score(
    task: Dict,
    model_keywords: List[str],
    default: float = 0.0
) -> float:
    """
    Get prediction score from first matching prediction.

    Used for uncertainty-based sorting (higher score = more uncertain).

    Args:
        task: Label Studio task
        model_keywords: Keywords to match in model_version
        default: Default score if no match found (default: 0.0)

    Returns:
        float: Prediction score from first matching prediction
    """
    predictions = task.get('predictions', [])

    for pred in predictions:
        model_version = pred.get('model_version', '').lower()
        # Check if any keyword matches
        if any(keyword.lower() in model_version for keyword in model_keywords):
            return pred.get('score', default)

    return default


# ============================================================================
# FETCH TASKS
# ============================================================================

def fetch_tasks(
    project_id: int,
    base_url: str,
    api_key: str,
    page_size: int = 2000,
    verbose: bool = True
) -> List[Dict]:
    """
    Fetch all tasks from Label Studio project.

    Args:
        project_id: Label Studio project ID
        base_url: Label Studio URL (e.g., 'http://localhost:8080')
        api_key: Label Studio API key
        page_size: Number of tasks per page (default: 2000)
        verbose: Print progress messages (default: True)

    Returns:
        List[Dict]: List of tasks (includes predictions and annotations)

    Example:
        >>> tasks = fetch_tasks(
        ...     project_id=42,
        ...     base_url='http://localhost:8080',
        ...     api_key='your_api_key'
        ... )
        >>> print(f"Fetched {len(tasks)} tasks")
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print("FETCHING TASKS FROM LABEL STUDIO")
        print(f"{'=' * 80}")
        print(f"Project ID: {project_id}")
        print(f"Base URL: {base_url}")

    # Connect to Label Studio
    client = LabelStudio(base_url=base_url, api_key=api_key)

    try:
        # Fetch tasks (fields='all' includes predictions and annotations)
        tasks_page = client.tasks.list(
            project=project_id,
            page_size=page_size,
            fields='all'
        )
        tasks_list = [task.model_dump() for task in tasks_page.items]

        # Delete Pydantic page object to free memory
        del tasks_page

        if verbose:
            print(f"Fetched {len(tasks_list)} tasks")
            print(f"{'=' * 80}\n")

        return tasks_list

    finally:
        # Clean up SDK client to free resources
        del client
        gc.collect()


# ============================================================================
# FILTER TASKS
# ============================================================================

def filter_tasks_by_models(
    tasks: List[Dict],
    required_models: List[List[str]],
    sort_by_model: Optional[List[str]] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Filter tasks that have predictions from ALL required models.

    Args:
        tasks: List of Label Studio tasks
        required_models: List of model keyword lists (AND logic between groups)
            Each inner list is OR logic (any keyword can match).
            Task must have predictions matching ALL groups.
        sort_by_model: Model keywords to use for uncertainty sorting (optional)
            If provided, sorts by this model's score (highest first).
            If None, uses first group in required_models.
        verbose: Print filtering statistics (default: True)

    Returns:
        List[Dict]: Filtered tasks as [{'task': task, 'score': float}, ...]

    Example:
        >>> # Filter: Must have GLiNER AND (Cerebras OR Ollama)
        >>> # Sort by: GLiNER's uncertainty score
        >>> filtered = filter_tasks_by_models(
        ...     tasks,
        ...     required_models=[
        ...         ['abhi-gliner'],           # Must have GLiNER
        ...         ['cerebras', 'ollama']     # Must have Cerebras OR Ollama
        ...     ],
        ...     sort_by_model=['abhi-gliner']  # Sort by GLiNER's score
        ... )
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print("FILTERING TASKS BY PREDICTIONS")
        print(f"{'=' * 80}")
        print(f"Total tasks: {len(tasks)}")
        print(f"Required models (AND): {required_models}")
        if sort_by_model:
            print(f"Sort by model: {sort_by_model}")

    # Determine which model to use for sorting
    if sort_by_model is None:
        sort_by_model = required_models[0]

    filtered = []

    for task in tasks:
        # Check if task has predictions from ALL required model groups (AND logic)
        has_all_required = True
        for model_group in required_models:
            # Check if task has prediction matching any keyword in this group (OR logic)
            if not has_prediction_from_model(task, model_group):
                has_all_required = False
                break

        if has_all_required:
            # Get uncertainty score for sorting
            score = get_prediction_score(task, sort_by_model)
            filtered.append({'task': task, 'score': score})

    # Sort by uncertainty (highest score = most uncertain = prioritize for training)
    filtered.sort(key=lambda x: x['score'], reverse=True)

    if verbose:
        print(f"✓ Filtered: {len(filtered)} tasks")

        # Show per-model statistics
        for keywords in required_models:
            count = sum(1 for t in tasks if has_prediction_from_model(t, keywords))
            print(f"  - Tasks with {keywords}: {count}")

        if filtered:
            print(f"  - Score range: {filtered[-1]['score']:.4f} to {filtered[0]['score']:.4f}")

        print(f"{'=' * 80}\n")

    return filtered


# ============================================================================
# SPLIT TRAIN/TEST
# ============================================================================

def split_train_test_annotated_priority(
    all_tasks: List[Dict],
    test_fraction: float = 0.1,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split tasks into train/test with annotation priority.

    Strategy:
    1. If annotated tasks exist:
       - test_fraction of annotated tasks → test set
       - Remaining annotated + all other tasks → train set
    2. If no annotated tasks:
       - test_fraction of all tasks → test set
       - Remaining tasks → train set

    Args:
        all_tasks: All Label Studio tasks
        test_fraction: Fraction for test set (default: 0.1 = 10%)
        seed: Random seed for reproducibility (default: 42)
        verbose: Print split statistics (default: True)

    Returns:
        Tuple[List[Dict], List[Dict]]: (train_tasks, test_tasks)

    Example:
        >>> train, test = split_train_test_annotated_priority(
        ...     all_tasks=tasks,
        ...     test_fraction=0.1
        ... )
        >>> print(f"Train: {len(train)}, Test: {len(test)}")
    """
    random.seed(seed)

    if verbose:
        print(f"\n{'=' * 80}")
        print("SPLITTING TRAIN/TEST WITH ANNOTATION PRIORITY")
        print(f"{'=' * 80}")
        print(f"Total tasks: {len(all_tasks)}")
        print(f"Test fraction: {test_fraction * 100:.0f}%")
        print(f"Random seed: {seed}")

    # Separate annotated and non-annotated tasks
    annotated = [t for t in all_tasks if has_annotations(t)]
    other = [t for t in all_tasks if not has_annotations(t)]

    if verbose:
        print(f"\nTask breakdown:")
        print(f"  - Annotated: {len(annotated)}")
        print(f"  - Other: {len(other)}")

    # ========================================================================
    # CASE 1: We have annotated tasks
    # ========================================================================
    if len(annotated) > 0:
        if verbose:
            print(f"\n✓ Using ANNOTATED TASKS for test set")

        # Shuffle annotated tasks
        shuffled = annotated.copy()
        random.shuffle(shuffled)

        # Split: test_fraction → test, rest → train
        n_test = max(1, int(len(shuffled) * test_fraction))
        test_tasks = shuffled[:n_test]
        train_tasks_ann = shuffled[n_test:]

        # Add all other tasks to train
        train_tasks = train_tasks_ann + other

        if verbose:
            print(f"\nSplit results:")
            print(f"  Test: {len(test_tasks)} annotated tasks")
            print(f"  Train: {len(train_tasks)} tasks")
            print(f"    - {len(train_tasks_ann)} annotated")
            print(f"    - {len(other)} other")

    # ========================================================================
    # CASE 2: No annotated tasks
    # ========================================================================
    else:
        if verbose:
            print(f"\n  No annotated tasks - splitting all tasks")

        # Shuffle all tasks
        shuffled = all_tasks.copy()
        random.shuffle(shuffled)

        # Split: test_fraction → test, rest → train
        n_test = max(1, int(len(shuffled) * test_fraction))
        test_tasks = shuffled[:n_test]
        train_tasks = shuffled[n_test:]

        if verbose:
            print(f"\nSplit results:")
            print(f"  Test: {len(test_tasks)} tasks")
            print(f"  Train: {len(train_tasks)} tasks")

    if verbose:
        print(f"{'=' * 80}\n")

    return train_tasks, test_tasks


def select_training_tasks_by_mix(
    train_wrapped: List[Dict],
    annotated_fraction: float = 0.75,
    prediction_keywords: Optional[List[str]] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split training tasks into annotated and prediction buckets based on ratio.
    """
    if not train_wrapped:
        return [], []

    annotated_fraction = max(0.0, min(1.0, annotated_fraction))
    prediction_keywords = prediction_keywords or ['cerebras', 'ollama']
    total = len(train_wrapped)
    desired_annotated = int(round(total * annotated_fraction))
    desired_annotated = min(total, max(0, desired_annotated))
    desired_llm = total - desired_annotated

    rng = random.Random(seed)

    annotated_pool = [item for item in train_wrapped if has_annotations(item['task'])]
    prediction_pool = [
        item for item in train_wrapped
        if has_prediction_from_model(item['task'], prediction_keywords)
    ]

    rng.shuffle(annotated_pool)
    rng.shuffle(prediction_pool)

    def task_key(item: Dict) -> int:
        task = item.get('task', {})
        return task.get('id', id(task))

    used_ids = set()
    selected_ann: List[Dict] = []
    selected_llm: List[Dict] = []

    def select_from_pool(pool: List[Dict], quota: int, bucket: List[Dict]):
        for entry in pool:
            key = task_key(entry)
            if key in used_ids:
                continue
            bucket.append(entry)
            used_ids.add(key)
            if len(bucket) >= quota:
                break

    select_from_pool(annotated_pool, desired_annotated, selected_ann)
    select_from_pool(prediction_pool, desired_llm, selected_llm)

    shuffled_all = train_wrapped.copy()
    rng.shuffle(shuffled_all)

    def fill_bucket(bucket: List[Dict], quota: int, source_pool: List[Dict], predicate):
        if quota <= 0:
            return
        for entry in source_pool:
            key = task_key(entry)
            if key in used_ids or not predicate(entry):
                continue
            bucket.append(entry)
            used_ids.add(key)
            if len(bucket) >= quota:
                break

    fill_bucket(
        selected_ann,
        desired_annotated,
        shuffled_all,
        lambda item: has_annotations(item['task'])
    )
    fill_bucket(
        selected_llm,
        desired_llm,
        shuffled_all,
        lambda item: has_prediction_from_model(item['task'], prediction_keywords)
    )

    for entry in shuffled_all:
        key = task_key(entry)
        if key in used_ids:
            continue
        if has_annotations(entry['task']):
            selected_ann.append(entry)
        else:
            selected_llm.append(entry)
        used_ids.add(key)

    annotated_shortfall = desired_annotated - len(selected_ann)
    llm_shortfall = desired_llm - len(selected_llm)

    if verbose:
        print(f"\n{'=' * 80}")
        print("TRAIN MIX SELECTION")
        print(f"{'=' * 80}")
        print(f"Total training tasks: {total}")
        print(f"Requested annotated fraction: {annotated_fraction:.2f}")
        print(f"Target counts -> annotated: {desired_annotated}, llm: {desired_llm}")
        print(f"Selected annotated: {len(selected_ann)}")
        print(f"Selected LLM: {len(selected_llm)}")
        if annotated_shortfall > 0:
            print(f"Warning: missing {annotated_shortfall} annotated samples to meet target")
        if llm_shortfall > 0:
            print(f"Warning: missing {llm_shortfall} LLM samples to meet target")
        print(f"{'=' * 80}\n")

    return selected_ann, selected_llm
