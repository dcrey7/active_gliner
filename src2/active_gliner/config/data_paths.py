import os

# Get the project root directory (assuming this file is in src2/active_slimmer/config/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MIT_movies_NER_train_path = os.path.join(_PROJECT_ROOT, "data", "mit-movie", "train.json")
MIT_movies_NER_test_path = os.path.join(_PROJECT_ROOT, "data", "mit-movie", "test.json")
MIT_movies_NER_dev_path = os.path.join(_PROJECT_ROOT, "data", "mit-movie", "dev.json")
MIT_movies_NER_labels_path = os.path.join(_PROJECT_ROOT, "data", "mit-movie", "labels.json")

