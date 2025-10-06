#!/bin/bash

# This script fixes ALL __init__.py files to minimal working versions
# Run from project root: bash fix_all_init_files.sh

echo "🔧 Fixing all __init__.py files..."

# Get the src directory
SRC_DIR="${1:-src}"

if [ ! -d "$SRC_DIR" ]; then
    echo "❌ Error: Directory $SRC_DIR not found!"
    echo "Usage: bash fix_all_init_files.sh [src_directory]"
    exit 1
fi

# 1. Config
echo "Fixing config/__init__.py..."
cat > "$SRC_DIR/config/__init__.py" << 'EOF'
"""Config Module"""
from .settings import Settings
GLOBAL_SEED = 42
BATCH_SIZE = 8
EOF

# 2. Data
echo "Fixing data/__init__.py..."
cat > "$SRC_DIR/data/__init__.py" << 'EOF'
"""Data Module"""
from .loader import load_mit_dataset
from .transforms import tokenize_text, convert_synthetic_to_ner_format
try:
    from .validator import NERValidator
    from .validation_report import ValidationReport
except:
    pass
EOF

# 3. Evaluation
echo "Fixing evaluation/__init__.py..."
cat > "$SRC_DIR/evaluation/__init__.py" << 'EOF'
"""Evaluation Module"""
from .evaluator import enhanced_evaluate
try:
    from .ner_evaluator import create_ner_evaluator
except:
    pass
EOF

# 4. Generation
echo "Fixing generation/__init__.py..."
cat > "$SRC_DIR/generation/__init__.py" << 'EOF'
"""Generation Module"""
try:
    from .label_generator import create_label_generator
except:
    pass
EOF

# 5. Training
echo "Fixing training/__init__.py..."
cat > "$SRC_DIR/training/__init__.py" << 'EOF'
"""Training Module"""
from .trainer import train_lora_model
EOF

# 6. Utils
echo "Fixing utils/__init__.py..."
cat > "$SRC_DIR/utils/__init__.py" << 'EOF'
"""Utils Module"""
from .device import setup_device
from .logging import setup_logging, get_logger
from .reproducibility import set_all_seeds
from .memory import cleanup_memory
EOF

# 7. Models
echo "Fixing models/__init__.py..."
cat > "$SRC_DIR/models/__init__.py" << 'EOF'
"""Models Module"""
try:
    from .gloner import GLONER
except:
    pass
EOF

# 8. Selection
echo "Fixing selection/__init__.py..."
cat > "$SRC_DIR/selection/__init__.py" << 'EOF'
"""Selection Module"""
from .strategies import get_lowest_score_examples_sorted
EOF

# 9. LLM Backends (keep minimal)
echo "Fixing llm_backends/__init__.py..."
cat > "$SRC_DIR/llm_backends/__init__.py" << 'EOF'
"""LLM Backends Module"""
try:
    from .factory import BackendFactory
except:
    pass
EOF

# 10. Prompting (keep minimal)
echo "Fixing prompting/__init__.py..."
cat > "$SRC_DIR/prompting/__init__.py" << 'EOF'
"""Prompting Module"""
EOF

# 11. Parsing (keep minimal)
echo "Fixing parsing/__init__.py..."
cat > "$SRC_DIR/parsing/__init__.py" << 'EOF'
"""Parsing Module"""
EOF

# 12. Caching (keep minimal)
echo "Fixing caching/__init__.py..."
cat > "$SRC_DIR/caching/__init__.py" << 'EOF'
"""Caching Module"""
EOF

echo "✅ All __init__.py files fixed!"
echo ""
echo "To verify, run:"
echo "  cd test"
echo "  uv run python3 -c 'from config import Settings; print(\"OK\")'

"
