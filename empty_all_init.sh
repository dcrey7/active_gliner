#!/bin/bash
# Empty ALL __init__.py files - simplest solution!

SRC_DIR="${1:-src}"

echo "🗑️  Emptying all __init__.py files in $SRC_DIR..."

# Find and empty all __init__.py files
find "$SRC_DIR" -name "__init__.py" -type f -exec sh -c 'echo "" > "$1"' _ {} \;

echo "✅ All __init__.py files are now empty!"
echo ""
echo "Now you MUST use full imports:"
echo "  from config.settings import Settings"
echo "  from data.loader import load_mit_dataset"
echo "  from models.gloner import GLONER"
