#!/bin/bash
# Run this script in your Docker/SSH environment to fix all import issues
# Usage: bash FIX_FOR_DOCKER.sh

echo "🔧 Fixing imports for Docker/SSH environment..."
echo ""

# Step 1: Empty all __init__.py files
echo "Step 1: Emptying all __init__.py files..."
find src -name "__init__.py" -type f -exec sh -c 'echo "" > "$1"' _ {} \;
echo "✅ All __init__.py files emptied"
echo ""

# Step 2: Clear Python cache
echo "Step 2: Clearing Python cache..."
find src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find test -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✅ Python cache cleared"
echo ""

# Step 3: Verify test file exists
echo "Step 3: Verifying test files..."
if [ -f "test/confidence_test_FT.py" ]; then
    echo "✅ confidence_test_FT.py found"
else
    echo "❌ test/confidence_test_FT.py not found!"
    echo "   Are you in the project root directory?"
    exit 1
fi
echo ""

# Step 4: Test imports
echo "Step 4: Testing imports..."
cd test
if uv run python3 -c "from config.settings import Settings; print('✓ Config OK')" 2>&1 | grep -q "Config OK"; then
    echo "✅ Config imports working"
else
    echo "❌ Config imports failed"
    exit 1
fi

if uv run python3 -c "from models.gloner import GLONER; print('✓ Models OK')" 2>&1 | grep -q "Models OK"; then
    echo "✅ Models imports working"
else
    echo "❌ Models imports failed"
    exit 1
fi

if uv run python3 -c "from generation.label_generator import create_label_generator; print('✓ Generation OK')" 2>&1 | grep -q "Generation OK"; then
    echo "✅ Generation imports working"
else
    echo "❌ Generation imports failed"
    exit 1
fi

echo ""
echo "🎉 All imports fixed and working!"
echo ""
echo "You can now run:"
echo "  cd test"
echo "  uv run confidence_test_FT.py"
