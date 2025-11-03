#!/bin/bash
# Smoke tests for Drug Repurposing System
# Tests basic functionality after pipeline execution

echo "=========================================="
echo "Drug Repurposing System - Smoke Tests"
echo "=========================================="

FAILED=0

# Test 1: Check if graph_data.pt exists
echo -e "\n[Test 1] Checking graph_data.pt exists..."
if [ -f "./outputs/graph_data.pt" ]; then
    echo "✓ PASS: graph_data.pt exists"
else
    echo "✗ FAIL: graph_data.pt not found"
    FAILED=$((FAILED + 1))
fi

# Test 2: Check if model exists
echo -e "\n[Test 2] Checking model_best.pt exists..."
if [ -f "./outputs/model_best.pt" ]; then
    echo "✓ PASS: model_best.pt exists"
else
    echo "✗ FAIL: model_best.pt not found"
    FAILED=$((FAILED + 1))
fi

# Test 3: Check if embeddings exist
echo -e "\n[Test 3] Checking embeddings.npy exists..."
if [ -f "./outputs/embeddings.npy" ]; then
    echo "✓ PASS: embeddings.npy exists"
else
    echo "✗ FAIL: embeddings.npy not found"
    FAILED=$((FAILED + 1))
fi

# Test 4: Check if predictions exist
echo -e "\n[Test 4] Checking predictions.csv exists..."
if [ -f "./outputs/predictions.csv" ]; then
    echo "✓ PASS: predictions.csv exists"
    
    # Test 5: Check predictions have at least 5 entries per disease
    echo -e "\n[Test 5] Checking prediction quality..."
    python -c "
import pandas as pd
df = pd.read_csv('./outputs/predictions.csv')
diseases = df['disease_id'].unique()
min_preds = df.groupby('disease_id').size().min()
if len(diseases) > 0 and min_preds >= 5:
    print(f'✓ PASS: Found predictions for {len(diseases)} diseases, min {min_preds} per disease')
    exit(0)
else:
    print(f'✗ FAIL: Insufficient predictions (diseases: {len(diseases)}, min preds: {min_preds})')
    exit(1)
    "
    if [ $? -ne 0 ]; then
        FAILED=$((FAILED + 1))
    fi
else
    echo "✗ FAIL: predictions.csv not found"
    FAILED=$((FAILED + 1))
fi

# Test 6: Test model forward pass
echo -e "\n[Test 6] Testing model forward pass..."
python -c "
import torch
import sys
sys.path.append('.')
from train_gcn import GCN

try:
    # Load model
    data = torch.load('./outputs/graph_data.pt', map_location='cpu')
    input_dim = data.x.shape[1]
    model = GCN(input_dim, 128, 128, 0.5)
    model.load_state_dict(torch.load('./outputs/model_best.pt', map_location='cpu'))
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    
    # Check output shape
    assert embeddings.shape[0] == data.num_nodes, 'Embedding shape mismatch'
    assert embeddings.shape[1] == 128, 'Embedding dimension mismatch'
    
    print(f'✓ PASS: Model forward pass successful (output shape: {embeddings.shape})')
    exit(0)
except Exception as e:
    print(f'✗ FAIL: Model forward pass failed: {e}')
    exit(1)
"
if [ $? -ne 0 ]; then
    FAILED=$((FAILED + 1))
fi

# Test 7: Check plots exist
echo -e "\n[Test 7] Checking plots exist..."
if [ -f "./outputs/plots/roc_curve.png" ] && [ -f "./outputs/plots/pr_curve.png" ]; then
    echo "✓ PASS: Plot files exist"
else
    echo "✗ FAIL: Plot files missing"
    FAILED=$((FAILED + 1))
fi

# Summary
echo -e "\n=========================================="
if [ $FAILED -eq 0 ]; then
    echo "All tests passed! ✓"
    echo "=========================================="
    exit 0
else
    echo "Tests failed: $FAILED"
    echo "=========================================="
    exit 1
fi

