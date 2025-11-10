"""Test ALL multipliers with simple 2-block Nov 5 architecture"""
from tensorflow import keras
import tensorflow as tf
import numpy as np
import glob
from model_builder import build_model
from operations import get_search_space
from dataloader import load_dataset
from training import evaluate_model, train_model

print(f"TensorFlow version: {tf.__version__}")

# Load MNIST
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('mnist')
# Use subset
x_train = x_train[:5000]
y_train = y_train[:5000]
x_val = x_val[:1000]
y_val = y_val[:1000]

# Expand dimensions if needed
if len(x_train.shape) == 3:
    x_train = x_train[..., None]
    x_val = x_val[..., None]
    x_test = x_test[..., None]

input_shape = x_train.shape[1:]
num_classes = 10

# EXACT Nov 5 working architecture
arch = [
    {'op': 'conv3x3', 'filters': 128, 'use_bn': False},
    {'op': 'conv5x5', 'filters': 128, 'use_bn': True}
]

print("=" * 70)
print("Training standard model with Nov 5 architecture...")
print(f"Architecture: {arch}")

# Build and train standard model
search_space_std = get_search_space(use_approximate=False, include_advanced=True)
model_std = build_model(arch, search_space_std, input_shape, num_classes, learning_rate=0.001)

history = model_std.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=128,
    verbose=0
)

std_accuracy = evaluate_model(model_std, x_val, y_val)
print(f"✓ Standard model validation accuracy: {std_accuracy:.4f}")

# Save weights
trained_weights = model_std.get_weights()

# Find all multipliers
multiplier_files = sorted(glob.glob('./multipliers/*.bin'))
print(f"\n{'=' * 70}")
print(f"Testing {len(multiplier_files)} multipliers...")
print(f"{'=' * 70}\n")

results = []
for idx, mul_file in enumerate(multiplier_files):
    mul_name = mul_file.split('/')[-1]

    # Build approximate model
    search_space_approx = get_search_space(
        use_approximate=True,
        mul_map_file=mul_file,
        include_advanced=True
    )

    model_approx = build_model(arch, search_space_approx, input_shape, num_classes, learning_rate=0.001)

    # Transfer weights
    try:
        model_approx.set_weights(trained_weights)
    except Exception as e:
        print(f"[{idx+1}/{len(multiplier_files)}] {mul_name}: ✗ Weight transfer failed: {e}")
        continue

    # Evaluate
    approx_accuracy = evaluate_model(model_approx, x_val, y_val)
    drop = std_accuracy - approx_accuracy
    drop_pct = (drop / std_accuracy) * 100

    results.append({
        'name': mul_name,
        'accuracy': approx_accuracy,
        'drop': drop,
        'drop_pct': drop_pct
    })

    print(f"[{idx+1}/{len(multiplier_files)}] {mul_name:20s}  Acc: {approx_accuracy:.4f}  Drop: {drop_pct:6.2f}%")

    # Cleanup
    del model_approx
    keras.backend.clear_session()

# Summary
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"Standard accuracy: {std_accuracy:.4f}\n")

# Sort by drop
results_sorted = sorted(results, key=lambda x: x['drop_pct'])

print("Best 10 multipliers:")
for i, r in enumerate(results_sorted[:10]):
    print(f"{i+1:2d}. {r['name']:20s}  Acc: {r['accuracy']:.4f}  Drop: {r['drop_pct']:6.2f}%")

print("\nWorst 10 multipliers:")
for i, r in enumerate(results_sorted[-10:]):
    print(f"{i+1:2d}. {r['name']:20s}  Acc: {r['accuracy']:.4f}  Drop: {r['drop_pct']:6.2f}%")

# Statistics
drops = [r['drop_pct'] for r in results]
excellent = sum(1 for d in drops if d <= 1)
good = sum(1 for d in drops if 1 < d <= 5)
medium = sum(1 for d in drops if 5 < d <= 10)
poor = sum(1 for d in drops if d > 10)

print(f"\nQuality distribution:")
print(f"  Excellent (≤1% drop):   {excellent}")
print(f"  Good (1-5% drop):       {good}")
print(f"  Medium (5-10% drop):    {medium}")
print(f"  Poor (>10% drop):       {poor}")
