"""Test the two-stage training pattern matching reference implementation"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from model_builder import build_model
from operations import get_search_space
from dataloader import load_dataset
from training import evaluate_model

print(f"TensorFlow version: {tf.__version__}")

# Load MNIST subset
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('mnist')
x_train, y_train = x_train[:5000], y_train[:5000]
x_val, y_val = x_val[:1000], y_val[:1000]

if len(x_train.shape) == 3:
    x_train = x_train[..., None]
    x_val = x_val[..., None]

input_shape, num_classes = x_train.shape[1:], 10
test_mul = './multipliers/mul8u_2P7.bin'

# Simple 2-block architecture
arch = [
    {'op': 'conv3x3', 'filters': 128, 'use_bn': False},
    {'op': 'conv5x5', 'filters': 128, 'use_bn': True}
]

print("=" * 80)
print("TWO-STAGE TRAINING PATTERN TEST (matching reference implementation)")
print("=" * 80)

# ==============================================================================
# STAGE 1: Train with STANDARD Conv2D
# ==============================================================================
print("\n→ STAGE 1: Training with STANDARD Conv2D layers...")
search_space_std = get_search_space(use_approximate=False, include_advanced=True)
model_std = build_model(arch, search_space_std, input_shape, num_classes, learning_rate=0.001)

print(f"  Model architecture (standard):")
print(f"  Total layers: {len(model_std.layers)}")
for i, layer in enumerate(model_std.layers):
    print(f"    {i}: {layer.__class__.__name__} {layer.name}")

# Train
history = model_std.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=128,
    verbose=0
)

std_accuracy = evaluate_model(model_std, x_val, y_val)
print(f"\n✓ Standard model validation accuracy: {std_accuracy:.4f}")

# Save trained weights
trained_weights = model_std.get_weights()
print(f"✓ Saved {len(trained_weights)} weight arrays")

# ==============================================================================
# STAGE 2: Build NEW model with FakeApproxConv2D, transfer weights
# ==============================================================================
print("\n→ STAGE 2: Building NEW model with FakeApproxConv2D...")
search_space_approx = get_search_space(
    use_approximate=True,
    mul_map_file=test_mul,
    include_advanced=True
)
model_approx = build_model(arch, search_space_approx, input_shape, num_classes, learning_rate=0.001)

print(f"  Model architecture (approximate):")
print(f"  Total layers: {len(model_approx.layers)}")
for i, layer in enumerate(model_approx.layers):
    print(f"    {i}: {layer.__class__.__name__} {layer.name}")

# Transfer weights
print("\n→ Transferring weights from standard to approximate model...")
try:
    model_approx.set_weights(trained_weights)
    print("✓ Weight transfer successful")
except Exception as e:
    print(f"✗ Weight transfer failed: {e}")
    print("\nDEBUG: Weight shapes comparison:")
    std_shapes = [w.shape for w in trained_weights]
    approx_shapes = [w.shape for w in model_approx.get_weights()]
    for i, (s_shape, a_shape) in enumerate(zip(std_shapes, approx_shapes)):
        match = "✓" if s_shape == a_shape else "✗"
        print(f"  {match} Weight {i}: std={s_shape}, approx={a_shape}")
    exit(1)

# Evaluate approximate model
approx_accuracy = evaluate_model(model_approx, x_val, y_val)
drop = std_accuracy - approx_accuracy
drop_pct = (drop / std_accuracy) * 100

print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)
print(f"  Standard accuracy:    {std_accuracy:.4f}")
print(f"  Approximate accuracy: {approx_accuracy:.4f}")
print(f"  Drop:                 {drop:.4f} ({drop_pct:.2f}%)")

if drop_pct < 5:
    print("\n✓ SUCCESS: Two-stage pattern works correctly!")
elif drop_pct < 50:
    print("\n⚠ PARTIAL: Some degradation but not total collapse")
else:
    print("\n✗ FAILURE: Severe degradation indicates architecture mismatch")

# ==============================================================================
# VERIFY: Predictions are different (not identical)
# ==============================================================================
print("\n→ Verifying models produce different predictions...")
sample = x_val[:10]
std_pred = model_std.predict(sample, verbose=0)
approx_pred = model_approx.predict(sample, verbose=0)

pred_diff = np.abs(std_pred - approx_pred).mean()
print(f"  Mean prediction difference: {pred_diff:.6f}")

if pred_diff > 0.001:
    print("✓ Models produce different predictions (approximate multiplier is active)")
else:
    print("⚠ Models produce nearly identical predictions (approximate might not be working)")
