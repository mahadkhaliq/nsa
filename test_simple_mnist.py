"""Test with simple 2-block architecture matching Nov 5 working version"""
from tensorflow import keras
import tensorflow as tf
import numpy as np
from model_builder import build_model
from operations import get_search_space
from dataloader import load_dataset
from training import evaluate_model, train_model

print(f"TensorFlow version: {tf.__version__}")

# Load MNIST
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset('mnist', 5000, 1000)
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
    verbose=1
)

std_accuracy = evaluate_model(model_std, x_val, y_val)
print(f"\n✓ Standard model validation accuracy: {std_accuracy:.4f}")

# Save weights
trained_weights = model_std.get_weights()

# Test with one multiplier
print("\n" + "=" * 70)
print("Testing with approximate multiplier: mul8u_2P7.bin")

search_space_approx = get_search_space(
    use_approximate=True,
    mul_map_file='./multipliers/mul8u_2P7.bin',
    include_advanced=True
)

model_approx = build_model(arch, search_space_approx, input_shape, num_classes, learning_rate=0.001)

# Transfer weights
try:
    model_approx.set_weights(trained_weights)
    print("✓ Weights transferred successfully")
except Exception as e:
    print(f"✗ Weight transfer failed: {e}")
    exit(1)

# Evaluate
approx_accuracy = evaluate_model(model_approx, x_val, y_val)
drop = std_accuracy - approx_accuracy
drop_pct = (drop / std_accuracy) * 100

print(f"\nResults:")
print(f"  Standard accuracy:    {std_accuracy:.4f}")
print(f"  Approximate accuracy: {approx_accuracy:.4f}")
print(f"  Drop:                 {drop:.4f} ({drop_pct:.2f}%)")

if drop_pct < 5:
    print("\n✓ SUCCESS: Multiplier shows good performance!")
elif drop_pct < 50:
    print("\n⚠ PARTIAL: Multiplier shows degradation but not total collapse")
else:
    print("\n✗ FAILURE: Multiplier shows severe degradation")
