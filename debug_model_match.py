"""Debug script to check if standard and approximate models match"""
from tensorflow import keras
from model_builder import build_model
from operations import get_search_space
import numpy as np

# Simple test architecture
arch = [
    {'op': 'conv3x3', 'filters': 32, 'use_bn': True},
    {'op': 'conv3x3', 'filters': 64, 'use_bn': False},
]

input_shape = (32, 32, 3)
num_classes = 10

# Build standard model
print("=" * 70)
print("Building STANDARD model...")
search_space_std = get_search_space(use_approximate=False, include_advanced=True)
model_std = build_model(arch, search_space_std, input_shape, num_classes, learning_rate=0.001)
print("\nStandard model summary:")
model_std.summary()

# Build approximate model
print("\n" + "=" * 70)
print("Building APPROXIMATE model...")
search_space_approx = get_search_space(use_approximate=True, mul_map_file='./multipliers/mul8u_2P7.bin', include_advanced=True)
model_approx = build_model(arch, search_space_approx, input_shape, num_classes, learning_rate=0.001)
print("\nApproximate model summary:")
model_approx.summary()

# Check layer counts
print("\n" + "=" * 70)
print("LAYER COMPARISON:")
print(f"Standard model layers: {len(model_std.layers)}")
print(f"Approximate model layers: {len(model_approx.layers)}")

# Check if weight shapes match
print("\n" + "=" * 70)
print("WEIGHT SHAPE COMPARISON:")
std_weights = model_std.get_weights()
approx_weights = model_approx.get_weights()

print(f"Standard weights count: {len(std_weights)}")
print(f"Approximate weights count: {len(approx_weights)}")

if len(std_weights) == len(approx_weights):
    print("\n✓ Weight counts match!")
    all_match = True
    for i, (sw, aw) in enumerate(zip(std_weights, approx_weights)):
        if sw.shape != aw.shape:
            print(f"  ✗ Weight {i}: standard {sw.shape} != approx {aw.shape}")
            all_match = False
    if all_match:
        print("✓ All weight shapes match!")
else:
    print("\n✗ Weight counts don't match - models are incompatible!")

# Test weight transfer
print("\n" + "=" * 70)
print("TESTING WEIGHT TRANSFER:")
try:
    model_approx.set_weights(model_std.get_weights())
    print("✓ Weight transfer successful!")

    # Test inference
    test_input = np.random.randn(1, 32, 32, 3).astype(np.float32)
    std_out = model_std.predict(test_input, verbose=0)
    approx_out = model_approx.predict(test_input, verbose=0)

    print(f"\nStandard output shape: {std_out.shape}")
    print(f"Approximate output shape: {approx_out.shape}")
    print(f"Output difference (should be large due to approx): {np.abs(std_out - approx_out).max():.6f}")

except Exception as e:
    print(f"✗ Weight transfer failed: {e}")
