"""Test if standard and approximate models have matching architectures"""
import sys
from model_builder import build_model
from operations import get_search_space

# Simple test architecture
arch = [
    {'op': 'conv3x3', 'filters': 32, 'use_bn': True},
    {'op': 'conv3x3', 'filters': 64, 'use_bn': False},
]

input_shape = (28, 28, 1)
num_classes = 10

# Build standard model
print("=" * 70)
print("Building STANDARD model...")
search_space_std = get_search_space(use_approximate=False, include_advanced=True)
model_std = build_model(arch, search_space_std, input_shape, num_classes, learning_rate=0.001)

print("\nStandard model layers:")
for i, layer in enumerate(model_std.layers):
    print(f"  {i}: {layer.name} - {layer.__class__.__name__}")
    if hasattr(layer, 'activation') and layer.activation is not None:
        print(f"      activation: {layer.activation}")

# Build approximate model
print("\n" + "=" * 70)
print("Building APPROXIMATE model...")
search_space_approx = get_search_space(use_approximate=True, mul_map_file='./multipliers/mul8u_2P7.bin', include_advanced=True)
model_approx = build_model(arch, search_space_approx, input_shape, num_classes, learning_rate=0.001)

print("\nApproximate model layers:")
for i, layer in enumerate(model_approx.layers):
    print(f"  {i}: {layer.name} - {layer.__class__.__name__}")
    if hasattr(layer, 'activation') and layer.activation is not None:
        print(f"      activation: {layer.activation}")

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
        match = "✓" if sw.shape == aw.shape else "✗"
        print(f"  {match} Weight {i}: standard {sw.shape} vs approx {aw.shape}")
        if sw.shape != aw.shape:
            all_match = False
    if all_match:
        print("\n✓ ALL WEIGHT SHAPES MATCH - Models are compatible!")
    else:
        print("\n✗ SOME SHAPES DON'T MATCH - Models are incompatible!")
else:
    print("\n✗ Weight counts don't match - models are incompatible!")
    print("\nStandard model summary:")
    model_std.summary()
    print("\nApproximate model summary:")
    model_approx.summary()
