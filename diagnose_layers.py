###!/usr/bin/env python3
"""
Diagnose layer structure differences between standard and approximate models
"""

from tensorflow import keras
from operations import get_search_space
from model_builder import build_model
import numpy as np

print("="*70)
print("LAYER STRUCTURE DIAGNOSIS")
print("="*70)

# Simple architecture
architecture = [
    {'op': 'conv3x3', 'filters': 32, 'use_bn': True},
    {'op': 'conv3x3', 'filters': 64, 'use_bn': True},
]

input_shape = (32, 32, 3)
num_classes = 10

print("\n1. Building STANDARD model...")
search_space_std = get_search_space(use_approximate=False, include_advanced=True)
model_std = build_model(architecture, search_space_std, 
                       input_shape=input_shape, num_classes=num_classes)

print("\n2. Building APPROXIMATE model...")
search_space_app = get_search_space(use_approximate=True, 
                                   mul_map_file='./multipliers/mul8u_2P7.bin',
                                   include_advanced=True)
model_app = build_model(architecture, search_space_app,
                       input_shape=input_shape, num_classes=num_classes)

print("\n3. Comparing layer structures...")
print("\nSTANDARD MODEL:")
print(f"  Total layers: {len(model_std.layers)}")
for i, layer in enumerate(model_std.layers):
    weights = layer.get_weights()
    if len(weights) > 0:
        print(f"  [{i}] {layer.__class__.__name__:20s} {layer.name:30s} Weights: {len(weights)} shapes: {[w.shape for w in weights]}")

print("\nAPPROXIMATE MODEL:")
print(f"  Total layers: {len(model_app.layers)}")
for i, layer in enumerate(model_app.layers):
    weights = layer.get_weights()
    if len(weights) > 0:
        print(f"  [{i}] {layer.__class__.__name__:20s} {layer.name:30s} Weights: {len(weights)} shapes: {[w.shape for w in weights]}")

print("\n4. Extracting trainable layers...")
std_layers = [l for l in model_std.layers if len(l.get_weights()) > 0]
app_layers = [l for l in model_app.layers if len(l.get_weights()) > 0]

print(f"\nStandard trainable layers: {len(std_layers)}")
print(f"Approximate trainable layers: {len(app_layers)}")

if len(std_layers) != len(app_layers):
    print("\n✗ MISMATCH: Different number of trainable layers!")
    print("\nThis is the root cause of the weight transfer failure.")
    print("FakeApproxConv2D likely has a different structure than Conv2D.")
else:
    print("\n✓ Layer count matches")
    print("\n5. Checking weight shapes...")
    
    for i, (std_layer, app_layer) in enumerate(zip(std_layers, app_layers)):
        std_weights = std_layer.get_weights()
        app_weights = app_layer.get_weights()
        
        print(f"\nLayer pair {i}:")
        print(f"  Standard:    {std_layer.__class__.__name__} - {len(std_weights)} weights")
        print(f"  Approximate: {app_layer.__class__.__name__} - {len(app_weights)} weights")
        
        if len(std_weights) != len(app_weights):
            print(f"  ✗ Different weight counts!")
        else:
            for j, (sw, aw) in enumerate(zip(std_weights, app_weights)):
                if sw.shape != aw.shape:
                    print(f"    Weight {j}: {sw.shape} vs {aw.shape} ✗ MISMATCH")
                else:
                    print(f"    Weight {j}: {sw.shape} ✓")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
