#!/usr/bin/env python3
"""
Understanding the correct tf-approximate workflow
Based on fake_approx_eval.py and fake_approx_train.py examples
"""

from tensorflow import keras
from operations import get_search_space
from model_builder import build_model
from dataloader import load_dataset
import numpy as np

print("="*70)
print("CORRECT TF-APPROXIMATE WORKFLOW TEST")
print("="*70)

# Load data
print("\n1. Loading data...")
x_train, y_train, x_val, y_val, _, _ = load_dataset('cifar10', num_val=6000)
if x_train.ndim == 3:
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)

x_train = x_train[:2000]
y_train = y_train[:2000]
x_val = x_val[:500]
y_val = y_val[:500]

architecture = [
    {'op': 'conv3x3', 'filters': 32, 'use_bn': True},
    {'op': 'conv3x3', 'filters': 64, 'use_bn': True},
]

# STEP 1: Train STANDARD model
print("\n2. Training STANDARD model...")
search_space_std = get_search_space(use_approximate=False, include_advanced=True)
model_std = build_model(architecture, search_space_std, 
                       input_shape=(32, 32, 3), num_classes=10)

model_std.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=10, batch_size=128, verbose=1)

_, std_acc = model_std.evaluate(x_val, y_val, verbose=0)
print(f"\nStandard model accuracy: {std_acc:.4f}")

# STEP 2: Save model weights
print("\n3. Saving model weights...")
model_std.save_weights('/tmp/standard_model.h5')
print("   Weights saved to /tmp/standard_model.h5")

# STEP 3: Build APPROXIMATE model (same architecture)
print("\n4. Building APPROXIMATE model with mul8u_2P7...")
search_space_app = get_search_space(use_approximate=True, 
                                   mul_map_file='./multipliers/mul8u_2P7.bin',
                                   include_advanced=True)
model_app = build_model(architecture, search_space_app,
                       input_shape=(32, 32, 3), num_classes=10)

# STEP 4: Load weights into approximate model
print("\n5. Loading weights into approximate model...")
try:
    model_app.load_weights('/tmp/standard_model.h5')
    print("   ✓ Weights loaded successfully using load_weights()")
    
    # STEP 5: Evaluate with approximate multiplier
    print("\n6. Evaluating with approximate multiplier...")
    _, app_acc = model_app.evaluate(x_val, y_val, verbose=0)
    print(f"   Approximate model accuracy: {app_acc:.4f}")
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"Standard accuracy:     {std_acc:.4f}")
    print(f"Approximate accuracy:  {app_acc:.4f}")
    print(f"Accuracy drop:         {(std_acc - app_acc):.4f} ({((std_acc - app_acc)/std_acc*100):.2f}%)")
    
    if abs(std_acc - app_acc) < 0.10:
        print("\n✓ SUCCESS: load_weights() approach works!")
        print("This is the correct workflow for tf-approximate.")
    else:
        print("\n✗ ISSUE: Large accuracy drop")
        print("May need to investigate further")
        
except Exception as e:
    print(f"   ✗ load_weights() failed: {e}")
    print("\n   Trying alternative: model.set_weights()")
    
    # Try alternative approach
    try:
        weights = model_std.get_weights()
        model_app.set_weights(weights)
        print("   ✓ set_weights() succeeded")
        
        _, app_acc = model_app.evaluate(x_val, y_val, verbose=0)
        print(f"   Approximate model accuracy: {app_acc:.4f}")
        
        print("\n" + "="*70)
        print("RESULTS:")
        print("="*70)
        print(f"Standard accuracy:     {std_acc:.4f}")
        print(f"Approximate accuracy:  {app_acc:.4f}")
        print(f"Accuracy drop:         {(std_acc - app_acc):.4f}")
        
    except Exception as e2:
        print(f"   ✗ set_weights() also failed: {e2}")

print("="*70)
