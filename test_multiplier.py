"""
GPU-based diagnostic to test if multipliers are actually different
Proper memory management for GPU
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob

# Proper GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"✓ GPU available: {physical_devices[0]}")
    except:
        pass
else:
    print("⚠ No GPU detected")

try:
    from keras.layers.fake_approx_convolutional import FakeApproxConv2D
    print("✓ FakeApproxConv2D imported successfully\n")
except ImportError as e:
    print(f"✗ Failed to import FakeApproxConv2D: {e}")
    exit(1)

print("="*70)
print("MULTIPLIER DIFFERENTIATION TEST (GPU)")
print("="*70)

# Load multiplier files
mul_dir = './multipliers'
mul_files = sorted(glob.glob(os.path.join(mul_dir, '*.bin')))[:10]  # Test first 10

if not mul_files:
    print(f"✗ No multiplier files found in {mul_dir}")
    exit(1)

print(f"Testing {len(mul_files)} multipliers\n")

# Load real CIFAR-10 data
print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train[:1000].astype('float32') / 255.0
y_train = y_train[:1000].flatten()
x_test = x_test[:500].astype('float32') / 255.0
y_test = y_test[:500].flatten()

print(f"Using {len(x_train)} train, {len(x_test)} test samples\n")

# ============================================================
# TEST 1: Are multipliers actually being loaded?
# ============================================================
print("="*70)
print("TEST 1: Multiplier File Loading")
print("="*70)

for i, mul_file in enumerate(mul_files[:3]):
    size = os.path.getsize(mul_file)
    name = os.path.basename(mul_file)
    
    # Try to create layer with this multiplier
    try:
        test_input = tf.constant(np.random.randn(1, 32, 32, 3).astype('float32'))
        layer = FakeApproxConv2D(16, 3, padding='same', mul_map_file=mul_file)
        output = layer(test_input)
        
        print(f"✓ {name} - Loaded and executed")
        print(f"    File size: {size} bytes, Output mean: {float(tf.reduce_mean(output)):.6f}")
        
        # Cleanup
        del layer, output
        keras.backend.clear_session()
        
    except Exception as e:
        print(f"✗ {name} - Failed: {e}")

print()

# ============================================================
# TEST 2: Do different multipliers produce different outputs?
# ============================================================
print("="*70)
print("TEST 2: Output Differentiation Test")
print("="*70)

test_batch = x_test[:10]  # Use real CIFAR images

# Get standard output
print("Creating standard model...")
inputs = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model_std = keras.Model(inputs, outputs)

pred_std = model_std.predict(test_batch, verbose=0)
print(f"Standard model predictions: {pred_std[0]}")
print()

# Test each multiplier
results = []
for mul_file in mul_files:
    name = os.path.basename(mul_file)
    
    # Build model with this multiplier
    inputs = keras.Input(shape=(32, 32, 3))
    x = FakeApproxConv2D(32, 3, padding='same', activation='relu', mul_map_file=mul_file)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    model_approx = keras.Model(inputs, outputs)
    
    # Copy weights from standard
    model_approx.set_weights(model_std.get_weights())
    
    # Predict
    pred_approx = model_approx.predict(test_batch, verbose=0)
    
    # Calculate difference from standard
    diff = float(np.mean(np.abs(pred_std - pred_approx)))
    
    print(f"{name:<25} Diff from standard: {diff:.6f}")
    results.append((name, diff))
    
    # Cleanup
    del model_approx
    keras.backend.clear_session()

print()

# Check if all diffs are the same (BAD) or different (GOOD)
diffs = [r[1] for r in results]
unique_diffs = len(set(f"{d:.5f}" for d in diffs))

if unique_diffs == 1:
    print("✗ CRITICAL: ALL multipliers produce IDENTICAL outputs!")
    print("  → Multipliers are NOT being used correctly!")
else:
    print(f"✓ Found {unique_diffs} different output patterns")
    print("  → Multipliers ARE being applied differently")

print()

# Cleanup
del model_std
keras.backend.clear_session()

# ============================================================
# TEST 3: Trained model with multipliers
# ============================================================
print("="*70)
print("TEST 3: Trained Model Performance")
print("="*70)

# Train a standard model
print("Training standard model...")
model_std = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation='softmax')
])

model_std.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_std.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)

_, std_acc = model_std.evaluate(x_test, y_test, verbose=0)
print(f"✓ Standard accuracy: {std_acc:.4f}\n")

if std_acc < 0.3:
    print("⚠ WARNING: Standard model accuracy is very low (<30%)")
    print("  → Weak model will be sensitive to approximation errors\n")

# Save weights
trained_weights = model_std.get_weights()
del model_std
keras.backend.clear_session()

# Test with multiple multipliers
print("Testing multipliers:")
mul_results = []

for mul_file in mul_files:
    name = os.path.basename(mul_file)
    
    # Build approximate model
    model_approx = keras.Sequential([
        FakeApproxConv2D(32, 3, padding='same', activation='relu', mul_map_file=mul_file, input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        FakeApproxConv2D(64, 3, padding='same', activation='relu', mul_map_file=mul_file),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model_approx.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Copy trained weights
    model_approx.set_weights(trained_weights)
    
    # Evaluate
    _, approx_acc = model_approx.evaluate(x_test, y_test, verbose=0)
    drop = std_acc - approx_acc
    drop_pct = (drop / std_acc * 100) if std_acc > 0 else 0
    
    print(f"{name:<25} Acc: {approx_acc:.4f}  Drop: {drop:.4f} ({drop_pct:+.1f}%)")
    mul_results.append((name, approx_acc, drop_pct))
    
    # Cleanup
    del model_approx
    keras.backend.clear_session()

print()

# Analyze results
accuracies = [r[1] for r in mul_results]
unique_accs = len(set(f"{a:.3f}" for a in accuracies))

print("="*70)
print("ANALYSIS")
print("="*70)

if unique_accs == 1:
    print("✗ CRITICAL ISSUE: All multipliers show IDENTICAL accuracy!")
    print(f"  All accuracy values: ~{accuracies[0]:.4f}")
    print("\nPossible causes:")
    print("  1. Multipliers are not being loaded/applied correctly")
    print("  2. FakeApproxConv2D has a bug")
    print("  3. Model collapsed completely with ALL multipliers")
elif max(accuracies) < 0.15:
    print("✗ CRITICAL: Model COLLAPSED with all multipliers!")
    print(f"  Standard: {std_acc:.4f}")
    print(f"  Best approx: {max(accuracies):.4f}")
    print("\nLikely cause:")
    print("  → Model is too weak to tolerate approximation errors")
    print("  → Need stronger baseline (70%+) before testing multipliers")
else:
    print(f"✓ Found {unique_accs} different accuracy levels")
    print(f"  Best multiplier: {max(accuracies):.4f}")
    print(f"  Worst multiplier: {min(accuracies):.4f}")
    print(f"  Range: {max(accuracies) - min(accuracies):.4f}")
    
    if max(accuracies) - min(accuracies) < 0.01:
        print("\n⚠ WARNING: Very small range - multipliers may not be working correctly")

print("\n" + "="*70)
