"""
Diagnostic script to verify FakeApproxConv2D is working correctly
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Try to import FakeApproxConv2D
try:
    from keras.layers.fake_approx_convolutional import FakeApproxConv2D
    print("✓ FakeApproxConv2D imported successfully")
except ImportError as e:
    print(f"✗ Failed to import FakeApproxConv2D: {e}")
    exit(1)

def test_multiplier_basic():
    """Test 1: Basic functionality"""
    print("\n" + "="*70)
    print("TEST 1: Basic FakeApproxConv2D Functionality")
    print("="*70)
    
    # Create a simple input
    x_test = np.random.randn(1, 32, 32, 3).astype('float32')
    
    # Standard Conv2D
    layer_std = keras.layers.Conv2D(16, 3, padding='same', activation='relu')
    output_std = layer_std(x_test)
    
    # Approximate Conv2D
    mul_file = './multipliers/mul8u_1JFF.bin'
    if not os.path.exists(mul_file):
        print(f"✗ Multiplier file not found: {mul_file}")
        return False
    
    layer_approx = FakeApproxConv2D(16, 3, padding='same', activation='relu', 
                                    mul_map_file=mul_file)
    output_approx = layer_approx(x_test)
    
    print(f"Standard output shape: {output_std.shape}")
    print(f"Approx output shape:   {output_approx.shape}")
    print(f"Standard output mean:  {np.mean(output_std):.4f}")
    print(f"Approx output mean:    {np.mean(output_approx):.4f}")
    
    if output_std.shape != output_approx.shape:
        print("✗ FAILED: Shape mismatch!")
        return False
    
    print("✓ PASSED: Basic functionality works")
    return True

def test_weight_transfer():
    """Test 2: Weight transfer between standard and approximate"""
    print("\n" + "="*70)
    print("TEST 2: Weight Transfer")
    print("="*70)
    
    # Create simple models
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Standard model
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model_std = keras.Model(inputs, x)
    
    # Approximate model
    mul_file = './multipliers/mul8u_1JFF.bin'
    x = FakeApproxConv2D(32, 3, padding='same', activation='relu', mul_map_file=mul_file)(inputs)
    x = FakeApproxConv2D(64, 3, padding='same', activation='relu', mul_map_file=mul_file)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model_approx = keras.Model(inputs, x)
    
    # Initialize weights in standard model
    _ = model_std(np.random.randn(1, 32, 32, 3).astype('float32'))
    _ = model_approx(np.random.randn(1, 32, 32, 3).astype('float32'))
    
    # Get weights
    weights_std = model_std.get_weights()
    
    print(f"Standard model has {len(weights_std)} weight arrays")
    print(f"Approximate model has {len(model_approx.get_weights())} weight arrays")
    
    # Try to copy weights
    try:
        model_approx.set_weights(weights_std)
        print("✓ PASSED: Weights copied successfully")
        
        # Verify weights are actually the same
        weights_approx = model_approx.get_weights()
        all_match = True
        for i, (w_std, w_approx) in enumerate(zip(weights_std, weights_approx)):
            if not np.allclose(w_std, w_approx):
                print(f"✗ Weight array {i} doesn't match!")
                all_match = False
        
        if all_match:
            print("✓ PASSED: All weights match after transfer")
        else:
            print("✗ FAILED: Some weights don't match")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: Could not copy weights: {e}")
        return False
    
    return True

def test_inference_difference():
    """Test 3: Check if approximate actually differs from standard"""
    print("\n" + "="*70)
    print("TEST 3: Inference Difference (Approximate vs Standard)")
    print("="*70)
    
    # Create test input
    x_test = np.random.randn(10, 32, 32, 3).astype('float32')
    
    # Build models
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Standard
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model_std = keras.Model(inputs, x)
    
    # Approximate
    mul_file = './multipliers/mul8u_1JFF.bin'
    x = FakeApproxConv2D(32, 3, padding='same', activation='relu', mul_map_file=mul_file)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model_approx = keras.Model(inputs, x)
    
    # Initialize and copy weights
    _ = model_std(x_test[:1])
    _ = model_approx(x_test[:1])
    model_approx.set_weights(model_std.get_weights())
    
    # Get predictions
    pred_std = model_std.predict(x_test, verbose=0)
    pred_approx = model_approx.predict(x_test, verbose=0)
    
    print(f"Standard predictions shape: {pred_std.shape}")
    print(f"Approx predictions shape:   {pred_approx.shape}")
    print(f"\nStandard prediction sample: {pred_std[0][:5]}")
    print(f"Approx prediction sample:   {pred_approx[0][:5]}")
    
    # Calculate difference
    diff = np.abs(pred_std - pred_approx)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"\nMean absolute difference: {mean_diff:.6f}")
    print(f"Max absolute difference:  {max_diff:.6f}")
    
    if mean_diff < 1e-6:
        print("✗ FAILED: No difference detected - multiplier not being used!")
        return False
    elif mean_diff > 0.5:
        print("✗ WARNING: Very large difference - may indicate problem")
    else:
        print("✓ PASSED: Reasonable difference detected")
    
    return True

def test_actual_cifar10():
    """Test 4: Actual CIFAR-10 test"""
    print("\n" + "="*70)
    print("TEST 4: CIFAR-10 Small Test")
    print("="*70)
    
    # Load small CIFAR-10 subset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train[:1000].astype('float32') / 255.0
    y_train = y_train[:1000].flatten()
    x_test = x_test[:200].astype('float32') / 255.0
    y_test = y_test[:200].flatten()
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Build simple model
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model_std = keras.Model(inputs, x)
    
    model_std.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train briefly
    print("\nTraining standard model...")
    model_std.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Evaluate standard
    _, std_acc = model_std.evaluate(x_test, y_test, verbose=0)
    print(f"Standard accuracy: {std_acc:.4f}")
    
    if std_acc < 0.3:
        print("✗ WARNING: Standard model accuracy is very low")
    
    # Build approximate model
    mul_file = './multipliers/mul8u_1JFF.bin'
    x = FakeApproxConv2D(32, 3, padding='same', activation='relu', mul_map_file=mul_file)(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = FakeApproxConv2D(64, 3, padding='same', activation='relu', mul_map_file=mul_file)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model_approx = keras.Model(inputs, x)
    
    model_approx.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Copy weights
    model_approx.set_weights(model_std.get_weights())
    
    # Evaluate approximate
    _, approx_acc = model_approx.evaluate(x_test, y_test, verbose=0)
    print(f"Approximate accuracy: {approx_acc:.4f}")
    print(f"Accuracy drop: {(std_acc - approx_acc):.4f} ({(std_acc - approx_acc)/std_acc*100:.2f}%)")
    
    if approx_acc < 0.15:
        print("✗ FAILED: Approximate model completely collapsed!")
        return False
    elif std_acc - approx_acc > 0.2:
        print("✗ WARNING: Very large accuracy drop")
    else:
        print("✓ PASSED: Reasonable accuracy maintained")
    
    return True

def main():
    print("="*70)
    print("APPROXIMATE MULTIPLIER DIAGNOSTIC TESTS")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Basic Functionality", test_multiplier_basic()))
    results.append(("Weight Transfer", test_weight_transfer()))
    results.append(("Inference Difference", test_inference_difference()))
    results.append(("CIFAR-10 Test", test_actual_cifar10()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Multiplier code is working correctly")
        print("  → The issue is likely model weakness, not code bugs")
    else:
        print("✗ SOME TESTS FAILED - There may be issues with the multiplier code")
    print("="*70)

if __name__ == '__main__':
    main()