"""Test where architecture complexity breaks approximate multipliers"""
from tensorflow import keras
import tensorflow as tf
from model_builder import build_model
from operations import get_search_space
from dataloader import load_dataset
from training import evaluate_model

print(f"TensorFlow version: {tf.__version__}")

# Load MNIST
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('mnist')
x_train, y_train = x_train[:5000], y_train[:5000]
x_val, y_val = x_val[:1000], y_val[:1000]

# Expand dimensions
if len(x_train.shape) == 3:
    x_train = x_train[..., None]
    x_val = x_val[..., None]

input_shape, num_classes = x_train.shape[1:], 10

# Test multiplier
test_mul = './multipliers/mul8u_2P7.bin'

# Define test architectures with increasing complexity
architectures = {
    "2-block (WORKING)": [
        {'op': 'conv3x3', 'filters': 128, 'use_bn': False},
        {'op': 'conv5x5', 'filters': 128, 'use_bn': True}
    ],
    "3-block (add conv3x3)": [
        {'op': 'conv3x3', 'filters': 64, 'use_bn': False},
        {'op': 'conv3x3', 'filters': 128, 'use_bn': False},
        {'op': 'conv5x5', 'filters': 128, 'use_bn': True}
    ],
    "3-block (add max_pool)": [
        {'op': 'conv3x3', 'filters': 128, 'use_bn': False},
        {'op': 'max_pool', 'filters': 128, 'use_bn': False},
        {'op': 'conv5x5', 'filters': 128, 'use_bn': True}
    ],
    "3-block (add conv1x1)": [
        {'op': 'conv3x3', 'filters': 128, 'use_bn': False},
        {'op': 'conv5x5', 'filters': 128, 'use_bn': True},
        {'op': 'conv1x1', 'filters': 64, 'use_bn': False}
    ],
    "4-block (complex - failing)": [
        {'op': 'conv3x3', 'filters': 64, 'use_bn': True},
        {'op': 'max_pool', 'filters': 64, 'use_bn': False},
        {'op': 'conv5x5', 'filters': 128, 'use_bn': True},
        {'op': 'conv1x1', 'filters': 64, 'use_bn': False}
    ],
}

print("=" * 80)
print("Testing architectures with increasing complexity")
print("=" * 80)

for name, arch in architectures.items():
    print(f"\n{'=' * 80}")
    print(f"Testing: {name}")
    print(f"Architecture: {arch}")
    print("=" * 80)

    # Build and train standard model
    search_space_std = get_search_space(use_approximate=False, include_advanced=True)
    model_std = build_model(arch, search_space_std, input_shape, num_classes, learning_rate=0.001)

    model_std.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=20, batch_size=128, verbose=0)

    std_acc = evaluate_model(model_std, x_val, y_val)
    trained_weights = model_std.get_weights()

    # Build approximate model
    search_space_approx = get_search_space(use_approximate=True, mul_map_file=test_mul, include_advanced=True)
    model_approx = build_model(arch, search_space_approx, input_shape, num_classes, learning_rate=0.001)

    try:
        model_approx.set_weights(trained_weights)
        approx_acc = evaluate_model(model_approx, x_val, y_val)
        drop_pct = ((std_acc - approx_acc) / std_acc) * 100

        status = "✓ WORKS" if drop_pct < 10 else "✗ FAILS"
        print(f"\n{status}")
        print(f"  Standard:    {std_acc:.4f}")
        print(f"  Approximate: {approx_acc:.4f}")
        print(f"  Drop:        {drop_pct:6.2f}%")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")

    # Cleanup
    del model_std, model_approx
    keras.backend.clear_session()

print(f"\n{'=' * 80}")
print("Test complete - check which architectures work and which fail")
print("=" * 80)
