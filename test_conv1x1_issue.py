"""Test if conv1x1 specifically breaks approximate multipliers"""
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

if len(x_train.shape) == 3:
    x_train = x_train[..., None]
    x_val = x_val[..., None]

input_shape, num_classes = x_train.shape[1:], 10
test_mul = './multipliers/mul8u_2P7.bin'

# Test 1: Working baseline (no conv1x1)
print("=" * 80)
print("TEST 1: Baseline WITHOUT conv1x1 (should work)")
print("=" * 80)

arch1 = [
    {'op': 'conv3x3', 'filters': 128, 'use_bn': False},
    {'op': 'conv5x5', 'filters': 128, 'use_bn': True}
]

search_space_std = get_search_space(use_approximate=False, include_advanced=True)
model_std = build_model(arch1, search_space_std, input_shape, num_classes, learning_rate=0.001)
model_std.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=128, verbose=0)
std_acc1 = evaluate_model(model_std, x_val, y_val)
weights1 = model_std.get_weights()

search_space_approx = get_search_space(use_approximate=True, mul_map_file=test_mul, include_advanced=True)
model_approx = build_model(arch1, search_space_approx, input_shape, num_classes, learning_rate=0.001)
model_approx.set_weights(weights1)
approx_acc1 = evaluate_model(model_approx, x_val, y_val)

print(f"Standard:    {std_acc1:.4f}")
print(f"Approximate: {approx_acc1:.4f}")
print(f"Drop:        {((std_acc1 - approx_acc1) / std_acc1 * 100):6.2f}%")

del model_std, model_approx
keras.backend.clear_session()

# Test 2: Add conv1x1 at the END (after other layers)
print("\n" + "=" * 80)
print("TEST 2: Add conv1x1 at the END")
print("=" * 80)

arch2 = [
    {'op': 'conv3x3', 'filters': 128, 'use_bn': False},
    {'op': 'conv5x5', 'filters': 128, 'use_bn': True},
    {'op': 'conv1x1', 'filters': 64, 'use_bn': False}
]

model_std = build_model(arch2, search_space_std, input_shape, num_classes, learning_rate=0.001)
model_std.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=128, verbose=0)
std_acc2 = evaluate_model(model_std, x_val, y_val)
weights2 = model_std.get_weights()

model_approx = build_model(arch2, search_space_approx, input_shape, num_classes, learning_rate=0.001)
model_approx.set_weights(weights2)
approx_acc2 = evaluate_model(model_approx, x_val, y_val)

print(f"Standard:    {std_acc2:.4f}")
print(f"Approximate: {approx_acc2:.4f}")
print(f"Drop:        {((std_acc2 - approx_acc2) / std_acc2 * 100):6.2f}%")

del model_std, model_approx
keras.backend.clear_session()

# Test 3: Conv1x1 ONLY (isolate the operation)
print("\n" + "=" * 80)
print("TEST 3: Single conv1x1 layer ONLY")
print("=" * 80)

arch3 = [
    {'op': 'conv1x1', 'filters': 128, 'use_bn': False}
]

model_std = build_model(arch3, search_space_std, input_shape, num_classes, learning_rate=0.001)
model_std.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=128, verbose=0)
std_acc3 = evaluate_model(model_std, x_val, y_val)
weights3 = model_std.get_weights()

model_approx = build_model(arch3, search_space_approx, input_shape, num_classes, learning_rate=0.001)
model_approx.set_weights(weights3)
approx_acc3 = evaluate_model(model_approx, x_val, y_val)

print(f"Standard:    {std_acc3:.4f}")
print(f"Approximate: {approx_acc3:.4f}")
print(f"Drop:        {((std_acc3 - approx_acc3) / std_acc3 * 100):6.2f}%")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("If Test 2 and/or Test 3 show high drop (>80%), then conv1x1")
print("is incompatible with FakeApproxConv2D and should be excluded.")
