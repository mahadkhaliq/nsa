"""Test with reference implementation architecture pattern"""
import tensorflow as tf
from tensorflow import keras
import glob
from model_builder import build_model_for_training, build_model_for_evaluation
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

# Configuration matching reference implementation
config = {
    'conv1_filters': 32,
    'conv1_kernel': 3,
    'conv1_multiplier': './multipliers/mul8u_2P7.bin',
    'conv2_filters': 64,
    'conv2_kernel': 3,
    'conv2_multiplier': './multipliers/mul8u_2P7.bin',
    'conv3_filters': 128,
    'conv3_kernel': 3,
    'conv3_multiplier': './multipliers/mul8u_2P7.bin',
    'dense1_size': 128,
    'dropout_rate': 0.3
}

print("=" * 80)
print("REFERENCE ARCHITECTURE PATTERN TEST")
print("=" * 80)
print(f"Architecture:")
print(f"  Block 1: {config['conv1_filters']} filters, {config['conv1_kernel']}x{config['conv1_kernel']} kernel")
print(f"  Block 2: {config['conv2_filters']} filters, {config['conv2_kernel']}x{config['conv2_kernel']} kernel")
print(f"  Block 3: {config['conv3_filters']} filters, {config['conv3_kernel']}x{config['conv3_kernel']} kernel")
print(f"  Dense: {config['dense1_size']}, dropout={config['dropout_rate']}")

# ==============================================================================
# STAGE 1: Train with STANDARD Conv2D (no approximate multipliers)
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 1: Training with standard Conv2D")
print("=" * 80)

model_std = build_model_for_training(config, input_shape, num_classes, learning_rate=0.001)

print(f"\nStandard model layers: {len(model_std.layers)}")
for i, layer in enumerate(model_std.layers[:10]):  # Show first 10 layers
    print(f"  {i}: {layer.__class__.__name__}")

history = model_std.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=128,
    verbose=1
)

std_accuracy = evaluate_model(model_std, x_val, y_val)
print(f"\n✓ Standard model validation accuracy: {std_accuracy:.4f}")

# Save weights
trained_weights = model_std.get_weights()
print(f"✓ Saved {len(trained_weights)} weight arrays")

# ==============================================================================
# STAGE 2: Evaluate with approximate multipliers
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 2: Evaluating with approximate multipliers")
print("=" * 80)

model_approx = build_model_for_evaluation(config, input_shape, num_classes, learning_rate=0.001, weights=trained_weights)

print(f"\nApproximate model layers: {len(model_approx.layers)}")
for i, layer in enumerate(model_approx.layers[:10]):  # Show first 10 layers
    print(f"  {i}: {layer.__class__.__name__}")

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
    print("\n✓ SUCCESS: Reference architecture pattern works!")
elif drop_pct < 50:
    print("\n⚠ PARTIAL: Some degradation but not total collapse")
else:
    print("\n✗ FAILURE: Severe degradation")

# ==============================================================================
# Test multiple multipliers
# ==============================================================================
print("\n" + "=" * 80)
print("Testing multiple multipliers...")
print("=" * 80)

multiplier_files = sorted(glob.glob('./multipliers/*.bin'))[:10]  # Test first 10
results = []

for mul_file in multiplier_files:
    mul_name = mul_file.split('/')[-1]

    # Update config with this multiplier
    test_config = config.copy()
    test_config['conv1_multiplier'] = mul_file
    test_config['conv2_multiplier'] = mul_file
    test_config['conv3_multiplier'] = mul_file

    # Build and evaluate
    model_test = build_model_for_evaluation(test_config, input_shape, num_classes, weights=trained_weights)
    test_acc = evaluate_model(model_test, x_val, y_val)
    test_drop = ((std_accuracy - test_acc) / std_accuracy) * 100

    results.append({'name': mul_name, 'accuracy': test_acc, 'drop_pct': test_drop})
    print(f"  {mul_name:20s}  Acc: {test_acc:.4f}  Drop: {test_drop:6.2f}%")

    del model_test
    keras.backend.clear_session()

print("\n" + "=" * 80)
print("Multiplier differentiation test:")
drops = [r['drop_pct'] for r in results]
if max(drops) - min(drops) > 5:
    print("✓ Multipliers show DIFFERENT performance (good!)")
else:
    print("✗ Multipliers show UNIFORM performance (bad)")
