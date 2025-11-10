"""Test hardware-aware NAS with ALL approximate multipliers"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
from dataloader import load_dataset
from nas_reference import run_nas_reference
from logger import NASLogger

# GPU memory management
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✓ GPU memory growth enabled")
    except:
        pass

print(f"TensorFlow version: {tf.__version__}")

# Load MNIST
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('mnist')
x_train, y_train = x_train[:2000], y_train[:2000]
x_val, y_val = x_val[:500], y_val[:500]

if len(x_train.shape) == 3:
    x_train = x_train[..., None]
    x_val = x_val[..., None]

input_shape = x_train.shape[1:]
num_classes = 10

print(f"Input shape: {input_shape}")
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")

# Load ALL multipliers
all_multipliers = sorted(glob.glob('./multipliers/*.bin'))
test_multipliers = all_multipliers[:5]  # Use first 5 for quick test
print(f"Found {len(all_multipliers)} multipliers, using {len(test_multipliers)} for test")

print("\n" + "="*80)
print("COMPARISON: Standard NAS vs Hardware-Aware NAS")
print("="*80)

logger = NASLogger(log_dir='logs')

# ============================================================================
# Test 1: Standard NAS (no approximate multiplier)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Standard NAS (no approximate multiplier)")
print("="*80)

results_std = run_nas_reference(
    x_train, y_train, x_val, y_val,
    input_shape, num_classes,
    num_trials=3,
    epochs_per_trial=10,
    batch_size=64,
    learning_rate=0.001,
    method='random',
    test_multiplier=None,
    use_approximate_in_search=False,
    logger=None  # Disable logging for cleaner output
)

print(f"\nStandard NAS Results:")
print(f"  Best standard accuracy: {results_std['best_result']['std_accuracy']:.4f}")
print(f"  All fitness scores: {[f'{f:.4f}' for f in results_std['all_fitness_scores']]}")

# ============================================================================
# Test 2: Hardware-Aware NAS (with ALL multipliers)
# ============================================================================
print("\n" + "="*80)
print(f"TEST 2: Hardware-Aware NAS (with {len(test_multipliers)} multipliers)")
for i, mul in enumerate(test_multipliers):
    print(f"  {i+1}. {mul.split('/')[-1]}")
print("="*80)

results_hw = run_nas_reference(
    x_train, y_train, x_val, y_val,
    input_shape, num_classes,
    num_trials=3,
    epochs_per_trial=10,
    batch_size=64,
    learning_rate=0.001,
    method='random',
    test_multipliers=test_multipliers,
    use_approximate_in_search=True,
    logger=None  # Disable logging for cleaner output
)

print(f"\nHardware-Aware NAS Results:")
print(f"  Best standard accuracy: {results_hw['best_result']['std_accuracy']:.4f}")
print(f"  Best mean approx accuracy: {results_hw['best_result']['mean_approx_accuracy']:.4f}")
print(f"  Best mean accuracy drop: {results_hw['best_result']['mean_accuracy_drop']:.4f}")
print(f"  Best fitness score: {results_hw['best_result']['fitness']:.4f}")
print(f"  All fitness scores: {[f'{f:.4f}' for f in results_hw['all_fitness_scores']]}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*80)
print("COMPARISON:")
print("="*80)

print(f"\nStandard NAS best config:")
print(f"  Standard accuracy: {results_std['best_result']['std_accuracy']:.4f}")

print(f"\nHardware-Aware NAS best config:")
print(f"  Standard accuracy: {results_hw['best_result']['std_accuracy']:.4f}")
print(f"  Mean approx accuracy: {results_hw['best_result']['mean_approx_accuracy']:.4f}")
drop_pct = (results_hw['best_result']['mean_accuracy_drop'] / results_hw['best_result']['std_accuracy']) * 100
print(f"  Mean accuracy drop: {drop_pct:.2f}%")
print(f"  Fitness (0.7*std + 0.3*robust): {results_hw['best_result']['fitness']:.4f}")
if results_hw['best_result']['multiplier_accuracies']:
    print(f"  Best/Worst multiplier: {max(results_hw['best_result']['multiplier_accuracies']):.4f} / {min(results_hw['best_result']['multiplier_accuracies']):.4f}")

print(f"\n{'='*80}")
print("Hardware-aware NAS finds architectures that are:")
print("  1. Still accurate with standard multipliers")
print(f"  2. ROBUST to approximate multiplier errors across {len(test_multipliers)} different multipliers")
print("  3. Optimized for AVERAGE performance across all multipliers")
print(f"{'='*80}")
