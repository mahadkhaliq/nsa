"""Quick test of hardware-aware NAS on CIFAR-10"""
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

# Load CIFAR-10
print("Loading CIFAR-10 dataset...")
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('cifar10')
x_train, y_train = x_train[:5000], y_train[:5000]  # Reduced for quick test
x_val, y_val = x_val[:1000], y_val[:1000]

print(f"Input shape: {x_train.shape[1:]}")
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Number of classes: {len(np.unique(y_train))}")

input_shape = x_train.shape[1:]
num_classes = 10

# Load multipliers for testing
all_multipliers = sorted(glob.glob('./multipliers/*.bin'))
test_multipliers = all_multipliers[:5]  # Use first 5 for quick test
print(f"\nFound {len(all_multipliers)} multipliers, using {len(test_multipliers)} for test")

print("\n" + "="*80)
print("CIFAR-10 Hardware-Aware NAS Quick Test")
print("="*80)

logger = NASLogger(log_dir='logs')

# Run hardware-aware NAS
results = run_nas_reference(
    x_train, y_train, x_val, y_val,
    input_shape, num_classes,
    num_trials=3,  # Quick test with 3 trials
    epochs_per_trial=15,  # CIFAR-10 needs more epochs
    batch_size=128,  # Larger batch for CIFAR-10
    learning_rate=0.001,
    method='random',
    test_multipliers=test_multipliers,
    use_approximate_in_search=True,
    logger=logger
)

print("\n" + "="*80)
print("Results:")
print("="*80)
print(f"Best standard accuracy: {results['best_result']['std_accuracy']:.4f}")
print(f"Best mean approx accuracy: {results['best_result']['mean_approx_accuracy']:.4f}")
print(f"Best mean accuracy drop: {results['best_result']['mean_accuracy_drop']:.4f}")
print(f"Best fitness score: {results['best_result']['fitness']:.4f}")

if results['best_result']['multiplier_accuracies']:
    mul_accs = results['best_result']['multiplier_accuracies']
    print(f"Best/Worst multiplier: {max(mul_accs):.4f} / {min(mul_accs):.4f}")

print(f"\nAll fitness scores: {[f'{f:.4f}' for f in results['all_fitness_scores']]}")

print(f"\n{'='*80}")
print("CIFAR-10 is more challenging than MNIST:")
print("  - Expect lower accuracies (~60-75% for quick tests)")
print("  - Needs more training epochs for convergence")
print("  - Benefits from larger batch sizes (128+)")
print("  - Color images (3 channels) vs grayscale (1 channel)")
print(f"{'='*80}")
