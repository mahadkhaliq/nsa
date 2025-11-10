"""Quick test of NAS with reference architecture (3 trials only)"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
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

# Quick NAS test
print("\n" + "="*80)
print("Quick NAS Test (3 trials, 10 epochs each)")
print("="*80)

logger = NASLogger(log_dir='logs')

results = run_nas_reference(
    x_train, y_train, x_val, y_val,
    input_shape, num_classes,
    num_trials=3,
    epochs_per_trial=10,
    batch_size=64,
    learning_rate=0.001,
    method='random',
    logger=logger
)

print("\n" + "="*80)
print("NAS Results:")
print("="*80)
print(f"Best accuracy: {results['best_accuracy']:.4f}")
print(f"Mean accuracy: {np.mean(results['all_accuracies']):.4f}")
print(f"Std accuracy: {np.std(results['all_accuracies']):.4f}")
print(f"\nBest config: {results['best_config']}")
print("\nAll accuracies:", [f"{a:.4f}" for a in results['all_accuracies']])
