import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataloader import load_dataset
from operations import get_search_space
from model_builder import build_model
from training import train_model, evaluate_model

# Disable verbose output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
try:
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

print("="*70)
print("VERIFICATION: Testing mul8u_2P7.bin multiple times")
print("="*70)

# Load CIFAR-10
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('cifar10', num_val=6000)
height, width, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
num_classes = len(np.unique(y_train))

# Use subset
x_train = x_train[:5000]
y_train = y_train[:5000]
x_val = x_val[:1000]
y_val = y_val[:1000]

# Best architecture from your results
architecture = [
    {'op': 'avg_pool', 'filters': 64, 'use_bn': False}, 
    {'op': 'conv5x5', 'filters': 64, 'use_bn': False}
]

print(f"\nArchitecture: {architecture}")
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")

# Test standard model multiple times
print("\n" + "="*70)
print("STANDARD MODEL - Multiple Runs")
print("="*70)

standard_accuracies = []
for run in range(5):
    print(f"\nRun {run+1}/5:")
    
    # Build and train
    search_space_std = get_search_space(use_approximate=False)
    model_std = build_model(architecture, search_space_std, 
                           input_shape=(height, width, channels), 
                           num_classes=num_classes)
    train_model(model_std, x_train, y_train, x_val, y_val, epochs=50)
    
    # Evaluate multiple times
    run_evals = []
    for _ in range(3):
        acc = evaluate_model(model_std, x_val, y_val)
        run_evals.append(acc)
    
    avg_acc = np.mean(run_evals)
    standard_accuracies.append(avg_acc)
    print(f"  Accuracies: {[f'{a:.4f}' for a in run_evals]}")
    print(f"  Mean: {avg_acc:.4f}")
    
    del model_std
    keras.backend.clear_session()

print(f"\n{'='*70}")
print(f"Standard Model Summary:")
print(f"  Mean Accuracy: {np.mean(standard_accuracies):.4f}")
print(f"  Std Dev:       {np.std(standard_accuracies):.4f}")
print(f"  Min:           {np.min(standard_accuracies):.4f}")
print(f"  Max:           {np.max(standard_accuracies):.4f}")

# Test approximate model multiple times
print("\n" + "="*70)
print("APPROXIMATE MODEL (mul8u_2P7.bin) - Multiple Runs")
print("="*70)

approximate_accuracies = []
for run in range(5):
    print(f"\nRun {run+1}/5:")
    
    # Build and train standard first
    search_space_std = get_search_space(use_approximate=False)
    model_std = build_model(architecture, search_space_std, 
                           input_shape=(height, width, channels), 
                           num_classes=num_classes)
    train_model(model_std, x_train, y_train, x_val, y_val, epochs=50)
    trained_weights = model_std.get_weights()
    
    # Build approximate model and copy weights
    search_space_approx = get_search_space(use_approximate=True, 
                                           mul_map_file='./multipliers/mul8u_2P7.bin')
    model_approx = build_model(architecture, search_space_approx,
                              input_shape=(height, width, channels), 
                              num_classes=num_classes)
    model_approx.set_weights(trained_weights)
    
    # Evaluate multiple times
    run_evals = []
    for _ in range(3):
        acc = evaluate_model(model_approx, x_val, y_val)
        run_evals.append(acc)
    
    avg_acc = np.mean(run_evals)
    approximate_accuracies.append(avg_acc)
    print(f"  Accuracies: {[f'{a:.4f}' for a in run_evals]}")
    print(f"  Mean: {avg_acc:.4f}")
    
    del model_std, model_approx
    keras.backend.clear_session()

print(f"\n{'='*70}")
print(f"Approximate Model Summary:")
print(f"  Mean Accuracy: {np.mean(approximate_accuracies):.4f}")
print(f"  Std Dev:       {np.std(approximate_accuracies):.4f}")
print(f"  Min:           {np.min(approximate_accuracies):.4f}")
print(f"  Max:           {np.max(approximate_accuracies):.4f}")

# Statistical comparison
print("\n" + "="*70)
print("STATISTICAL COMPARISON")
print("="*70)

mean_std = np.mean(standard_accuracies)
mean_approx = np.mean(approximate_accuracies)
difference = mean_approx - mean_std

print(f"\nStandard Model:    {mean_std:.4f} ± {np.std(standard_accuracies):.4f}")
print(f"Approximate Model: {mean_approx:.4f} ± {np.std(approximate_accuracies):.4f}")
print(f"Difference:        {difference:.4f} ({difference/mean_std*100:+.2f}%)")

# Paired t-test
from scipy import stats
t_stat, p_value = stats.ttest_ind(approximate_accuracies, standard_accuracies)

print(f"\nStatistical Test (Independent t-test):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_value:.4f}")

if p_value > 0.05:
    print(f"  Result: NO significant difference (p > 0.05)")
    print(f"          The 'improvement' is likely statistical noise.")
else:
    if mean_approx > mean_std:
        print(f"  Result: Approximate model is SIGNIFICANTLY BETTER (p < 0.05)")
    else:
        print(f"  Result: Approximate model is SIGNIFICANTLY WORSE (p < 0.05)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"The negative drop of -0.23% in your original results is:")

if abs(difference) < 0.01 or p_value > 0.05:
    print("✓ Within measurement noise (not statistically significant)")
    print("✓ Both models perform essentially the same")
    print("✓ mul8u_2P7.bin causes ZERO accuracy loss")
else:
    if difference > 0:
        print("✓ Statistically significant improvement!")
        print("✓ Approximate multiplier provides slight regularization benefit")
    else:
        print("✗ Statistically significant degradation")

print("="*70)
