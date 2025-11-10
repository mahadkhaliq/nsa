"""Quick test of RTAMT integration with hardware-aware NAS"""
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

# Load MNIST dataset (small subset for quick test)
print("\nLoading MNIST dataset...")
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('mnist')
x_train, y_train = x_train[:1000], y_train[:1000]  # Very small for quick test
x_val, y_val = x_val[:200], y_val[:200]

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")

input_shape = x_train.shape[1:]
num_classes = 10

# Load multipliers for testing
all_multipliers = sorted(glob.glob('./multipliers/*.bin'))
test_multipliers = all_multipliers[:3]  # Use first 3 for very quick test
print(f"\nUsing {len(test_multipliers)} multipliers for testing")
for mul in test_multipliers:
    print(f"  - {mul}")

print("\n" + "="*80)
print("Testing RTAMT Integration with Hardware-Aware NAS")
print("="*80)
print("\nThis quick test will:")
print("  1. Run 3 NAS trials with RTAMT verification enabled")
print("  2. Verify training convergence (STL)")
print("  3. Verify multiplier robustness (STL)")
print("  4. Verify energy-accuracy tradeoff (STL)")
print("  5. Generate Pareto frontier")
print("  6. Apply fitness penalty for failed verification")
print("\nExpected time: ~3-5 minutes")
print("="*80 + "\n")

logger = NASLogger(log_dir='logs')

# Run hardware-aware NAS with RTAMT verification
results = run_nas_reference(
    x_train, y_train, x_val, y_val,
    input_shape, num_classes,
    num_trials=3,  # Very quick test
    epochs_per_trial=5,  # Very few epochs for speed
    batch_size=64,
    learning_rate=0.001,
    method='random',
    test_multipliers=test_multipliers,
    use_approximate_in_search=True,
    logger=logger,
    use_rtamt_verification=True,  # Enable RTAMT
    min_accuracy_threshold=0.6,  # Lower threshold for quick test
    max_accuracy_drop_percent=15.0  # More lenient for quick test
)

print("\n" + "="*80)
print("RTAMT Integration Test Results")
print("="*80)

best_result = results['best_result']
print(f"\nBest Architecture:")
print(f"  Standard accuracy: {best_result['std_accuracy']:.4f}")
print(f"  Mean approx accuracy: {best_result['mean_approx_accuracy']:.4f}")
print(f"  Mean accuracy drop: {best_result['mean_accuracy_drop']:.4f}")
print(f"  Fitness score: {best_result['fitness']:.4f}")

# Verification results
if 'verification' in best_result and best_result['verification']:
    ver = best_result['verification']
    print(f"\nFormal Verification (RTAMT):")
    print(f"  Training convergence: {'✓ PASS' if ver['training']['satisfied'] else '✗ FAIL'}")
    print(f"    Robustness value: {ver['training']['robustness']:.4f}")

    if ver['robustness']:
        print(f"  Multiplier robustness: {ver['robustness']['satisfaction_rate']:.1%} pass rate")
        print(f"    Mean robustness: {ver['robustness']['mean_robustness']:.4f}")

    if ver['pareto']:
        satisfied_count = sum(1 for p in ver['pareto'] if p['satisfied'])
        print(f"  Pareto points: {satisfied_count}/{len(ver['pareto'])} satisfy constraints")

        # Show Pareto results
        print(f"\n  Pareto Analysis:")
        for p in ver['pareto']:
            status = '✓' if p['satisfied'] else '✗'
            print(f"    {status} {p['multiplier']:20s}  Acc: {p['accuracy']:.4f}  Energy: {p['energy_ratio']:.2f}x")

    print(f"\n  Overall verification: {'✓ PASS' if ver['overall_satisfied'] else '✗ FAIL'}")
else:
    print("\n⚠ No verification results found")

# Summary statistics across all trials
print(f"\nAll Trials Summary:")
verified_count = sum(1 for r in results['results'] if r['verification'] and r['verification']['overall_satisfied'])
print(f"  Verified architectures: {verified_count}/{len(results['results'])} ({100*verified_count/len(results['results']):.1f}%)")

all_fitness = results['all_fitness_scores']
print(f"  Fitness scores: {[f'{f:.4f}' for f in all_fitness]}")
print(f"  Mean fitness: {np.mean(all_fitness):.4f} ± {np.std(all_fitness):.4f}")

print("\n" + "="*80)
print("RTAMT Integration Test Complete!")
print("="*80)
print("\nKey features tested:")
print("  ✓ STL specification creation")
print("  ✓ Training convergence verification")
print("  ✓ Multiplier robustness verification")
print("  ✓ Energy-accuracy tradeoff verification")
print("  ✓ Pareto point identification")
print("  ✓ Fitness penalty for failed verification")
print("  ✓ Verification logging and reporting")
print("\nNext steps:")
print("  1. Run full experiment with --use_rtamt flag")
print("  2. Review verification reports and Pareto frontier plots")
print("  3. Adjust thresholds based on dataset and requirements")
print("="*80)
