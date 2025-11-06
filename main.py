from tensorflow import keras
from nas import run_nas
from model_builder import build_model
from operations import get_search_space
from training import evaluate_model, train_model
from logger import NASLogger
import numpy as np
from dataloader import load_dataset
import os
import glob

# Initialize logger
logger = NASLogger(log_dir='logs')

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('cifar10', num_val=6000)

# ✅ FIX: Properly handle image dimensions
if x_train.ndim == 3:  # Grayscale images (H, W)
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)

# ✅ FIX: Extract shape correctly
height, width, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
num_classes = len(np.unique(y_train))

logger.log(f"Dataset: CIFAR-10")
logger.log(f"Input shape: ({height}, {width}, {channels})")
logger.log(f"Number of classes: {num_classes}")
logger.log(f"Training samples: {len(x_train)}")

# Use subset for faster experimentation
x_train = x_train[:5000]
y_train = y_train[:5000]
x_val = x_val[:1000]
y_val = y_val[:1000]

logger.log_section("STEP 1: NAS with STANDARD multipliers")
results_std, best_std = run_nas(
    x_train, y_train, x_val, y_val,
    num_trials=10,
    num_blocks=2,
    use_approximate=False,
    logger=logger,
    input_shape=(height, width, channels),  # ✅ Pass shape
    num_classes=num_classes
)

logger.log_section("STEP 2: Train BEST architecture")
logger.log(f"Best Architecture: {best_std['architecture']}")

search_space_std = get_search_space(use_approximate=False)
model_std = build_model(
    best_std['architecture'], 
    search_space_std, 
    input_shape=(height, width, channels),  # ✅ Correct shape
    num_classes=num_classes
)

logger.log("Training standard model...")
train_model(model_std, x_train, y_train, x_val, y_val, epochs=50)

std_accuracy = evaluate_model(model_std, x_val, y_val)
logger.log_training(epochs=50, std_accuracy=std_accuracy)
logger.log_best_architecture(best_std['architecture'], std_accuracy)

trained_weights = model_std.get_weights()

logger.log_section("STEP 3: Test with DIFFERENT APPROXIMATE MULTIPLIERS")

multiplier_dir = './multipliers'
multiplier_files = sorted(glob.glob(os.path.join(multiplier_dir, '*.bin')))

if not multiplier_files:
    logger.log(f"⚠ No .bin files found in {multiplier_dir}")
    logger.log("Testing with empty mul_map_file (baseline approximation)...")
    multiplier_files = ['']

logger.log(f"Found {len(multiplier_files)} multipliers to test\n")

approx_results = []

for mul_file in multiplier_files:
    mul_name = os.path.basename(mul_file) if mul_file else 'BASELINE'
    
    search_space_approx = get_search_space(use_approximate=True, mul_map_file=mul_file)
    model_approx = build_model(
        best_std['architecture'], 
        search_space_approx,
        input_shape=(height, width, channels),  # ✅ Correct shape
        num_classes=num_classes
    )
    
    try:
        model_approx.set_weights(trained_weights)
    except Exception as e:
        logger.log(f"✗ Failed to copy weights for {mul_name}: {e}")
        continue
    
    approx_accuracy = evaluate_model(model_approx, x_val, y_val)
    accuracy_drop = std_accuracy - approx_accuracy
    drop_percent = (accuracy_drop / std_accuracy) * 100 if std_accuracy > 0 else 0
    
    logger.log_multiplier_test(mul_name, approx_accuracy, accuracy_drop, drop_percent)
    
    approx_results.append({
        'multiplier': mul_name,
        'file': mul_file,
        'accuracy': approx_accuracy,
        'drop': accuracy_drop,
        'drop_percent': drop_percent
    })
    
    del model_approx
    keras.backend.clear_session()

# Log summary
logger.log_summary(std_accuracy, approx_results)

# Print table
logger.log(f"\n{'Multiplier':<30} {'Accuracy':<12} {'Drop':<12} {'Drop %':<10}")
logger.log("-" * 70)

approx_results.sort(key=lambda x: x['accuracy'], reverse=True)

for result in approx_results:
    logger.log(f"{result['multiplier']:<30} {result['accuracy']:<12.4f} {result['drop']:<12.4f} {result['drop_percent']:<10.2f}%")

logger.log_section("BEST APPROXIMATE MULTIPLIER")
if approx_results:
    best_approx = approx_results[0]
    logger.log(f"Multiplier:  {best_approx['multiplier']}")
    logger.log(f"Accuracy:    {best_approx['accuracy']:.4f}")
    logger.log(f"Drop:        {best_approx['drop']:.4f} ({best_approx['drop_percent']:.2f}%)")

logger.close()
