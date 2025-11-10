import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Try to import FakeApproxConv2D (only needed for evaluation)
try:
    from keras.layers.fake_approx_convolutional import FakeApproxConv2D
    APPROX_AVAILABLE = True
except ImportError:
    print("Warning: FakeApproxConv2D not available. Will use standard Conv2D only.")
    APPROX_AVAILABLE = False

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

##============================================================================##
## GPU Setup
##============================================================================##
physical_devices = tf.config.list_physical_devices('GPU')
try:
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

##============================================================================##
## Data Loading - CIFAR-10
##============================================================================##
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Create validation split
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")
print(f"Input shape: {x_train.shape[1:]}")


bin_path = './multipliers'
entries = os.listdir(bin_path)
bin_files = [os.path.join(bin_path,entry) for entry in entries if os.path.isfile(os.path.join(bin_path,entry))]
print(bin_path)
print(entries)
print(bin_files)


##============================================================================##
## Data Augmentation (Compatible with older TensorFlow)
##============================================================================##
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# No augmentation for validation/test
val_datagen = ImageDataGenerator()

##============================================================================##
## Search Space Definition - Adjusted for CIFAR
##============================================================================##
SEARCH_SPACE = {
    'conv1_filters': [16, 24, 32, 48, 64],
    'conv1_kernel': [3, 5],
    'conv1_multiplier': bin_files,
    'conv2_filters': [32, 48, 64, 96, 128],
    'conv2_kernel': [3, 5],
    'conv2_multiplier': bin_files,
    'conv3_filters': [64, 96, 128, 192, 256],
    'conv3_kernel': [3, 5],
    'conv3_multiplier': bin_files,
    'dense1_size': [128, 256, 512],
    'dropout_rate': [0.3, 0.4, 0.5],
}

APPROX_NAMES = {None: 'Accurate'}
for path in bin_files:
    APPROX_NAMES[path] = os.path.basename(path)
##============================================================================##
## Model Builders
##============================================================================##
def build_model_for_training(config, use_augmentation=True):
    """Build model with STANDARD Conv2D for training"""
    layers = []
    
    # Conv Block 1
    layers.extend([
        tf.keras.layers.Conv2D(
            filters=config['conv1_filters'],
            kernel_size=(config['conv1_kernel'], config['conv1_kernel']),
            padding='same',
            activation='relu',
            input_shape=(32, 32, 3)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=config['conv1_filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])
    
    # Conv Block 2
    layers.extend([
        tf.keras.layers.Conv2D(
            filters=config['conv2_filters'],
            kernel_size=(config['conv2_kernel'], config['conv2_kernel']),
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=config['conv2_filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])
    
    # Conv Block 3
    layers.extend([
        tf.keras.layers.Conv2D(
            filters=config['conv3_filters'],
            kernel_size=(config['conv3_kernel'], config['conv3_kernel']),
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])
    
    # Dense layers with dropout
    layers.extend([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(config['dense1_size'], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(config['dropout_rate']),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_model_for_evaluation(config, weights=None):
    """Build model with FakeApproxConv2D for evaluation"""
    if not APPROX_AVAILABLE:
        return build_model_for_training(config, use_augmentation=False)
    
    layers = []
    
    # Conv Block 1 - with approximate multiplier
    if config['conv1_multiplier'] is None:
        layers.append(tf.keras.layers.Conv2D(
            filters=config['conv1_filters'],
            kernel_size=(config['conv1_kernel'], config['conv1_kernel']),
            padding='same',
            activation='relu',
            input_shape=(32, 32, 3)
        ))
    else:
        layers.append(FakeApproxConv2D(
            filters=config['conv1_filters'],
            kernel_size=(config['conv1_kernel'], config['conv1_kernel']),
            padding='same',
            activation='relu',
            mul_map_file=config['conv1_multiplier'],
            input_shape=(32, 32, 3)
        ))
    
    layers.extend([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=config['conv1_filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])
    
    # Conv Block 2 - with approximate multiplier
    if config['conv2_multiplier'] is None:
        layers.append(tf.keras.layers.Conv2D(
            filters=config['conv2_filters'],
            kernel_size=(config['conv2_kernel'], config['conv2_kernel']),
            padding='same',
            activation='relu'
        ))
    else:
        layers.append(FakeApproxConv2D(
            filters=config['conv2_filters'],
            kernel_size=(config['conv2_kernel'], config['conv2_kernel']),
            padding='same',
            activation='relu',
            mul_map_file=config['conv2_multiplier']
        ))
    
    layers.extend([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=config['conv2_filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])
    
    # Conv Block 3 - with approximate multiplier
    if config['conv3_multiplier'] is None:
        layers.append(tf.keras.layers.Conv2D(
            filters=config['conv3_filters'],
            kernel_size=(config['conv3_kernel'], config['conv3_kernel']),
            padding='same',
            activation='relu'
        ))
    else:
        layers.append(FakeApproxConv2D(
            filters=config['conv3_filters'],
            kernel_size=(config['conv3_kernel'], config['conv3_kernel']),
            padding='same',
            activation='relu',
            mul_map_file=config['conv3_multiplier']
        ))
    
    layers.extend([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])
    
    # Dense layers
    layers.extend([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(config['dense1_size'], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(config['dropout_rate']),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model = tf.keras.Sequential(layers)
    
    # Transfer weights if provided
    if weights is not None:
        model.set_weights(weights)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def config_to_string(config):
    """Convert config to readable string"""
    c1_mult = APPROX_NAMES[config['conv1_multiplier']]
    c2_mult = APPROX_NAMES[config['conv2_multiplier']]
    c3_mult = APPROX_NAMES[config['conv3_multiplier']]
    return (f"C1:{config['conv1_filters']}f-{config['conv1_kernel']}k-{c1_mult} | "
            f"C2:{config['conv2_filters']}f-{config['conv2_kernel']}k-{c2_mult} | "
            f"C3:{config['conv3_filters']}f-{config['conv3_kernel']}k-{c3_mult} | "
            f"D:{config['dense1_size']}-drop{config['dropout_rate']}")

def estimate_energy(config):
    """Estimate relative energy consumption"""
    energy = 0.0
    
    # Conv1 energy
    conv1_ops = config['conv1_filters'] * (config['conv1_kernel'] ** 2) * 2  # Two conv layers
    if config['conv1_multiplier'] is None:
        energy += conv1_ops * 1.0
    elif 'mul8u_1JFF' in config['conv1_multiplier']:
        energy += conv1_ops * 0.7
    else:
        energy += conv1_ops * 0.5
    
    # Conv2 energy
    conv2_ops = config['conv2_filters'] * (config['conv2_kernel'] ** 2) * 2
    if config['conv2_multiplier'] is None:
        energy += conv2_ops * 1.0
    elif 'mul8u_1JFF' in config['conv2_multiplier']:
        energy += conv2_ops * 0.7
    else:
        energy += conv2_ops * 0.5
    
    # Conv3 energy
    conv3_ops = config['conv3_filters'] * (config['conv3_kernel'] ** 2)
    if config['conv3_multiplier'] is None:
        energy += conv3_ops * 1.0
    elif 'mul8u_1JFF' in config['conv3_multiplier']:
        energy += conv3_ops * 0.7
    else:
        energy += conv3_ops * 0.5
    
    # Dense layer
    energy += config['dense1_size'] * 0.1
    
    return energy

##============================================================================##
## Pareto Frontier Analysis
##============================================================================##
def compute_pareto_frontier(results):
    """Compute Pareto frontier for accuracy vs energy"""
    points = [(r['accuracy'], r['energy'], i) for i, r in enumerate(results)]
    points.sort(key=lambda x: (-x[0], x[1]))
    
    pareto_indices = []
    min_energy = float('inf')
    
    for acc, energy, idx in points:
        if energy < min_energy:
            pareto_indices.append(idx)
            min_energy = energy
    
    return sorted(pareto_indices)

##============================================================================##
## Plotting Functions
##============================================================================##
def plot_nas_results(results, output_dir, method):
    """Generate plots for NAS results"""
    os.makedirs(output_dir, exist_ok=True)
    
    pareto_indices = compute_pareto_frontier(results)
    
    # Plot 1: Accuracy vs Trial
    plt.figure()
    accuracies = [r['accuracy'] for r in results]
    trials = [r['trial'] for r in results]
    plt.plot(trials, accuracies, 'b-o', label='Test Accuracy')
    if method == 'two_stage':
        accuracies_no_approx = [r['accuracy_without_approx'] for r in results]
        plt.plot(trials, accuracies_no_approx, 'r--o', label='Accuracy (No Approx)')
    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs Trial (CIFAR-10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_trial.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Energy vs Accuracy with Pareto Frontier
    plt.figure()
    energies = [r['energy'] for r in results]
    plt.scatter(energies, accuracies, c='blue', alpha=0.5, s=50, label='All Configurations')
    
    pareto_energies = [results[i]['energy'] for i in pareto_indices]
    pareto_accuracies = [results[i]['accuracy'] for i in pareto_indices]
    plt.scatter(pareto_energies, pareto_accuracies, c='red', s=150, marker='s', 
                label='Pareto Frontier', edgecolors='black', linewidths=2)
    plt.plot(pareto_energies, pareto_accuracies, 'r--', alpha=0.5, linewidth=2)
    
    best_idx = np.argmax(accuracies)
    plt.scatter(energies[best_idx], accuracies[best_idx], c='green', s=200, 
                marker='*', label='Best Accuracy', edgecolors='black', linewidths=2)
    
    plt.xlabel('Estimated Energy')
    plt.ylabel('Test Accuracy')
    plt.title('Energy vs Accuracy Trade-off (CIFAR-10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'energy_vs_accuracy_pareto.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Training history of best model
    best_result = results[best_idx]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs = range(1, len(best_result['history']['accuracy']) + 1)
    plt.plot(epochs, best_result['history']['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, best_result['history']['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Best Model: Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, best_result['history']['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, best_result['history']['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Best Model: Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_best_config_summary(results, output_dir):
    """Save detailed summary"""
    best_idx = np.argmax([r['accuracy'] for r in results])
    best_result = results[best_idx]
    pareto_indices = compute_pareto_frontier(results)
    
    best_summary = {
        'Best Configuration': config_to_string(best_result['config']),
        'Test Accuracy': float(best_result['accuracy']),
        'Test Loss': float(best_result['loss']),
        'Estimated Energy': float(best_result['energy']),
        'Configuration Details': {
            'Conv1 Filters': int(best_result['config']['conv1_filters']),
            'Conv1 Kernel Size': int(best_result['config']['conv1_kernel']),
            'Conv1 Multiplier': APPROX_NAMES[best_result['config']['conv1_multiplier']],
            'Conv2 Filters': int(best_result['config']['conv2_filters']),
            'Conv2 Kernel Size': int(best_result['config']['conv2_kernel']),
            'Conv2 Multiplier': APPROX_NAMES[best_result['config']['conv2_multiplier']],
            'Conv3 Filters': int(best_result['config']['conv3_filters']),
            'Conv3 Kernel Size': int(best_result['config']['conv3_kernel']),
            'Conv3 Multiplier': APPROX_NAMES[best_result['config']['conv3_multiplier']],
            'Dense1 Size': int(best_result['config']['dense1_size']),
            'Dropout Rate': float(best_result['config']['dropout_rate'])
        }
    }
    
    if 'accuracy_without_approx' in best_result:
        best_summary['Accuracy (No Approx)'] = float(best_result['accuracy_without_approx'])
        best_summary['Accuracy Degradation'] = float(best_result['accuracy_degradation'])
    
    pareto_summary = []
    for idx in pareto_indices:
        r = results[idx]
        pareto_config = {
            'Configuration': config_to_string(r['config']),
            'Test Accuracy': float(r['accuracy']),
            'Estimated Energy': float(r['energy']),
            'Trial': int(r['trial'])
        }
        if 'accuracy_without_approx' in r:
            pareto_config['Accuracy (No Approx)'] = float(r['accuracy_without_approx'])
            pareto_config['Accuracy Degradation'] = float(r['accuracy_degradation'])
        pareto_summary.append(pareto_config)
    
    summary = {
        'Dataset': 'CIFAR-10',
        'Best Configuration': best_summary,
        'Pareto Frontier': pareto_summary
    }
    
    with open(os.path.join(output_dir, 'best_config_and_pareto_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

##============================================================================##
## Two-Stage NAS
##============================================================================##
def bayesian_nas_fixed(num_trials=20, epochs_per_trial=10, output_dir='nas_results_cifar'):
    """Two-stage NAS: Train with standard Conv2D, evaluate with approximate"""
    print("="*80)
    print("BAYESIAN OPTIMIZATION NAS - CIFAR-10 (TWO-STAGE)")
    print("="*80)
    
    X_tried = []
    y_tried = []
    results = []
    
    for trial in range(num_trials):
        print(f"\n{'='*80}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*80}")
        
        # Configuration selection
        if trial < 5:
            config = {
                'conv1_filters': np.random.choice(SEARCH_SPACE['conv1_filters']),
                'conv1_kernel': np.random.choice(SEARCH_SPACE['conv1_kernel']),
                'conv1_multiplier': np.random.choice(SEARCH_SPACE['conv1_multiplier']),
                'conv2_filters': np.random.choice(SEARCH_SPACE['conv2_filters']),
                'conv2_kernel': np.random.choice(SEARCH_SPACE['conv2_kernel']),
                'conv2_multiplier': np.random.choice(SEARCH_SPACE['conv2_multiplier']),
                'conv3_filters': np.random.choice(SEARCH_SPACE['conv3_filters']),
                'conv3_kernel': np.random.choice(SEARCH_SPACE['conv3_kernel']),
                'conv3_multiplier': np.random.choice(SEARCH_SPACE['conv3_multiplier']),
                'dense1_size': np.random.choice(SEARCH_SPACE['dense1_size']),
                'dropout_rate': np.random.choice(SEARCH_SPACE['dropout_rate']),
            }
        else:
            best_idx = np.argmax(y_tried)
            best_config = X_tried[best_idx]
            config = best_config.copy()
            if np.random.random() < 0.7:
                mutation_key = np.random.choice(list(SEARCH_SPACE.keys()))
                config[mutation_key] = np.random.choice(SEARCH_SPACE[mutation_key])
        
        print(f"Config: {config_to_string(config)}")
        
        # Stage 1: Train with standard Conv2D
        print("→ Stage 1: Training with standard Conv2D...")
        training_model = build_model_for_training(config)
        
        # Create data generators
        train_generator = train_datagen.flow(x_train, y_train, batch_size=128)
        
        history = training_model.fit(
            train_generator,
            steps_per_epoch=len(x_train) // 128,
            validation_data=(x_val, y_val),
            epochs=epochs_per_trial,
            verbose=1
        )
        
        trained_weights = training_model.get_weights()
        
        # Stage 2: Evaluate with approximate layers
        print("→ Stage 2: Evaluating with approximate multipliers...")
        eval_model = build_model_for_evaluation(config, weights=trained_weights)
        
        test_loss, test_accuracy = eval_model.evaluate(x_test, y_test, verbose=0)
        train_test_loss, train_test_accuracy = training_model.evaluate(x_test, y_test, verbose=0)
        energy = estimate_energy(config)
        
        X_tried.append(config)
        y_tried.append(test_accuracy)
        
        result = {
            'config': config,
            'accuracy': test_accuracy,
            'accuracy_without_approx': train_test_accuracy,
            'accuracy_degradation': train_test_accuracy - test_accuracy,
            'loss': test_loss,
            'energy': energy,
            'history': history.history,
            'trial': trial
        }
        results.append(result)
        
        print(f"\n✓ Results:")
        print(f"  Accuracy (with approx):    {test_accuracy:.4f}")
        print(f"  Accuracy (without approx): {train_test_accuracy:.4f}")
        print(f"  Degradation:               {train_test_accuracy - test_accuracy:.4f}")
        print(f"  Energy:                    {energy:.2f}")
        print(f"  Best so far:               {max(y_tried):.4f}")
        
        del training_model
        del eval_model
        tf.keras.backend.clear_session()
    
    plot_nas_results(results, output_dir, 'two_stage')
    save_best_config_summary(results, output_dir)
    
    return results

##============================================================================##
## Main Execution
##============================================================================##
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NAS for CIFAR-10 with Approximate Computing')
    parser.add_argument('--trials', type=int, default=15, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per trial')
    parser.add_argument('--output_dir', type=str, default='nas_results_cifar', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"\nApproximate layers available: {APPROX_AVAILABLE}")
    print(f"Dataset: CIFAR-10")
    print(f"Trials: {args.trials}")
    print(f"Epochs per trial: {args.epochs}\n")
    
    results = bayesian_nas_fixed(
        num_trials=args.trials,
        epochs_per_trial=args.epochs,
        output_dir=args.output_dir
    )
    
    # Summary
    accuracies = [r['accuracy'] for r in results]
    best_idx = np.argmax(accuracies)
    best_result = results[best_idx]
    pareto_indices = compute_pareto_frontier(results)
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Trials: {len(results)}")
    print(f"Best Accuracy: {max(accuracies):.4f}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    
    degradations = [r['accuracy_degradation'] for r in results]
    print(f"Mean Accuracy Degradation: {np.mean(degradations):.4f} ± {np.std(degradations):.4f}")
    
    print(f"\nBest Configuration:")
    print(f"  {config_to_string(best_result['config'])}")
    print(f"  Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Energy: {best_result['energy']:.2f}")
    
    print(f"\nPareto Frontier: {len(pareto_indices)} configurations")
    for idx in pareto_indices[:3]:  # Show top 3
        r = results[idx]
        print(f"  Trial {r['trial']+1}: Acc={r['accuracy']:.4f}, Energy={r['energy']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()
