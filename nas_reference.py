"""
NAS implementation using reference architecture pattern.
Matches the proven two-stage training approach from reference implementation.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model_builder import build_model_for_training, build_model_for_evaluation
from training import evaluate_model

# Search space for reference architecture
SEARCH_SPACE = {
    'conv1_filters': [16, 24, 32, 48, 64],
    'conv1_kernel': [3, 5],
    'conv2_filters': [32, 48, 64, 96, 128],
    'conv2_kernel': [3, 5],
    'conv3_filters': [64, 96, 128, 192, 256],
    'conv3_kernel': [3, 5],
    'dense1_size': [64, 128, 256],
    'dropout_rate': [0.3, 0.4, 0.5],
}


def random_config():
    """Generate random configuration from search space"""
    return {
        'conv1_filters': np.random.choice(SEARCH_SPACE['conv1_filters']),
        'conv1_kernel': np.random.choice(SEARCH_SPACE['conv1_kernel']),
        'conv1_multiplier': None,  # Set during multiplier testing
        'conv2_filters': np.random.choice(SEARCH_SPACE['conv2_filters']),
        'conv2_kernel': np.random.choice(SEARCH_SPACE['conv2_kernel']),
        'conv2_multiplier': None,
        'conv3_filters': np.random.choice(SEARCH_SPACE['conv3_filters']),
        'conv3_kernel': np.random.choice(SEARCH_SPACE['conv3_kernel']),
        'conv3_multiplier': None,
        'dense1_size': np.random.choice(SEARCH_SPACE['dense1_size']),
        'dropout_rate': np.random.choice(SEARCH_SPACE['dropout_rate']),
    }


def mutate_config(config, mutation_rate=0.3):
    """Mutate configuration for evolutionary search"""
    new_config = config.copy()

    # Randomly mutate each parameter
    if np.random.random() < mutation_rate:
        new_config['conv1_filters'] = np.random.choice(SEARCH_SPACE['conv1_filters'])
    if np.random.random() < mutation_rate:
        new_config['conv1_kernel'] = np.random.choice(SEARCH_SPACE['conv1_kernel'])
    if np.random.random() < mutation_rate:
        new_config['conv2_filters'] = np.random.choice(SEARCH_SPACE['conv2_filters'])
    if np.random.random() < mutation_rate:
        new_config['conv2_kernel'] = np.random.choice(SEARCH_SPACE['conv2_kernel'])
    if np.random.random() < mutation_rate:
        new_config['conv3_filters'] = np.random.choice(SEARCH_SPACE['conv3_filters'])
    if np.random.random() < mutation_rate:
        new_config['conv3_kernel'] = np.random.choice(SEARCH_SPACE['conv3_kernel'])
    if np.random.random() < mutation_rate:
        new_config['dense1_size'] = np.random.choice(SEARCH_SPACE['dense1_size'])
    if np.random.random() < mutation_rate:
        new_config['dropout_rate'] = np.random.choice(SEARCH_SPACE['dropout_rate'])

    return new_config


def config_to_string(config):
    """Convert config to readable string"""
    return (f"C1:{config['conv1_filters']}f-{config['conv1_kernel']}k | "
            f"C2:{config['conv2_filters']}f-{config['conv2_kernel']}k | "
            f"C3:{config['conv3_filters']}f-{config['conv3_kernel']}k | "
            f"D:{config['dense1_size']}-drop{config['dropout_rate']}")


def run_nas_reference(x_train, y_train, x_val, y_val, input_shape, num_classes,
                     num_trials=20, epochs_per_trial=15, batch_size=64,
                     learning_rate=0.001, method='evolutionary',
                     test_multipliers=None, use_approximate_in_search=True,
                     logger=None):
    """
    Hardware-aware NAS with reference architecture pattern.
    Evaluates architectures with ALL approximate multipliers during search.

    Args:
        x_train, y_train: Training data
        x_val, y_val: Validation data
        input_shape: Input shape tuple
        num_classes: Number of classes
        num_trials: Number of NAS trials
        epochs_per_trial: Training epochs per trial
        batch_size: Training batch size
        learning_rate: Learning rate
        method: 'random' or 'evolutionary'
        test_multipliers: LIST of multiplier file paths for approximate evaluation
        use_approximate_in_search: If True, use approximate multipliers during NAS
        logger: Logger instance

    Returns:
        dict: Best configuration and results
    """
    if logger:
        logger.log(f"\n{'='*80}")
        logger.log(f"Hardware-Aware NAS with Reference Architecture")
        logger.log(f"{'='*80}")
        logger.log(f"Method: {method}")
        logger.log(f"Trials: {num_trials}")
        logger.log(f"Epochs per trial: {epochs_per_trial}")
        logger.log(f"Batch size: {batch_size}")
        logger.log(f"Approximate evaluation: {use_approximate_in_search}")
        if use_approximate_in_search and test_multipliers:
            logger.log(f"Testing with {len(test_multipliers)} multipliers")

    configs_tried = []
    fitness_scores = []
    results = []

    for trial in range(num_trials):
        if logger:
            logger.log(f"\n{'='*80}")
            logger.log(f"Trial {trial + 1}/{num_trials}")
            logger.log(f"{'='*80}")

        # Generate configuration
        if trial < 5 or method == 'random':
            # Random search for first 5 trials or if method is random
            config = random_config()
        else:
            # Evolutionary: mutate best config
            best_idx = np.argmax(fitness_scores)
            config = mutate_config(configs_tried[best_idx])

        if logger:
            logger.log(f"Config: {config_to_string(config)}")

        # Stage 1: Train with standard Conv2D
        if logger:
            logger.log(f"→ Training standard model...")

        model_std = build_model_for_training(config, input_shape, num_classes, learning_rate)

        history = model_std.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs_per_trial,
            batch_size=batch_size,
            verbose=0
        )

        std_accuracy = evaluate_model(model_std, x_val, y_val)
        trained_weights = model_std.get_weights()

        # Stage 2: Evaluate with ALL approximate multipliers (if enabled)
        multiplier_accuracies = []
        mean_approx_accuracy = std_accuracy
        mean_accuracy_drop = 0.0

        if use_approximate_in_search and test_multipliers:
            if logger:
                logger.log(f"→ Evaluating with {len(test_multipliers)} approximate multipliers...")

            for mul_file in test_multipliers:
                # Set multiplier for all blocks
                approx_config = config.copy()
                approx_config['conv1_multiplier'] = mul_file
                approx_config['conv2_multiplier'] = mul_file
                approx_config['conv3_multiplier'] = mul_file

                model_approx = build_model_for_evaluation(
                    approx_config, input_shape, num_classes,
                    learning_rate, weights=trained_weights
                )

                approx_accuracy = evaluate_model(model_approx, x_val, y_val)
                multiplier_accuracies.append(approx_accuracy)

                del model_approx
                keras.backend.clear_session()

            # Calculate mean performance across ALL multipliers
            mean_approx_accuracy = np.mean(multiplier_accuracies)
            mean_accuracy_drop = std_accuracy - mean_approx_accuracy

        # Fitness function: balance standard accuracy and robustness to approximate multipliers
        # We want high standard accuracy AND low AVERAGE drop across ALL multipliers
        if use_approximate_in_search and test_multipliers:
            # Multi-objective: 70% standard accuracy + 30% robustness (low average drop)
            # Robustness = average performance across all multipliers
            robustness_score = max(0, 1 - (mean_accuracy_drop / std_accuracy))
            fitness = 0.7 * std_accuracy + 0.3 * robustness_score
        else:
            # Single objective: just standard accuracy
            fitness = std_accuracy

        configs_tried.append(config)
        fitness_scores.append(fitness)

        result = {
            'trial': trial,
            'config': config,
            'std_accuracy': std_accuracy,
            'mean_approx_accuracy': mean_approx_accuracy,
            'mean_accuracy_drop': mean_accuracy_drop,
            'multiplier_accuracies': multiplier_accuracies,
            'fitness': fitness,
            'history': history.history
        }
        results.append(result)

        if logger:
            logger.log(f"✓ Standard accuracy: {std_accuracy:.4f}")
            if use_approximate_in_search and test_multipliers:
                drop_pct = (mean_accuracy_drop / std_accuracy) * 100 if std_accuracy > 0 else 0
                logger.log(f"  Mean approx accuracy (across {len(test_multipliers)} muls): {mean_approx_accuracy:.4f}")
                logger.log(f"  Mean accuracy drop: {mean_accuracy_drop:.4f} ({drop_pct:.2f}%)")
                logger.log(f"  Best/Worst approx: {max(multiplier_accuracies):.4f} / {min(multiplier_accuracies):.4f}")
                logger.log(f"  Fitness score: {fitness:.4f}")
            logger.log(f"  Best fitness so far: {max(fitness_scores):.4f}")

        # Cleanup
        del model_std
        keras.backend.clear_session()

    # Find best configuration based on fitness
    best_idx = np.argmax(fitness_scores)
    best_config = configs_tried[best_idx]
    best_result = results[best_idx]

    if logger:
        logger.log(f"\n{'='*80}")
        logger.log(f"NAS Complete")
        logger.log(f"{'='*80}")
        logger.log(f"Best configuration:")
        logger.log(f"  {config_to_string(best_config)}")
        logger.log(f"  Standard accuracy: {best_result['std_accuracy']:.4f}")
        if use_approximate_in_search and test_multipliers:
            logger.log(f"  Mean approx accuracy: {best_result['mean_approx_accuracy']:.4f}")
            logger.log(f"  Mean accuracy drop: {best_result['mean_accuracy_drop']:.4f}")
            if best_result['multiplier_accuracies']:
                logger.log(f"  Best/Worst multiplier: {max(best_result['multiplier_accuracies']):.4f} / {min(best_result['multiplier_accuracies']):.4f}")
        logger.log(f"  Fitness score: {best_result['fitness']:.4f}")
        logger.log(f"  Mean fitness: {np.mean(fitness_scores):.4f} ± {np.std(fitness_scores):.4f}")

    return {
        'best_config': best_config,
        'best_result': best_result,
        'all_configs': configs_tried,
        'all_fitness_scores': fitness_scores,
        'results': results
    }
