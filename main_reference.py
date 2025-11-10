"""
Main script for NAS with Reference Architecture Pattern.
Two-stage approach: Train with standard Conv2D, evaluate with approximate multipliers.
"""
import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')

from dataloader import load_dataset
from nas_reference import run_nas_reference, config_to_string
from model_builder import build_model_for_training, build_model_for_evaluation
from training import evaluate_model
from logger import NASLogger
from rtamt_verifier import NASVerifier, estimate_energy_ratio
from pareto_visualization import plot_pareto_frontier, plot_accuracy_energy_scatter, plot_nas_verification_summary

# GPU memory management
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✓ GPU memory growth enabled for {len(physical_devices)} device(s)")
    except:
        print("⚠ Could not enable GPU memory growth")

print(f"TensorFlow version: {tf.__version__}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='NAS with Reference Architecture and Approximate Computing'
    )

    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10', 'cifar100', 'fashion_mnist'])
    parser.add_argument('--train_samples', type=int, default=5000)
    parser.add_argument('--val_samples', type=int, default=1000)

    # NAS
    parser.add_argument('--nas_trials', type=int, default=15)
    parser.add_argument('--nas_method', type=str, default='evolutionary',
                       choices=['random', 'evolutionary'])
    parser.add_argument('--skip_nas', action='store_true')
    parser.add_argument('--nas_use_multipliers', action='store_true',
                       help='Use ALL multipliers for hardware-aware NAS')
    parser.add_argument('--nas_num_multipliers', type=int, default=10,
                       help='Number of multipliers to use for NAS (default: 10)')

    # Training
    parser.add_argument('--epochs_per_trial', type=int, default=15)
    parser.add_argument('--final_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # Multipliers
    parser.add_argument('--multiplier_dir', type=str, default='./multipliers')
    parser.add_argument('--skip_multipliers', action='store_true')
    parser.add_argument('--test_all_multipliers', action='store_true')

    # RTAMT Formal Verification
    parser.add_argument('--use_rtamt', action='store_true',
                       help='Enable RTAMT formal verification during NAS')
    parser.add_argument('--rtamt_min_accuracy', type=float, default=0.7,
                       help='Minimum accuracy threshold for RTAMT verification')
    parser.add_argument('--rtamt_max_drop', type=float, default=10.0,
                       help='Maximum accuracy drop percentage for robustness verification')

    # Output
    parser.add_argument('--log_dir', type=str, default='logs')

    return parser.parse_args()


def get_default_config(dataset):
    """Get default configuration based on dataset"""
    if dataset in ['mnist', 'fashion_mnist']:
        return {
            'conv1_filters': 16,
            'conv1_kernel': 3,
            'conv1_multiplier': None,
            'conv2_filters': 32,
            'conv2_kernel': 3,
            'conv2_multiplier': None,
            'conv3_filters': 64,
            'conv3_kernel': 3,
            'conv3_multiplier': None,
            'dense1_size': 64,
            'dropout_rate': 0.3
        }
    else:  # CIFAR
        return {
            'conv1_filters': 32,
            'conv1_kernel': 3,
            'conv1_multiplier': None,
            'conv2_filters': 64,
            'conv2_kernel': 3,
            'conv2_multiplier': None,
            'conv3_filters': 128,
            'conv3_kernel': 3,
            'conv3_multiplier': None,
            'dense1_size': 128,
            'dropout_rate': 0.3
        }


def main():
    args = parse_args()

    # Create logger
    logger = NASLogger(log_dir=args.log_dir)
    logger.log_section("NAS with Reference Architecture Pattern")
    logger.log(f"Dataset: {args.dataset}")
    logger.log(f"TensorFlow version: {tf.__version__}")

    # Load dataset
    logger.log_section("STEP 1: Loading Dataset")
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(args.dataset)
    x_train, y_train = x_train[:args.train_samples], y_train[:args.train_samples]
    x_val, y_val = x_val[:args.val_samples], y_val[:args.val_samples]

    # Handle dimensions
    if len(x_train.shape) == 3:
        x_train = x_train[..., None]
        x_val = x_val[..., None]
        x_test = x_test[..., None]

    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    logger.log(f"Training samples: {len(x_train)}")
    logger.log(f"Validation samples: {len(x_val)}")
    logger.log(f"Test samples: {len(x_test)}")
    logger.log(f"Input shape: {input_shape}")
    logger.log(f"Number of classes: {num_classes}")

    # ========================================================================
    # STEP 2: Neural Architecture Search or Use Default
    # ========================================================================
    if args.skip_nas:
        logger.log_section("STEP 2: Using Default Architecture")
        best_config = get_default_config(args.dataset)
        logger.log(f"Config: {config_to_string(best_config)}")
    else:
        logger.log_section("STEP 2: Neural Architecture Search")

        # Use hardware-aware NAS if multipliers requested
        nas_multipliers = None
        if args.nas_use_multipliers:
            multiplier_files = sorted(glob.glob(os.path.join(args.multiplier_dir, '*.bin')))
            if multiplier_files:
                nas_multipliers = multiplier_files[:args.nas_num_multipliers]
                logger.log(f"Hardware-aware NAS enabled with {len(nas_multipliers)} multipliers")
            else:
                logger.log(f"⚠ No multipliers found in {args.multiplier_dir}, using standard NAS")
        else:
            logger.log("Standard NAS (no approximate multiplier evaluation)")

        nas_results = run_nas_reference(
            x_train, y_train, x_val, y_val,
            input_shape, num_classes,
            num_trials=args.nas_trials,
            epochs_per_trial=args.epochs_per_trial,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            method=args.nas_method,
            test_multipliers=nas_multipliers,
            use_approximate_in_search=nas_multipliers is not None,
            logger=logger,
            use_rtamt_verification=args.use_rtamt,
            min_accuracy_threshold=args.rtamt_min_accuracy,
            max_accuracy_drop_percent=args.rtamt_max_drop
        )
        best_config = nas_results['best_config']

    # ========================================================================
    # STEP 3: Train Final Model with Best Architecture
    # ========================================================================
    logger.log_section("STEP 3: Training Final Model")
    logger.log(f"Architecture: {config_to_string(best_config)}")
    logger.log(f"Training for {args.final_epochs} epochs...")

    model_std = build_model_for_training(
        best_config, input_shape, num_classes, args.learning_rate
    )

    history = model_std.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    std_accuracy = evaluate_model(model_std, x_val, y_val)
    trained_weights = model_std.get_weights()

    logger.log(f"\n✓ Final model validation accuracy: {std_accuracy:.4f}")
    logger.log(f"  Training accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.log(f"  Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Save weights
    weights_file = os.path.join(args.log_dir, 'standard_model_weights.h5')
    model_std.save_weights(weights_file)
    logger.log(f"✓ Weights saved to {weights_file}")

    # ========================================================================
    # STEP 4: Test with Approximate Multipliers
    # ========================================================================
    if args.skip_multipliers:
        logger.log_section("STEP 4: Multiplier Testing Skipped")
    else:
        logger.log_section("STEP 4: Testing Approximate Multipliers")

        multiplier_files = sorted(glob.glob(os.path.join(args.multiplier_dir, '*.bin')))

        if not multiplier_files:
            logger.log(f"⚠ No .bin files found in {args.multiplier_dir}")
        else:
            if not args.test_all_multipliers and len(multiplier_files) > 10:
                logger.log(f"Found {len(multiplier_files)} multipliers, testing first 10")
                multiplier_files = multiplier_files[:10]

            logger.log(f"Testing {len(multiplier_files)} multipliers")
            logger.log(f"Standard accuracy: {std_accuracy:.4f}\n")

            results = []

            for idx, mul_file in enumerate(multiplier_files):
                mul_name = os.path.basename(mul_file)
                logger.log(f"[{idx+1}/{len(multiplier_files)}] {mul_name}")

                # Create config with this multiplier
                test_config = best_config.copy()
                test_config['conv1_multiplier'] = mul_file
                test_config['conv2_multiplier'] = mul_file
                test_config['conv3_multiplier'] = mul_file

                # Build and evaluate
                model_approx = build_model_for_evaluation(
                    test_config, input_shape, num_classes,
                    learning_rate=args.learning_rate,
                    weights=trained_weights
                )

                approx_acc = evaluate_model(model_approx, x_val, y_val)
                drop_pct = ((std_accuracy - approx_acc) / std_accuracy) * 100

                results.append({
                    'name': mul_name,
                    'accuracy': approx_acc,
                    'drop_pct': drop_pct
                })

                logger.log(f"  Accuracy: {approx_acc:.4f}  Drop: {drop_pct:6.2f}%")

                del model_approx
                keras.backend.clear_session()

            # Summary
            logger.log(f"\n{'='*80}")
            logger.log("Multiplier Testing Summary")
            logger.log(f"{'='*80}")

            results_sorted = sorted(results, key=lambda x: x['drop_pct'])

            # Quality categories
            excellent = [r for r in results if r['drop_pct'] <= 1]
            good = [r for r in results if 1 < r['drop_pct'] <= 5]
            medium = [r for r in results if 5 < r['drop_pct'] <= 10]
            poor = [r for r in results if r['drop_pct'] > 10]

            logger.log(f"\nQuality Distribution:")
            logger.log(f"  Excellent (≤1% drop):  {len(excellent)}")
            logger.log(f"  Good (1-5% drop):      {len(good)}")
            logger.log(f"  Medium (5-10% drop):   {len(medium)}")
            logger.log(f"  Poor (>10% drop):      {len(poor)}")

            logger.log(f"\nTop 5 Best Multipliers:")
            for i, r in enumerate(results_sorted[:5]):
                logger.log(f"  {i+1}. {r['name']:20s}  Acc: {r['accuracy']:.4f}  Drop: {r['drop_pct']:6.2f}%")

            logger.log(f"\nTop 5 Worst Multipliers:")
            for i, r in enumerate(results_sorted[-5:]):
                logger.log(f"  {i+1}. {r['name']:20s}  Acc: {r['accuracy']:.4f}  Drop: {r['drop_pct']:6.2f}%")

    # ========================================================================
    # STEP 5: Generate RTAMT Verification Report (if enabled)
    # ========================================================================
    if args.use_rtamt and not args.skip_nas and 'verifier' in nas_results and nas_results['verifier']:
        logger.log_section("STEP 5: Formal Verification Report")

        verifier = nas_results['verifier']
        best_result = nas_results['best_result']

        # Prepare multiplier results for verification report
        multiplier_results_for_report = []
        if 'verification' in best_result and best_result['verification'] and best_result['verification']['pareto']:
            for pareto_point in best_result['verification']['pareto']:
                multiplier_results_for_report.append({
                    'name': pareto_point['multiplier'],
                    'accuracy': pareto_point['accuracy'],
                    'energy_ratio': pareto_point['energy_ratio']
                })

        # Generate comprehensive verification report
        if multiplier_results_for_report:
            training_history = {
                'accuracy': best_result['history']['accuracy'],
                'val_accuracy': best_result['history']['val_accuracy'],
                'loss': best_result['history']['loss']
            }

            verification_report = verifier.generate_verification_report(
                config=best_config,
                training_history=training_history,
                std_accuracy=best_result['std_accuracy'],
                multiplier_results=multiplier_results_for_report
            )

            # Print summary
            summary_str = verifier.get_summary_string(verification_report)
            logger.log(summary_str)

            # Save verification report as JSON
            import json
            from logger import convert_to_json_serializable
            report_file = os.path.join(args.log_dir, f'verification_report_{logger.timestamp}.json')
            with open(report_file, 'w') as f:
                json_safe_report = convert_to_json_serializable(verification_report)
                json.dump(json_safe_report, f, indent=2)
            logger.log(f"\nVerification report saved to: {report_file}")

            # Generate visualizations
            logger.log(f"\nGenerating Pareto frontier visualizations...")

            # Pareto frontier plot
            pareto_plot = os.path.join(args.log_dir, f'pareto_frontier_{logger.timestamp}.png')
            pareto_stats = plot_pareto_frontier(
                multiplier_results_for_report,
                output_file=pareto_plot,
                title=f'Accuracy-Energy Pareto Frontier ({args.dataset.upper()})'
            )
            logger.log(f"  Pareto frontier plot: {pareto_plot}")
            logger.log(f"  Pareto-optimal points: {pareto_stats['pareto_optimal_count']}/{pareto_stats['total_points']}")

            # Scatter plot
            scatter_plot = os.path.join(args.log_dir, f'accuracy_energy_scatter_{logger.timestamp}.png')
            plot_accuracy_energy_scatter(multiplier_results_for_report, output_file=scatter_plot)
            logger.log(f"  Scatter plot: {scatter_plot}")

            # Verification summary plot
            summary_plot = os.path.join(args.log_dir, f'verification_summary_{logger.timestamp}.png')
            plot_nas_verification_summary(verification_report, output_file=summary_plot)
            logger.log(f"  Verification summary: {summary_plot}")

    logger.log(f"\n{'='*80}")
    logger.log("Experiment Complete")
    logger.log(f"Results saved to: {args.log_dir}/")
    logger.log(f"{'='*80}")


if __name__ == '__main__':
    main()
