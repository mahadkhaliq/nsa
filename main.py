"""
Main script for Neural Architecture Search with Approximate Multipliers
Features:
- Configurable learning rate
- Saves predictions, misclassifications, confusion matrices
- Generates training curves and analysis plots
- Compatible with TensorFlow 2.1.0+
"""

from tensorflow import keras
import tensorflow as tf
from nas import run_nas
from model_builder import build_model, count_model_params, estimate_flops
from operations import get_search_space

# Print TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")
from training import evaluate_model, train_model
from logger import NASLogger
from validation_utils import save_predictions, plot_confidence_distribution
import numpy as np
from dataloader import load_dataset
import os
import glob
import argparse
import matplotlib
matplotlib.use('Agg')  # For headless systems

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='NAS with Approximate Computing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'cifar10', 'cifar100', 'fashion_mnist'],
                       help='Dataset to use')
    parser.add_argument('--train_samples', type=int, default=5000,
                       help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=1000,
                       help='Number of validation samples')
    
    # NAS parameters
    parser.add_argument('--nas_trials', type=int, default=20,
                       help='Number of NAS trials')
    parser.add_argument('--nas_blocks', type=int, default=4,
                       help='Number of blocks in architecture')
    parser.add_argument('--nas_method', type=str, default='evolutionary',
                       choices=['random', 'evolutionary'],
                       help='NAS search method')
    parser.add_argument('--skip_nas', action='store_true',
                       help='Skip NAS and use predefined architecture')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs for final model')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--use_lr_schedule', action='store_true',
                       help='Use learning rate scheduling')
    
    # Multiplier testing
    parser.add_argument('--multiplier_dir', type=str, default='./multipliers',
                       help='Directory containing .bin multiplier files')
    parser.add_argument('--skip_multipliers', action='store_true',
                       help='Skip multiplier testing')
    parser.add_argument('--test_all_multipliers', action='store_true',
                       help='Test all multipliers (default: first 10)')
    
    # Output and visualization
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logs')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions and visualizations')
    parser.add_argument('--top_k_errors', type=int, default=20,
                       help='Number of top misclassifications to save')
    
    return parser.parse_args()

def get_class_names(dataset):
    """Get class names for dataset"""
    class_names_dict = {
        'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck'],
        'mnist': [str(i) for i in range(10)],
        'fashion_mnist': ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'cifar100': [str(i) for i in range(100)]
    }
    return class_names_dict.get(dataset, None)

def get_predefined_architecture(dataset):
    """Get a good predefined architecture for quick testing"""
    if dataset in ['mnist', 'fashion_mnist']:
        return [
            {'op': 'conv3x3', 'filters': 64, 'use_bn': True},
            {'op': 'max_pool', 'filters': 64, 'use_bn': False},
            {'op': 'conv5x5', 'filters': 128, 'use_bn': True},
            {'op': 'conv1x1', 'filters': 64, 'use_bn': False}
        ]
    else:  # CIFAR
        return [
            {'op': 'conv3x3', 'filters': 32, 'use_bn': True},
            {'op': 'conv3x3', 'filters': 64, 'use_bn': True},
            {'op': 'max_pool', 'filters': 64, 'use_bn': False},
            {'op': 'conv3x3', 'filters': 128, 'use_bn': True},
            {'op': 'conv1x1', 'filters': 128, 'use_bn': False}
        ]

def create_lr_schedule(initial_lr, epochs):
    """Create learning rate schedule"""
    def scheduler(epoch, lr):
        if epoch < epochs // 3:
            return initial_lr
        elif epoch < 2 * epochs // 3:
            return initial_lr * 0.1
        else:
            return initial_lr * 0.01
    return scheduler

def main():
    args = parse_args()
    
    # Initialize logger
    logger = NASLogger(log_dir=args.log_dir)
    
    # Create results directory
    results_dir = os.path.join(args.log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    logger.log("="*70)
    logger.log("CONFIGURATION")
    logger.log("="*70)
    logger.log(f"Dataset: {args.dataset}")
    logger.log(f"Training samples: {args.train_samples}")
    logger.log(f"Validation samples: {args.val_samples}")
    logger.log(f"NAS trials: {args.nas_trials}")
    logger.log(f"NAS blocks: {args.nas_blocks}")
    logger.log(f"NAS method: {args.nas_method}")
    logger.log(f"Training epochs: {args.epochs}")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Learning rate: {args.learning_rate}")
    logger.log(f"LR scheduling: {'Yes' if args.use_lr_schedule else 'No'}")
    logger.log(f"Multiplier dir: {args.multiplier_dir}")
    logger.log(f"Save predictions: {'Yes' if args.save_predictions else 'No'}")
    logger.log(f"Results dir: {results_dir}")
    
    # ========================================================================
    # STEP 0: Load and Prepare Dataset
    # ========================================================================
    logger.log_section("STEP 0: Loading Dataset")
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(
        args.dataset, 
        num_val=6000
    )
    
    # Handle grayscale images
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)
        x_test = np.expand_dims(x_test, -1)
    
    # Extract shape information
    height, width, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    num_classes = len(np.unique(y_train))
    
    logger.log(f"Input shape: ({height}, {width}, {channels})")
    logger.log(f"Number of classes: {num_classes}")
    logger.log(f"Training samples (full): {len(x_train)}")
    logger.log(f"Validation samples (full): {len(x_val)}")
    logger.log(f"Test samples: {len(x_test)}")
    
    # Use subset for faster experimentation
    x_train = x_train[:args.train_samples]
    y_train = y_train[:args.train_samples]
    x_val = x_val[:args.val_samples]
    y_val = y_val[:args.val_samples]
    
    logger.log(f"\nUsing subset:")
    logger.log(f"  Training: {len(x_train)} samples")
    logger.log(f"  Validation: {len(x_val)} samples")
    
    # Get class names
    class_names = get_class_names(args.dataset)
    if class_names:
        logger.log(f"  Classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
    
    # ========================================================================
    # STEP 1: Neural Architecture Search
    # ========================================================================
    if args.skip_nas:
        logger.log_section("STEP 1: Using Predefined Architecture (NAS Skipped)")
        best_architecture = get_predefined_architecture(args.dataset)
        logger.log(f"Architecture: {best_architecture}")
        
        # Quick evaluation
        search_space_std = get_search_space(use_approximate=False, include_advanced=True)
        model_std = build_model(
            best_architecture, search_space_std, 
            input_shape=(height, width, channels), 
            num_classes=num_classes,
            learning_rate=args.learning_rate
        )
        train_model(model_std, x_train, y_train, x_val, y_val, 
                   epochs=10, batch_size=args.batch_size, verbose=0)
        quick_acc = evaluate_model(model_std, x_val, y_val)
        logger.log(f"Quick evaluation accuracy: {quick_acc:.4f}")
        
        best_std = {
            'architecture': best_architecture,
            'accuracy': quick_acc
        }
        results_std = [best_std]
        
        del model_std
        keras.backend.clear_session()
    else:
        logger.log_section("STEP 1: Neural Architecture Search")
        results_std, best_std = run_nas(
            x_train, y_train, x_val, y_val,
            num_trials=args.nas_trials,
            num_blocks=args.nas_blocks,
            use_approximate=False,
            logger=logger,
            input_shape=(height, width, channels),
            num_classes=num_classes,
            method=args.nas_method
        )
        
        # Log top 3 architectures
        logger.log("\nTop 3 Architectures:")
        sorted_results = sorted(results_std, key=lambda x: x['accuracy'], reverse=True)
        for i, result in enumerate(sorted_results[:3]):
            logger.log(f"{i+1}. Accuracy: {result['accuracy']:.4f}")
            logger.log(f"   {result['architecture']}")
    
    # ========================================================================
    # STEP 2: Train Best Architecture
    # ========================================================================
    logger.log_section("STEP 2: Training Best Architecture")
    logger.log(f"Best Architecture: {best_std['architecture']}")
    
    # Build model
    search_space_std = get_search_space(use_approximate=False, include_advanced=True)
    model_std = build_model(
        best_std['architecture'], 
        search_space_std, 
        input_shape=(height, width, channels),
        num_classes=num_classes,
        learning_rate=args.learning_rate
    )
    
    # Calculate model complexity
    params = count_model_params(model_std)
    flops = estimate_flops(best_std['architecture'], (height, width, channels))
    
    logger.log(f"\nModel Complexity:")
    logger.log(f"  Trainable Parameters: {params:,}")
    logger.log(f"  Estimated FLOPs: {flops:,}")
    logger.log(f"  Model Size (approx): {params * 4 / (1024**2):.2f} MB (float32)")
    
    # Setup learning rate schedule if requested
    callbacks = []
    if args.use_lr_schedule:
        lr_scheduler = keras.callbacks.LearningRateScheduler(
            create_lr_schedule(args.learning_rate, args.epochs)
        )
        callbacks.append(lr_scheduler)
        logger.log(f"\nUsing learning rate schedule:")
        logger.log(f"  Epochs 0-{args.epochs//3}: {args.learning_rate}")
        logger.log(f"  Epochs {args.epochs//3+1}-{2*args.epochs//3}: {args.learning_rate*0.1}")
        logger.log(f"  Epochs {2*args.epochs//3+1}-{args.epochs}: {args.learning_rate*0.01}")
    
    # Train with progress
    logger.log(f"\nTraining for {args.epochs} epochs...")
    
    if callbacks:
        history = model_std.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = train_model(
            model_std, x_train, y_train, x_val, y_val, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1
        )
    
    # Evaluate
    std_accuracy = evaluate_model(model_std, x_val, y_val)
    logger.log_training(epochs=args.epochs, std_accuracy=std_accuracy)
    logger.log_best_architecture(best_std['architecture'], std_accuracy)
    
    # Save trained weights
    trained_weights = model_std.get_weights()
    
    # Training summary
    logger.log(f"\nTraining Summary:")
    logger.log(f"  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.log(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    logger.log(f"  Final Training Loss: {history.history['loss'][-1]:.4f}")
    logger.log(f"  Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    # Check for overfitting
    train_val_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
    if train_val_gap > 0.1:
        logger.log(f"  ⚠ Warning: Large train-val gap ({train_val_gap:.4f}) suggests overfitting")
    
    # Save training curves
    logger.log("\nSaving training curves...")
    logger.log_training_curves(history, results_dir, 'standard_model')
    
    # Save standard model weights
    weights_file = os.path.join(results_dir, 'standard_model_weights.h5')
    model_std.save_weights(weights_file)
    logger.log(f"\n✓ Standard model weights saved to {weights_file}")
    
    # Save predictions if requested
    if args.save_predictions:
        logger.log("\nSaving standard model predictions and visualizations...")
        std_pred_results = save_predictions(
            model_std, x_val, y_val, 
            results_dir, 
            'standard', 
            class_names,
            top_k=args.top_k_errors
        )
        logger.log(f"  ✓ Predictions saved to {results_dir}/")
        logger.log(f"  ✓ Confusion matrix saved")
        logger.log(f"  ✓ Top {args.top_k_errors} misclassifications visualized")
    
    # ========================================================================
    # STEP 3: Test with Approximate Multipliers
    # ========================================================================
    if args.skip_multipliers:
        logger.log_section("STEP 3: Multiplier Testing Skipped")
    else:
        logger.log_section("STEP 3: Testing Approximate Multipliers")
        
        # Find multiplier files
        multiplier_files = sorted(glob.glob(os.path.join(args.multiplier_dir, '*.bin')))
        
        if not multiplier_files:
            logger.log(f"⚠ No .bin files found in {args.multiplier_dir}")
            logger.log("Skipping multiplier testing...")
        else:
            # Limit number of multipliers if not testing all
            if not args.test_all_multipliers and len(multiplier_files) > 10:
                logger.log(f"Found {len(multiplier_files)} multipliers")
                logger.log(f"Testing first 10 (use --test_all_multipliers to test all)")
                multiplier_files = multiplier_files[:10]
            
            logger.log(f"Testing {len(multiplier_files)} multipliers")
            logger.log(f"Standard model accuracy: {std_accuracy:.4f}")
            logger.log(f"Using saved weights from: {weights_file}\n")
            
            approx_results = []
            all_pred_results = []
            
            multiplier_results_dir = os.path.join(results_dir, 'multipliers')
            os.makedirs(multiplier_results_dir, exist_ok=True)
            
            for idx, mul_file in enumerate(multiplier_files):
                mul_name = os.path.basename(mul_file)
                logger.log(f"[{idx+1}/{len(multiplier_files)}] Testing: {mul_name}")
                
                try:
                    # Build approximate model with same architecture
                    search_space_approx = get_search_space(
                        use_approximate=True, 
                        mul_map_file=mul_file,
                        include_advanced=True
                    )
                    
                    model_approx = build_model(
                        best_std['architecture'], 
                        search_space_approx,
                        input_shape=(height, width, channels),
                        num_classes=num_classes,
                        learning_rate=args.learning_rate
                    )
                    
                    # Load weights from standard model
                    # NOTE: Model should already be built by build_model(), don't rebuild
                    try:
                        model_approx.load_weights(weights_file)
                        if idx == 0:  # Log only for first multiplier
                            logger.log(f"  ✓ Weights loaded successfully (no rebuild needed)")
                    except Exception as e:
                        logger.log(f"  Warning: Weight loading failed, rebuilding: {e}")
                        model_approx.build(input_shape=(None, height, width, channels))
                        model_approx.load_weights(weights_file)

                    # Calibrate with validation data (critical for quantization)
                    calibration_out = model_approx.predict(x_val[:100], verbose=0)
                    if idx == 0:  # Log only for first multiplier
                        logger.log(f"  Calibration output range: [{calibration_out.min():.4f}, {calibration_out.max():.4f}]")

                    # Evaluate with approximate multiplier on TEST set
                    approx_accuracy = evaluate_model(model_approx, x_test, y_test)
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
                    
                    # Save predictions for this multiplier if requested
                    if args.save_predictions:
                        mul_pred_results = save_predictions(
                            model_approx, x_val, y_val,
                            multiplier_results_dir,
                            mul_name.replace('.bin', ''),
                            class_names,
                            top_k=10
                        )
                        all_pred_results.append(mul_pred_results)
                    
                except Exception as e:
                    logger.log(f"  ✗ Failed: {e}\n")
                finally:
                    # Cleanup
                    if 'model_approx' in locals():
                        del model_approx
                    keras.backend.clear_session()
    
    # ========================================================================
            # ================================================================
            # STEP 4: Analysis and Summary
            # ================================================================
            logger.log_section("STEP 4: Results Analysis")
            
            if approx_results:
                # Summary statistics
                logger.log_summary(std_accuracy, approx_results)
                
                # Detailed table
                logger.log(f"\n{'Multiplier':<30} {'Accuracy':<12} {'Drop':<12} {'Drop %':<10}")
                logger.log("-" * 70)
                
                # Sort by accuracy (best to worst)
                approx_results.sort(key=lambda x: x['accuracy'], reverse=True)
                
                for result in approx_results:
                    logger.log(f"{result['multiplier']:<30} {result['accuracy']:<12.4f} "
                             f"{result['drop']:<12.4f} {result['drop_percent']:<10.2f}%")
                
                # Best multiplier
                logger.log_section("BEST APPROXIMATE MULTIPLIER")
                best_approx = approx_results[0]
                logger.log(f"Multiplier:  {best_approx['multiplier']}")
                logger.log(f"Accuracy:    {best_approx['accuracy']:.4f}")
                logger.log(f"Drop:        {best_approx['drop']:.4f} ({best_approx['drop_percent']:.2f}%)")
                
                # Top 5 multipliers
                logger.log(f"\nTop 5 Multipliers:")
                for i, result in enumerate(approx_results[:5]):
                    logger.log(f"{i+1}. {result['multiplier']:<30} "
                             f"Acc: {result['accuracy']:.4f} "
                             f"Drop: {result['drop_percent']:+.2f}%")
                
                # Efficiency estimate
                logger.log(f"\nEstimated Energy Savings:")
                excellent_count = sum(1 for r in approx_results if abs(r['drop']) <= 0.01)
                good_count = sum(1 for r in approx_results if 0.01 < abs(r['drop']) <= 0.05)
                logger.log(f"  Excellent (≤1% drop): {excellent_count} multipliers")
                logger.log(f"  Good (1-5% drop):     {good_count} multipliers")
                logger.log(f"  Potential energy savings: 30-50% (typical for approximate multipliers)")
                logger.log(f"  Model parameters: {params:,}")
                logger.log(f"  Estimated MACs per inference: {flops//2:,}")
                
                # Generate comparison plots if predictions were saved
                if args.save_predictions and all_pred_results:
                    logger.log("\nGenerating comparison plots...")
                    try:
                        if 'std_pred_results' in locals():
                            plot_confidence_distribution(
                                [std_pred_results] + all_pred_results[:5],  # Compare with top 5
                                results_dir
                            )
                        logger.log(f"  ✓ Confidence distribution plots saved")
                    except Exception as e:
                        logger.log(f"  ✗ Failed to generate plots: {e}")
            else:
                logger.log("⚠ No multipliers were successfully tested")
    
    # ========================================================================
    # Final Cleanup and Summary
    # ========================================================================
    logger.log_section("FINAL SUMMARY")
    logger.log(f"Dataset: {args.dataset}")
    logger.log(f"Best Architecture: {best_std['architecture']}")
    logger.log(f"Standard Model Accuracy: {std_accuracy:.4f}")
    logger.log(f"Model Parameters: {params:,}")
    logger.log(f"Estimated FLOPs: {flops:,}")
    logger.log(f"Learning Rate: {args.learning_rate}")
    
    if not args.skip_multipliers and 'approx_results' in locals() and approx_results:
        logger.log(f"\nMultiplier Testing:")
        logger.log(f"  Total tested: {len(approx_results)}")
        logger.log(f"  Best accuracy: {approx_results[0]['accuracy']:.4f}")
        logger.log(f"  Best multiplier: {approx_results[0]['multiplier']}")
        
        # Count quality categories
        excellent = sum(1 for r in approx_results if abs(r['drop']) <= 0.01)
        good = sum(1 for r in approx_results if 0.01 < abs(r['drop']) <= 0.05)
        medium = sum(1 for r in approx_results if 0.05 < abs(r['drop']) <= 0.10)
        poor = sum(1 for r in approx_results if abs(r['drop']) > 0.10)
        
        logger.log(f"\nQuality Distribution:")
        logger.log(f"  Excellent (≤1%):   {excellent}")
        logger.log(f"  Good (1-5%):       {good}")
        logger.log(f"  Medium (5-10%):    {medium}")
        logger.log(f"  Poor (>10%):       {poor}")
    
    logger.log(f"\nOutput Locations:")
    logger.log(f"  Logs: {args.log_dir}/")
    logger.log(f"  Results: {results_dir}/")
    if args.save_predictions:
        logger.log(f"  Predictions: {results_dir}/standard_predictions.json")
        logger.log(f"  Confusion Matrix: {results_dir}/standard_confusion_matrix.png")
        logger.log(f"  Training Curves: {results_dir}/standard_model_training_curves.png")
        if not args.skip_multipliers:
            logger.log(f"  Multiplier Results: {multiplier_results_dir}/")
    
    # Close logger and save results
    logger.close()
    
    # Cleanup
    del model_std
    keras.backend.clear_session()
    
    print("\n✓ All done! Check the results directory for detailed outputs.")

if __name__ == '__main__':
    main()
