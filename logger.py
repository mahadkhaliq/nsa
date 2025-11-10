import logging
import os
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

class NASLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'nas_run_{self.timestamp}.log')
        self.json_file = os.path.join(log_dir, f'nas_results_{self.timestamp}.json')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.results = {
            'start_time': self.timestamp,
            'nas_results': [],
            'multiplier_results': [],
            'best_architecture': None,
            'summary': {}
        }
        
        self.logger.info("="*70)
        self.logger.info("NAS with Approximate Multipliers - Started")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("="*70)
    
    def log(self, message):
        self.logger.info(message)
    
    def log_section(self, title):
        self.logger.info("\n" + "="*70)
        self.logger.info(title)
        self.logger.info("="*70)
    
    def log_nas_trial(self, trial, total, architecture, accuracy):
        self.logger.info(f"Trial {trial}/{total}")
        self.logger.info(f"  Architecture: {architecture}")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        
        self.results['nas_results'].append({
            'trial': int(trial),
            'architecture': convert_to_json_serializable(architecture),
            'accuracy': float(accuracy),
            'timestamp': datetime.now().isoformat()
        })
    
    def log_best_architecture(self, architecture, accuracy):
        self.logger.info("\n" + "="*60)
        self.logger.info("BEST ARCHITECTURE FOUND")
        self.logger.info("="*60)
        self.logger.info(f"Architecture: {architecture}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info("="*60)
        
        self.results['best_architecture'] = {
            'architecture': convert_to_json_serializable(architecture),
            'accuracy': float(accuracy),
            'timestamp': datetime.now().isoformat()
        }
    
    def log_training(self, epochs, std_accuracy):
        self.logger.info(f"Training completed: {epochs} epochs")
        self.logger.info(f"✓ Standard model accuracy: {std_accuracy:.4f}")
    
    def log_multiplier_test(self, mul_name, accuracy, drop, drop_percent):
        self.logger.info(f"Testing: {mul_name}")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Drop:     {drop:.4f} ({drop_percent:.2f}%)")
        
        self.results['multiplier_results'].append({
            'multiplier': str(mul_name),
            'accuracy': float(accuracy),
            'drop': float(drop),
            'drop_percent': float(drop_percent),
            'timestamp': datetime.now().isoformat()
        })
    
    def log_summary(self, std_accuracy, approx_results):
        self.log_section("FINAL RESULTS SUMMARY")
        
        self.logger.info(f"Standard Accuracy: {std_accuracy:.4f}")
        self.logger.info(f"Total Multipliers Tested: {len(approx_results)}")
        
        if approx_results:
            accuracies = [r['accuracy'] for r in approx_results]
            drops = [r['drop'] for r in approx_results]
            
            self.logger.info(f"\nAccuracy Statistics:")
            self.logger.info(f"  Best:  {max(accuracies):.4f}")
            self.logger.info(f"  Worst: {min(accuracies):.4f}")
            self.logger.info(f"  Mean:  {sum(accuracies)/len(accuracies):.4f}")
            
            self.logger.info(f"\nDrop Statistics:")
            self.logger.info(f"  Min Drop:  {min(drops):.4f} ({min(drops)/std_accuracy*100:.2f}%)")
            self.logger.info(f"  Max Drop:  {max(drops):.4f} ({max(drops)/std_accuracy*100:.2f}%)")
            self.logger.info(f"  Mean Drop: {sum(drops)/len(drops):.4f}")
            
            excellent = len([d for d in drops if d <= 0.01])
            good = len([d for d in drops if 0.01 < d <= 0.05])
            medium = len([d for d in drops if 0.05 < d <= 0.10])
            poor = len([d for d in drops if d > 0.10])
            
            self.logger.info(f"\nMultiplier Categories:")
            self.logger.info(f"  Excellent (≤1% drop):   {excellent}")
            self.logger.info(f"  Good (1-5% drop):       {good}")
            self.logger.info(f"  Medium (5-10% drop):    {medium}")
            self.logger.info(f"  Poor (>10% drop):       {poor}")
            
            self.results['summary'] = {
                'standard_accuracy': float(std_accuracy),
                'total_multipliers': int(len(approx_results)),
                'best_accuracy': float(max(accuracies)),
                'worst_accuracy': float(min(accuracies)),
                'mean_accuracy': float(sum(accuracies)/len(accuracies)),
                'min_drop': float(min(drops)),
                'max_drop': float(max(drops)),
                'mean_drop': float(sum(drops)/len(drops)),
                'categories': {
                    'excellent': int(excellent),
                    'good': int(good),
                    'medium': int(medium),
                    'poor': int(poor)
                }
            }
    
    def log_training_curves(self, history, output_dir, model_name='best_model'):
        """Save training curves as images and JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Plot accuracy and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot
        ax1.plot(epochs, history.history['accuracy'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(epochs, history.history['loss'], 'b-', label='Training', linewidth=2)
        ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_training_curves.png'), dpi=150)
        plt.close()
        
        # Save history as JSON
        history_dict = {
            'epochs': len(epochs),
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        with open(os.path.join(output_dir, f'{model_name}_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"  ✓ Training curves saved to {output_dir}/")
    
    def save_json(self):
        self.results['end_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert all results to JSON serializable format
        json_safe_results = convert_to_json_serializable(self.results)
        
        try:
            with open(self.json_file, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
            self.logger.info(f"\n✓ Results saved to: {self.json_file}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")
            # Save as pickle as fallback
            import pickle
            pickle_file = self.json_file.replace('.json', '.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.results, f)
            self.logger.info(f"✓ Results saved as pickle: {pickle_file}")
    
    def close(self):
        self.save_json()
        self.logger.info("\n" + "="*70)
        self.logger.info("NAS Run Completed")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("="*70)