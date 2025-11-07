import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def save_predictions(model, x_val, y_val, output_dir, model_name='standard', 
                    class_names=None, top_k=20):
    """
    Save predictions, misclassifications, and confusion matrix
    
    Args:
        model: Trained model
        x_val, y_val: Validation data
        output_dir: Directory to save results
        model_name: Name for this model (e.g., 'standard', 'mul8u_2P7')
        class_names: List of class names
        top_k: Number of worst misclassifications to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    predictions = model.predict(x_val, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    # Find correct and incorrect predictions
    correct_mask = (predicted_classes == y_val)
    incorrect_mask = ~correct_mask
    
    # Calculate metrics
    accuracy = np.mean(correct_mask)
    
    # Save prediction details
    results = {
        'model_name': model_name,
        'total_samples': len(y_val),
        'correct_predictions': int(np.sum(correct_mask)),
        'incorrect_predictions': int(np.sum(incorrect_mask)),
        'accuracy': float(accuracy),
        'predictions': []
    }
    
    for i in range(len(y_val)):
        results['predictions'].append({
            'index': int(i),
            'true_label': int(y_val[i]),
            'predicted_label': int(predicted_classes[i]),
            'confidence': float(confidence_scores[i]),
            'correct': bool(correct_mask[i])
        })
    
    # Save JSON
    with open(os.path.join(output_dir, f'{model_name}_predictions.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save confusion matrix
    cm = confusion_matrix(y_val, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Save classification report
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_val)))]
    
    report = classification_report(y_val, predicted_classes, 
                                   target_names=class_names, 
                                   output_dict=True)
    with open(os.path.join(output_dir, f'{model_name}_classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save worst misclassifications
    if np.sum(incorrect_mask) > 0:
        save_misclassified_images(x_val, y_val, predicted_classes, confidence_scores,
                                 incorrect_mask, output_dir, model_name, 
                                 class_names, top_k)
    
    return results

def save_misclassified_images(x_val, y_val, predicted_classes, confidence_scores,
                              incorrect_mask, output_dir, model_name, class_names, top_k):
    """Save images of worst misclassifications"""
    
    misclassified_dir = os.path.join(output_dir, f'{model_name}_misclassified')
    os.makedirs(misclassified_dir, exist_ok=True)
    
    # Get indices of misclassified samples, sorted by confidence (most confident errors first)
    incorrect_indices = np.where(incorrect_mask)[0]
    incorrect_confidences = confidence_scores[incorrect_indices]
    
    # Sort by confidence (descending) - these are the most confidently wrong predictions
    sorted_indices = incorrect_indices[np.argsort(-incorrect_confidences)]
    
    # Save top K worst misclassifications
    num_to_save = min(top_k, len(sorted_indices))
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.ravel()
    
    for i in range(min(20, num_to_save)):
        idx = sorted_indices[i]
        img = x_val[idx]
        true_label = y_val[idx]
        pred_label = predicted_classes[idx]
        confidence = confidence_scores[idx]
        
        ax = axes[i]
        
        # Handle grayscale vs color
        if img.shape[-1] == 1:
            ax.imshow(img[:, :, 0], cmap='gray')
        else:
            ax.imshow(img)
        
        true_name = class_names[true_label] if class_names else str(true_label)
        pred_name = class_names[pred_label] if class_names else str(pred_label)
        
        ax.set_title(f'True: {true_name}\nPred: {pred_name}\nConf: {confidence:.2f}', 
                    fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(min(20, num_to_save), 20):
        axes[i].axis('off')
    
    plt.suptitle(f'Top 20 Misclassifications - {model_name}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(misclassified_dir, 'top_misclassifications.png'), dpi=150)
    plt.close()

def plot_confidence_distribution(results_list, output_dir):
    """Plot confidence distribution for multiple models"""
    plt.figure(figsize=(12, 6))
    
    for result in results_list:
        model_name = result['model_name']
        confidences = [p['confidence'] for p in result['predictions']]
        correct = [p['correct'] for p in result['predictions']]
        
        correct_conf = [c for c, cor in zip(confidences, correct) if cor]
        incorrect_conf = [c for c, cor in zip(confidences, correct) if not cor]
        
        plt.hist(correct_conf, bins=50, alpha=0.5, label=f'{model_name} (Correct)')
        plt.hist(incorrect_conf, bins=50, alpha=0.5, label=f'{model_name} (Incorrect)')
    
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=150)
    plt.close()