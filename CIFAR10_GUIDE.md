# CIFAR-10 Hardware-Aware NAS Guide

## Overview

CIFAR-10 is more challenging than MNIST:
- **Color images**: 32x32x3 (RGB) vs 28x28x1 (grayscale)
- **More complex**: Natural images vs handwritten digits
- **Lower accuracy**: Expect 70-85% vs 95-99% on MNIST
- **Longer training**: Needs more epochs to converge

## Key Differences from MNIST

### 1. Architecture Configuration
CIFAR-10 uses larger default architecture:
```python
{
    'conv1_filters': 32,    # vs 16 for MNIST
    'conv2_filters': 64,    # vs 32 for MNIST
    'conv3_filters': 128,   # vs 64 for MNIST
    'dense1_size': 128,     # vs 64 for MNIST
    'dropout_rate': 0.3
}
```

### 2. Training Hyperparameters
- **Batch size**: 128 (vs 64 for MNIST)
- **Epochs per trial**: 20+ (vs 15 for MNIST)
- **Final epochs**: 50+ (vs 30 for MNIST)
- **Learning rate**: 0.001 (same as MNIST)

### 3. Data Volume
- **Training samples**: 50,000 total (use 20,000-30,000 for NAS)
- **Validation samples**: 10,000 total (use 5,000 for NAS)
- **Test samples**: 10,000

## Running CIFAR-10 Experiments

### Quick Test (3 trials, 5 multipliers) - ~30-45 minutes
```bash
python test_cifar10_quick.py
```

### Full Hardware-Aware NAS - ~12-16 hours
```bash
python main_reference.py \
  --dataset cifar10 \
  --nas_trials 20 \
  --nas_method evolutionary \
  --epochs_per_trial 20 \
  --final_epochs 50 \
  --train_samples 20000 \
  --val_samples 5000 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --nas_use_multipliers \
  --nas_num_multipliers 10 \
  --test_all_multipliers \
  --log_dir logs/cifar10_hw_nas
```

### Standard NAS (baseline) - ~6-8 hours
```bash
python main_reference.py \
  --dataset cifar10 \
  --nas_trials 20 \
  --nas_method evolutionary \
  --epochs_per_trial 20 \
  --final_epochs 50 \
  --train_samples 20000 \
  --val_samples 5000 \
  --batch_size 128 \
  --test_all_multipliers \
  --log_dir logs/cifar10_standard_nas
```

### Test All Multipliers (default arch) - ~1 hour
```bash
python main_reference.py \
  --dataset cifar10 \
  --skip_nas \
  --final_epochs 50 \
  --train_samples 20000 \
  --val_samples 5000 \
  --batch_size 128 \
  --test_all_multipliers \
  --log_dir logs/cifar10_default_all_muls
```

## Expected Performance

### Baseline (Standard Architecture, No Approximate)
- **Accuracy**: 75-85% validation accuracy
- **Training time**: ~30-50 minutes for 50 epochs

### With Approximate Multipliers
- **Good multipliers**: 1-5% accuracy drop
- **Medium multipliers**: 5-15% accuracy drop
- **Poor multipliers**: 15%+ accuracy drop or catastrophic failure

### Hardware-Aware NAS Benefits
- Should find architectures with **better robustness** to approximate multipliers
- Target: <5% average accuracy drop across good multipliers
- May sacrifice 1-2% standard accuracy for better robustness

## Recommended Experiment Sequence

### Phase 1: Quick Validation (1 hour)
```bash
# 1. Quick test (3 trials)
python test_cifar10_quick.py

# 2. Test all multipliers with default architecture
python main_reference.py --dataset cifar10 --skip_nas --batch_size 128 \
  --final_epochs 50 --train_samples 20000 --val_samples 5000 \
  --test_all_multipliers --log_dir logs/cifar10_test_muls
```

### Phase 2: Standard NAS Baseline (8 hours)
```bash
python main_reference.py --dataset cifar10 --nas_trials 20 \
  --epochs_per_trial 20 --final_epochs 50 --batch_size 128 \
  --train_samples 20000 --val_samples 5000 --test_all_multipliers \
  --log_dir logs/cifar10_standard_nas
```

### Phase 3: Hardware-Aware NAS (16 hours)
```bash
python main_reference.py --dataset cifar10 --nas_trials 20 \
  --nas_use_multipliers --nas_num_multipliers 10 \
  --epochs_per_trial 20 --final_epochs 50 --batch_size 128 \
  --train_samples 20000 --val_samples 5000 --test_all_multipliers \
  --log_dir logs/cifar10_hw_nas
```

### Phase 4: Extended Hardware-Aware NAS (24+ hours)
```bash
# With more trials and all multipliers
python main_reference.py --dataset cifar10 --nas_trials 30 \
  --nas_use_multipliers --nas_num_multipliers 36 \
  --epochs_per_trial 25 --final_epochs 75 --batch_size 128 \
  --train_samples 30000 --val_samples 10000 --test_all_multipliers \
  --log_dir logs/cifar10_hw_nas_extended
```

## Tips for CIFAR-10

### 1. GPU Memory
- CIFAR-10 uses more memory due to 3 color channels
- Use batch size 128 (or 64 if memory limited)
- Enable GPU memory growth

### 2. Training Convergence
- CIFAR-10 needs more epochs than MNIST
- Monitor validation accuracy - should improve up to ~40-50 epochs
- If stuck at ~40%, increase epochs or check learning rate

### 3. Data Augmentation (Future Enhancement)
For production, consider adding:
- Random horizontal flip
- Random crop with padding
- Cutout/MixUp
(Not currently implemented in this version)

### 4. Architecture Search Space
Current search space works well for CIFAR-10:
- Filter ranges: 16-256 (appropriate for color images)
- Kernel sizes: 3x3 and 5x5
- Dense layer: 64-256 units
- Dropout: 0.3-0.5

## Comparison: MNIST vs CIFAR-10

| Aspect | MNIST | CIFAR-10 |
|--------|-------|----------|
| Image size | 28x28x1 | 32x32x3 |
| Complexity | Simple (digits) | Complex (objects) |
| Typical accuracy | 98-99% | 75-85% |
| Conv filters | 16→32→64 | 32→64→128 |
| Epochs needed | 15-30 | 30-75 |
| Batch size | 64 | 128 |
| NAS time (20 trials) | ~2-4 hours | ~8-16 hours |

## Interpreting Results

### Good Results for CIFAR-10
- **Standard accuracy**: 75%+ is good, 80%+ is very good
- **Multiplier robustness**: <5% average drop for good multipliers
- **Consistent performance**: Similar accuracy across multiple runs

### Red Flags
- **Accuracy < 60%**: Undertrained or architecture too small
- **Large variation**: Check if model is converging
- **All multipliers fail**: Check FakeApproxConv2D compatibility

## Next Steps After CIFAR-10

Once you have good CIFAR-10 results:
1. **CIFAR-100**: 100 classes instead of 10
2. **Tiny ImageNet**: 200 classes, 64x64 images
3. **Full ImageNet**: Production-scale dataset

## Troubleshooting

### Low Accuracy (~40-50%)
- Increase epochs (try 75-100)
- Increase training samples
- Check if model is too small

### Out of Memory
- Reduce batch size to 64 or 32
- Reduce filter sizes in search space
- Use smaller architecture

### Very Slow Training
- Reduce number of samples
- Reduce epochs_per_trial (but keep final_epochs high)
- Use fewer NAS trials
