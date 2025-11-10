# Quick Start: RTAMT Integration

## Prerequisites

Install required packages:
```bash
pip install tensorflow>=2.10.0 rtamt numpy matplotlib
```

## Quick Test (3-5 minutes)

Test RTAMT integration with a quick experiment:

```bash
python test_rtamt_integration.py
```

**What it does**:
- Runs 3 NAS trials with RTAMT verification enabled
- Tests all three STL specifications (convergence, robustness, energy-accuracy)
- Shows fitness penalties for failed verification
- Generates Pareto frontier analysis
- Time: ~3-5 minutes

**Expected output**:
```
================================================================================
RTAMT Integration Test Results
================================================================================

Best Architecture:
  Standard accuracy: 0.8542
  Mean approx accuracy: 0.8123
  Mean accuracy drop: 0.0419
  Fitness score: 0.7234

Formal Verification (RTAMT):
  Training convergence: ✓ PASS
    Robustness value: 0.0524
  Multiplier robustness: 66.7% pass rate
    Mean robustness: 3.2145
  Pareto points: 2/3 satisfy constraints

  Pareto Analysis:
    ✓ mul8u_2P7.bin      Acc: 0.8354  Energy: 0.65x
    ✗ mul8u_3P8.bin      Acc: 0.7892  Energy: 0.58x
    ✓ mul8u_5NG.bin      Acc: 0.8123  Energy: 0.52x

  Overall verification: ✓ PASS

All Trials Summary:
  Verified architectures: 2/3 (66.7%)
  Fitness scores: ['0.7234', '0.6891', '0.7012']
  Mean fitness: 0.7046 ± 0.0141

RTAMT Integration Test Complete!
================================================================================
```

## Full MNIST Experiment (2-3 hours)

Run complete hardware-aware NAS with RTAMT verification:

```bash
python main_reference.py \
  --dataset mnist \
  --nas_trials 20 \
  --nas_use_multipliers \
  --nas_num_multipliers 10 \
  --use_rtamt \
  --rtamt_min_accuracy 0.8 \
  --rtamt_max_drop 10.0 \
  --epochs_per_trial 15 \
  --final_epochs 30 \
  --train_samples 10000 \
  --val_samples 2000 \
  --test_all_multipliers \
  --log_dir logs/mnist_hw_nas_rtamt
```

**Output files** (in `logs/mnist_hw_nas_rtamt/`):
- `nas_run_<timestamp>.log` - Full detailed log with verification results
- `nas_results_<timestamp>.json` - NAS results in JSON format
- `verification_report_<timestamp>.json` - RTAMT verification report
- `pareto_frontier_<timestamp>.png` - Pareto frontier visualization
- `accuracy_energy_scatter_<timestamp>.png` - Scatter plot
- `verification_summary_<timestamp>.png` - Verification summary visualization
- `standard_model_weights.h5` - Trained model weights

## Full CIFAR-10 Experiment (8-12 hours)

```bash
python main_reference.py \
  --dataset cifar10 \
  --nas_trials 20 \
  --nas_use_multipliers \
  --nas_num_multipliers 10 \
  --use_rtamt \
  --rtamt_min_accuracy 0.70 \
  --rtamt_max_drop 10.0 \
  --batch_size 128 \
  --epochs_per_trial 20 \
  --final_epochs 50 \
  --train_samples 20000 \
  --val_samples 5000 \
  --test_all_multipliers \
  --log_dir logs/cifar10_hw_nas_rtamt
```

## Understanding the Command-Line Flags

### RTAMT Flags
- `--use_rtamt` - **Enable RTAMT formal verification** (main flag)
- `--rtamt_min_accuracy 0.8` - Minimum accuracy threshold (e.g., 80%)
- `--rtamt_max_drop 10.0` - Maximum allowed accuracy drop percentage (e.g., 10%)

### Hardware-Aware NAS Flags
- `--nas_use_multipliers` - Enable hardware-aware NAS (evaluates with approximate multipliers)
- `--nas_num_multipliers 10` - Number of multipliers to use during NAS search
- `--test_all_multipliers` - Test ALL multipliers after finding best architecture

### Dataset Flags
- `--dataset mnist` - Dataset to use (mnist, cifar10, cifar100, fashion_mnist)
- `--train_samples 10000` - Number of training samples
- `--val_samples 2000` - Number of validation samples

### Training Flags
- `--nas_trials 20` - Number of NAS trials (architectures to try)
- `--epochs_per_trial 15` - Epochs for each NAS trial
- `--final_epochs 30` - Epochs for final model training
- `--batch_size 64` - Batch size (128 for CIFAR-10)

### Output
- `--log_dir logs/my_experiment` - Directory to save results

## What Gets Verified

### 1. Training Convergence (STL)
**Property**: Model must reach target accuracy within allowed epochs

**STL**: `eventually[0:max_epochs](val_acc >= min_accuracy)`

**Example**: With `--rtamt_min_accuracy 0.8` and `--epochs_per_trial 15`:
- Model must reach 80% validation accuracy within 15 epochs

### 2. Multiplier Robustness (STL)
**Property**: Accuracy drop with approximate multipliers must stay within bounds

**STL**: `always(accuracy_drop <= max_drop_percent)`

**Example**: With `--rtamt_max_drop 10.0`:
- Accuracy drop must not exceed 10% for any multiplier

### 3. Energy-Accuracy Tradeoff (STL)
**Property**: Low-energy multipliers must maintain acceptable accuracy

**STL**: `always((energy_ratio <= 1.5) implies (accuracy >= min_accuracy))`

**Example**: Multipliers using ≤1.5x energy must maintain ≥min_accuracy

## Viewing Results

After experiment completes:

```bash
# View log file
cat logs/mnist_hw_nas_rtamt/nas_run_*.log

# View verification report
cat logs/mnist_hw_nas_rtamt/verification_report_*.json

# View plots (copy to local machine if running on server)
scp user@server:~/nsa/logs/mnist_hw_nas_rtamt/*.png .
```

## Interpreting Verification Report

Sample `verification_report_*.json`:
```json
{
  "verification_results": {
    "training": {
      "robustness": 0.0524,
      "satisfied": true,
      "property": "Eventually reaches minimum accuracy"
    },
    "robustness": {
      "mean_robustness": 3.2145,
      "min_robustness": -1.234,
      "satisfaction_rate": 0.8,
      "robustness_values": [5.2, 3.1, -1.2, 4.5, 2.8]
    },
    "pareto": [
      {
        "multiplier": "mul8u_2P7.bin",
        "accuracy": 0.8854,
        "energy_ratio": 0.65,
        "robustness": 0.1854,
        "satisfied": true
      }
    ]
  },
  "overall_verdict": {
    "passed": true,
    "training_converged": true,
    "robustness_rate": 0.8,
    "has_good_pareto_point": true
  }
}
```

**Key metrics**:
- `robustness > 0`: Property satisfied (good!)
- `robustness < 0`: Property violated (failed)
- `satisfaction_rate`: Percentage of multipliers that passed (≥70% recommended)
- `passed`: Overall verdict (all properties satisfied)

## Tuning Thresholds

### Too Strict (all architectures fail)
**Symptoms**: `Verified architectures: 0/20 (0.0%)`

**Solution**: Make thresholds more lenient
```bash
--rtamt_min_accuracy 0.70 \  # Lower minimum accuracy
--rtamt_max_drop 15.0        # Allow larger drop
```

### Too Lenient (all architectures pass)
**Symptoms**: `Verified architectures: 20/20 (100.0%)`

**Solution**: Make thresholds stricter
```bash
--rtamt_min_accuracy 0.90 \  # Higher minimum accuracy
--rtamt_max_drop 5.0         # Allow smaller drop
```

### Recommended Starting Points
| Dataset | min_accuracy | max_drop |
|---------|-------------|----------|
| MNIST | 0.80 | 10.0 |
| Fashion-MNIST | 0.75 | 12.0 |
| CIFAR-10 | 0.70 | 10.0 |
| CIFAR-100 | 0.60 | 15.0 |

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'rtamt'`
**Solution**:
```bash
pip install rtamt
```

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution**:
```bash
pip install tensorflow>=2.10.0
```

### Issue: No verification report generated
**Reason**: RTAMT not enabled or NAS skipped

**Solution**: Ensure you have:
1. `--use_rtamt` flag
2. NOT using `--skip_nas` flag

### Issue: All architectures fail verification
**Reason**: Thresholds too strict for dataset

**Solution**: Lower thresholds:
```bash
--rtamt_min_accuracy 0.65 --rtamt_max_drop 15.0
```

## Summary of What Was Implemented

✅ **RTAMT formal verification** integrated into NAS
✅ **Three STL specifications** (convergence, robustness, energy-accuracy)
✅ **Pareto frontier generation** with visualization
✅ **Fitness-guided search** (20% penalty for failed verification)
✅ **Comprehensive reports** (JSON + 3 types of plots)
✅ **Full documentation** with examples

**To enable**: Just add `--use_rtamt` to your command!

## Quick Reference

### Minimal Command (Standard NAS + RTAMT)
```bash
python main_reference.py --use_rtamt --log_dir logs/test
```

### Recommended Command (Hardware-Aware NAS + RTAMT)
```bash
python main_reference.py \
  --nas_use_multipliers \
  --use_rtamt \
  --test_all_multipliers \
  --log_dir logs/hw_nas_rtamt
```

### Full Command (All Features)
```bash
python main_reference.py \
  --dataset mnist \
  --nas_trials 20 \
  --nas_use_multipliers \
  --nas_num_multipliers 10 \
  --use_rtamt \
  --rtamt_min_accuracy 0.8 \
  --rtamt_max_drop 10.0 \
  --epochs_per_trial 15 \
  --final_epochs 30 \
  --train_samples 10000 \
  --val_samples 2000 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --test_all_multipliers \
  --log_dir logs/full_experiment
```

For more details, see:
- **RTAMT_INTEGRATION_GUIDE.md** - Complete documentation
- **RTAMT_INTEGRATION_SUMMARY.md** - Implementation summary
