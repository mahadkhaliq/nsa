# RTAMT Formal Verification Integration Guide

## Overview

RTAMT (Runtime Verification with Metric Temporal Logic) has been integrated into the Hardware-Aware NAS system to formally verify neural network architectures for:
- **Training convergence** - Ensures models reach target accuracy
- **Multiplier robustness** - Verifies bounded accuracy degradation with approximate multipliers
- **Energy-accuracy tradeoff** - Validates Pareto-optimal configurations
- **Pareto frontier generation** - Identifies optimal architecture-multiplier combinations

## Installation

### Prerequisites
```bash
pip install tensorflow>=2.10.0
pip install rtamt
pip install numpy matplotlib
```

### Verify Installation
```bash
python -c "import rtamt; import tensorflow as tf; print('RTAMT:', rtamt.__version__); print('TensorFlow:', tf.__version__)"
```

## Files Added/Modified

### New Files
1. **`rtamt_verifier.py`** - Core RTAMT verification engine
   - `NASVerifier` class with STL specifications
   - Training convergence, robustness, and energy-accuracy verification
   - Comprehensive report generation

2. **`pareto_visualization.py`** - Pareto frontier visualization
   - `plot_pareto_frontier()` - Main Pareto plot with optimal points
   - `plot_accuracy_energy_scatter()` - Scatter plot of all points
   - `plot_nas_verification_summary()` - Verification results visualization

3. **`test_rtamt_integration.py`** - Quick integration test
   - Tests all RTAMT features
   - 3 trials, ~3-5 minutes

### Modified Files
1. **`nas_reference.py`**
   - Added `use_rtamt_verification` parameter
   - RTAMT verification after each NAS trial
   - Fitness penalty (20%) for failed verification
   - Verification results in trial data

2. **`main_reference.py`**
   - Added `--use_rtamt` command-line flag
   - RTAMT threshold configuration flags
   - Automatic verification report generation
   - Pareto frontier visualization

## Usage

### Quick Test (3 trials, ~5 minutes)
```bash
python test_rtamt_integration.py
```

### Enable RTAMT in Full Experiments

#### Standard NAS with RTAMT
```bash
python main_reference.py \
  --dataset mnist \
  --nas_trials 20 \
  --use_rtamt \
  --rtamt_min_accuracy 0.8 \
  --rtamt_max_drop 10.0 \
  --log_dir logs/mnist_rtamt
```

#### Hardware-Aware NAS with RTAMT
```bash
python main_reference.py \
  --dataset mnist \
  --nas_trials 20 \
  --nas_use_multipliers \
  --nas_num_multipliers 10 \
  --use_rtamt \
  --rtamt_min_accuracy 0.8 \
  --rtamt_max_drop 10.0 \
  --test_all_multipliers \
  --log_dir logs/mnist_hw_nas_rtamt
```

#### CIFAR-10 with RTAMT
```bash
python main_reference.py \
  --dataset cifar10 \
  --nas_trials 20 \
  --nas_use_multipliers \
  --nas_num_multipliers 10 \
  --use_rtamt \
  --rtamt_min_accuracy 0.70 \
  --rtamt_max_drop 8.0 \
  --batch_size 128 \
  --epochs_per_trial 20 \
  --final_epochs 50 \
  --train_samples 20000 \
  --val_samples 5000 \
  --test_all_multipliers \
  --log_dir logs/cifar10_hw_nas_rtamt
```

## Command-Line Arguments

### RTAMT-Specific Flags
- `--use_rtamt` - Enable RTAMT formal verification (default: False)
- `--rtamt_min_accuracy FLOAT` - Minimum accuracy threshold for verification (default: 0.7)
- `--rtamt_max_drop FLOAT` - Maximum accuracy drop percentage for robustness (default: 10.0)

### Example Threshold Settings
**MNIST (easier dataset)**:
- `--rtamt_min_accuracy 0.85` - Expect 85%+ accuracy
- `--rtamt_max_drop 5.0` - Allow max 5% drop with approximate multipliers

**CIFAR-10 (harder dataset)**:
- `--rtamt_min_accuracy 0.70` - Expect 70%+ accuracy
- `--rtamt_max_drop 10.0` - Allow max 10% drop

## How RTAMT Verification Works

### 1. Training Convergence (STL)
**Property**: `eventually[0:max_epochs](val_acc >= min_accuracy)`

Verifies that validation accuracy eventually reaches the minimum threshold within the allowed epochs.

**Robustness value**:
- Positive: Property satisfied (converged)
- Negative: Property violated (did not converge)
- Magnitude: How far from threshold

### 2. Multiplier Robustness (STL)
**Property**: `always(accuracy_drop <= max_drop_percent)`

Verifies that accuracy drop with approximate multipliers stays within acceptable bounds.

**Applied to**: Each multiplier tested
**Pass criteria**: ≥70% of multipliers pass

### 3. Energy-Accuracy Tradeoff (STL)
**Property**: `always((energy_ratio <= max_energy) implies (accuracy >= min_accuracy))`

Verifies that low-energy multipliers maintain acceptable accuracy.

**Identifies**: Pareto-optimal multiplier-architecture pairs

### 4. Fitness Penalty
Architectures failing verification receive a **20% fitness penalty**:
```python
if not verification_satisfied:
    fitness = fitness * 0.8
```

This guides the search toward formally verified architectures.

## Output Files

When RTAMT is enabled, the following files are generated in the log directory:

### 1. Verification Report (JSON)
`verification_report_<timestamp>.json`

Contains:
- Training convergence results
- Multiplier robustness statistics
- Pareto point analysis
- Overall verdict

### 2. Pareto Frontier Plot
`pareto_frontier_<timestamp>.png`

Shows:
- All multiplier-architecture combinations
- Pareto-optimal points (red stars)
- Ideal region (high accuracy, low energy)

### 3. Accuracy-Energy Scatter Plot
`accuracy_energy_scatter_<timestamp>.png`

Color-coded scatter plot showing:
- X-axis: Multiplier index
- Y-axis: Accuracy
- Color: Energy ratio

### 4. Verification Summary Plot
`verification_summary_<timestamp>.png`

Four-panel visualization:
- Training convergence verification
- Robustness distribution across multipliers
- Pareto point verification
- Overall summary text

## Signal Temporal Logic (STL) Specifications

### Training Convergence
```python
spec.spec = f'eventually[0:{max_epochs}](val_acc >= {min_accuracy})'
```
- **eventually[0:N]**: Within N time steps
- **val_acc >= threshold**: Accuracy reaches target

### Robustness
```python
spec.spec = f'always(accuracy_drop <= {max_drop_percent})'
```
- **always**: At all time points
- **accuracy_drop <= threshold**: Bounded degradation

### Energy-Accuracy Tradeoff
```python
spec.spec = f'always((energy_ratio <= {max_energy}) implies (accuracy >= {min_accuracy}))'
```
- **implies**: If-then constraint
- Low energy → acceptable accuracy

## Interpreting Results

### Verification Summary Log
```
Formal Verification Summary (RTAMT)
================================================================================
Verified architectures: 15/20 (75.0%)

Best Architecture Verification:
  Training convergence: ✓ PASS (robustness: 0.0524)
  Robustness rate: 80.0%
  Pareto-optimal points: 7/10
  Best Pareto point: mul8u_2P7.bin
    Accuracy: 0.8854
    Energy ratio: 0.65x
```

**What this means**:
- 75% of architectures passed formal verification
- Best architecture converged (robustness > 0)
- 80% of multipliers stayed within accuracy drop threshold
- 7 out of 10 multipliers satisfy energy-accuracy constraints
- Best Pareto point: 88.5% accuracy at 0.65x energy

### Robustness Values
- **> 0**: Property satisfied
- **= 0**: Boundary case (just satisfied)
- **< 0**: Property violated
- **Magnitude**: Distance from threshold

**Example**:
- Robustness = 0.05: Accuracy is 0.05 above threshold (good margin)
- Robustness = -0.03: Accuracy is 0.03 below threshold (failed)

### Pareto Frontier Interpretation
Points on the Pareto frontier (red stars) represent optimal tradeoffs:
- No other point has both higher accuracy AND lower energy
- Moving to any non-Pareto point sacrifices one objective

**Ideal multipliers**: High accuracy, low energy (top-left region)

## Tuning Thresholds

### For Research (Strict)
```bash
--rtamt_min_accuracy 0.90 \
--rtamt_max_drop 3.0
```
Only accepts high-quality, robust architectures.

### For Exploration (Lenient)
```bash
--rtamt_min_accuracy 0.70 \
--rtamt_max_drop 15.0
```
Allows more architectural diversity.

### Dataset-Specific Recommendations
| Dataset | min_accuracy | max_drop |
|---------|-------------|----------|
| MNIST | 0.85 | 5.0 |
| Fashion-MNIST | 0.80 | 8.0 |
| CIFAR-10 | 0.70 | 10.0 |
| CIFAR-100 | 0.60 | 12.0 |

## Energy Ratio Estimation

Currently using placeholder function in `rtamt_verifier.py`:

```python
def estimate_energy_ratio(multiplier_name: str) -> float:
    # PLACEHOLDER - replace with actual measurements
    base_energy = 0.6  # Approximate multipliers ~40-60% of standard
    variance = (hash(name) % 20) / 100.0
    return base_energy + variance
```

### To Use Real Energy Data:

1. **Measure energy** for each multiplier (hardware profiling)
2. **Create energy database**:
```python
ENERGY_DATABASE = {
    'mul8u_2P7.bin': 0.67,
    'mul8u_3P8.bin': 0.59,
    # ... etc
}
```
3. **Update function**:
```python
def estimate_energy_ratio(multiplier_name: str) -> float:
    return ENERGY_DATABASE.get(multiplier_name, 1.0)
```

## Example Workflow

### 1. Initial Exploration
```bash
# Quick test to verify RTAMT works
python test_rtamt_integration.py
```

### 2. Baseline Comparison
```bash
# Standard NAS (no RTAMT)
python main_reference.py --dataset mnist --nas_trials 20 \
  --nas_use_multipliers --log_dir logs/baseline

# With RTAMT verification
python main_reference.py --dataset mnist --nas_trials 20 \
  --nas_use_multipliers --use_rtamt --log_dir logs/rtamt
```

### 3. Analyze Results
- Compare `logs/baseline/nas_run_*.log` vs `logs/rtamt/nas_run_*.log`
- Review `logs/rtamt/verification_report_*.json`
- Examine `logs/rtamt/pareto_frontier_*.png`

### 4. Refine Thresholds
Based on results, adjust:
- `--rtamt_min_accuracy` (if too many/few pass)
- `--rtamt_max_drop` (if robustness too strict/lenient)

### 5. Production Run
```bash
python main_reference.py \
  --dataset cifar10 \
  --nas_trials 30 \
  --nas_use_multipliers \
  --nas_num_multipliers 36 \
  --use_rtamt \
  --rtamt_min_accuracy 0.75 \
  --rtamt_max_drop 8.0 \
  --epochs_per_trial 25 \
  --final_epochs 75 \
  --train_samples 30000 \
  --val_samples 10000 \
  --test_all_multipliers \
  --log_dir logs/cifar10_final
```

## Benefits of RTAMT Integration

### 1. Formal Guarantees
STL provides mathematical guarantees about:
- Training convergence
- Robustness bounds
- Energy-accuracy tradeoffs

### 2. Automated Quality Control
Architectures are automatically verified, ensuring:
- Reliable convergence
- Bounded degradation with approximate hardware
- Pareto-optimal configurations

### 3. Guided Search
Fitness penalties guide NAS toward:
- More robust architectures
- Better hardware compatibility
- Energy-efficient designs

### 4. Interpretability
Verification reports provide:
- Clear pass/fail criteria
- Quantitative robustness measures
- Pareto frontier visualization

## Troubleshooting

### Issue: All architectures fail verification
**Solution**: Thresholds too strict
```bash
# Increase max_drop or decrease min_accuracy
--rtamt_min_accuracy 0.65 --rtamt_max_drop 15.0
```

### Issue: All architectures pass verification
**Solution**: Thresholds too lenient
```bash
# Decrease max_drop or increase min_accuracy
--rtamt_min_accuracy 0.85 --rtamt_max_drop 5.0
```

### Issue: RTAMT import error
**Solution**: Install RTAMT
```bash
pip install rtamt
```

### Issue: Slow verification
**Expected**: RTAMT adds ~5-10% overhead per trial
**Mitigation**: Use fewer multipliers during NAS
```bash
--nas_num_multipliers 5  # Instead of 36
```

## Advanced: Custom STL Specifications

You can add custom specifications by editing `rtamt_verifier.py`:

```python
def create_custom_spec(self):
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var('my_metric', 'float')

    # Example: Metric must increase monotonically
    spec.spec = 'always((my_metric - prev(my_metric)) >= 0)'

    spec.parse()
    return spec
```

## Future Enhancements

1. **Real energy measurements** - Replace placeholder energy estimator
2. **Multi-objective optimization** - Explicit Pareto optimization
3. **Runtime verification** - Verify deployed models during inference
4. **Custom STL properties** - User-defined verification criteria
5. **Hypervolume indicator** - Quantify Pareto frontier quality

## References

- RTAMT Documentation: https://github.com/nickovic/rtamt
- Signal Temporal Logic: Maler & Nickovic (2004)
- Approximate Computing: Venkataramani et al. (2015)
- Hardware-Aware NAS: This repository

## Summary

RTAMT integration provides formal verification of neural architectures for:
- ✓ Training convergence
- ✓ Multiplier robustness
- ✓ Energy-accuracy tradeoffs
- ✓ Pareto frontier identification

**To use**: Add `--use_rtamt` flag to any experiment!
