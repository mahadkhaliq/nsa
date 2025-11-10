# Essential Files for Hardware-Aware NAS

## Core Implementation Files (KEEP THESE)

### 1. Main Entry Points
- **`main_reference.py`** - PRIMARY script for running full NAS experiments
- **`test_hardware_aware_nas.py`** - Quick test comparing standard vs hardware-aware NAS

### 2. Core Modules
- **`nas_reference.py`** - Hardware-aware NAS implementation with multi-multiplier evaluation
- **`model_builder.py`** - Two model builders (training vs evaluation with approximate multipliers)
- **`dataloader.py`** - Dataset loading (MNIST, CIFAR10, CIFAR100, Fashion-MNIST)
- **`training.py`** - Model evaluation utilities
- **`logger.py`** - Logging and results tracking

### 3. Configuration
- **`multipliers/`** - Directory containing all .bin multiplier files (36 files)

---

## Legacy/Test Files (CAN BE ARCHIVED OR DELETED)

### Old Test Files (for debugging - not needed for experiments)
- `test_simple_mnist.py` - Basic MNIST test
- `test_multiplier.py` - Single multiplier testing
- `test_two_stage_pattern.py` - Pattern validation
- `test_reference_architecture.py` - Architecture pattern test
- `test_nas_quick.py` - Quick NAS test (3 trials)
- `test_all_multipliers_simple.py` - Simple multiplier sweep
- `test_modelmul8u_2P7.py` - Specific multiplier test
- `test_architecture_match.py` - Architecture validation
- `test_architecture_complexity.py` - Complexity analysis
- `test_conv1x1_issue.py` - Debugging script
- `test_workflow.py` - Workflow validation

### Debug/Diagnostic Files
- `debug_model_match.py` - Model matching debug
- `diagnose_layers.py` - Layer diagnostics
- `multiplier_verfication_code.py` - Multiplier verification

### Old/Deprecated Implementations
- **`main.py`** - OLD main script (replaced by main_reference.py)
- **`nas.py`** - OLD NAS implementation (replaced by nas_reference.py)
- `version_0.py` - Very old version
- `architecture.py` - Old architecture definitions
- `operations.py` - Old operations
- `validation_utils.py` - Old validation utilities

---

## Running Full Experiments

### 1. Quick Test (3 trials, 5 multipliers) - ~10 minutes
```bash
python test_hardware_aware_nas.py
```
This compares standard NAS vs hardware-aware NAS with 5 multipliers.

### 2. Standard NAS (no multipliers) - ~2 hours
```bash
python main_reference.py \
  --dataset mnist \
  --nas_trials 20 \
  --nas_method evolutionary \
  --epochs_per_trial 15 \
  --final_epochs 30 \
  --train_samples 10000 \
  --val_samples 2000 \
  --log_dir logs/standard_nas
```

### 3. Hardware-Aware NAS (with ALL 36 multipliers) - ~8-12 hours
```bash
python main_reference.py \
  --dataset mnist \
  --nas_trials 20 \
  --nas_method evolutionary \
  --epochs_per_trial 15 \
  --final_epochs 30 \
  --train_samples 10000 \
  --val_samples 2000 \
  --nas_use_multipliers \
  --nas_num_multipliers 36 \
  --test_all_multipliers \
  --log_dir logs/hardware_aware_nas_all
```

### 4. Hardware-Aware NAS (with 10 best multipliers) - ~3-4 hours
First, identify the 10 best multipliers from the previous run, then:
```bash
python main_reference.py \
  --dataset mnist \
  --nas_trials 20 \
  --nas_method evolutionary \
  --epochs_per_trial 15 \
  --final_epochs 30 \
  --train_samples 10000 \
  --val_samples 2000 \
  --nas_use_multipliers \
  --nas_num_multipliers 10 \
  --test_all_multipliers \
  --log_dir logs/hardware_aware_nas_10
```

### 5. Test with All Multipliers (using default architecture) - ~30 minutes
```bash
python main_reference.py \
  --dataset mnist \
  --skip_nas \
  --test_all_multipliers \
  --train_samples 10000 \
  --val_samples 2000 \
  --final_epochs 30 \
  --log_dir logs/default_arch_all_multipliers
```

---

## For CIFAR-10 (After MNIST Results)

See **CIFAR10_GUIDE.md** for detailed CIFAR-10 instructions.

### Quick CIFAR-10 Test (3 trials, ~30-45 min)
```bash
python test_cifar10_quick.py
```

### CIFAR-10 Hardware-Aware NAS (16+ hours)
```bash
python main_reference.py \
  --dataset cifar10 \
  --nas_trials 20 \
  --nas_method evolutionary \
  --epochs_per_trial 20 \
  --final_epochs 50 \
  --train_samples 20000 \
  --val_samples 5000 \
  --nas_use_multipliers \
  --nas_num_multipliers 10 \
  --test_all_multipliers \
  --batch_size 128 \
  --learning_rate 0.001 \
  --log_dir logs/cifar10_hw_nas
```

### CIFAR-10 Standard NAS Baseline (8 hours)
```bash
python main_reference.py \
  --dataset cifar10 \
  --nas_trials 20 \
  --epochs_per_trial 20 \
  --final_epochs 50 \
  --train_samples 20000 \
  --val_samples 5000 \
  --batch_size 128 \
  --test_all_multipliers \
  --log_dir logs/cifar10_standard_nas
```

---

## Recommended Experiment Sequence

### Phase 1: Quick Validation (30 minutes)
```bash
# 1. Test hardware-aware NAS works
python test_hardware_aware_nas.py

# 2. Test all multipliers with default architecture
python main_reference.py --skip_nas --test_all_multipliers --log_dir logs/test_all_muls
```

### Phase 2: Full MNIST Experiments (12-16 hours total)
```bash
# 1. Standard NAS (baseline)
python main_reference.py \
  --nas_trials 20 \
  --train_samples 10000 \
  --val_samples 2000 \
  --log_dir logs/mnist_standard_nas

# 2. Hardware-Aware NAS with ALL multipliers
python main_reference.py \
  --nas_trials 20 \
  --nas_use_multipliers \
  --nas_num_multipliers 36 \
  --test_all_multipliers \
  --train_samples 10000 \
  --val_samples 2000 \
  --log_dir logs/mnist_hw_nas_all
```

### Phase 3: CIFAR-10 (after analyzing MNIST results)
Run CIFAR-10 experiments with the best configuration found from MNIST.

---

## Key Command-Line Arguments

### Dataset Options
- `--dataset mnist|cifar10|cifar100|fashion_mnist` (default: mnist)
- `--train_samples N` (default: 5000)
- `--val_samples N` (default: 1000)

### NAS Options
- `--nas_trials N` - Number of architecture trials (default: 15)
- `--nas_method random|evolutionary` (default: evolutionary)
- `--skip_nas` - Skip NAS and use default architecture
- `--nas_use_multipliers` - Enable hardware-aware NAS
- `--nas_num_multipliers N` - Number of multipliers to use in NAS (default: 10)

### Training Options
- `--epochs_per_trial N` - Epochs per NAS trial (default: 15)
- `--final_epochs N` - Epochs for final model (default: 30)
- `--batch_size N` (default: 64)
- `--learning_rate FLOAT` (default: 0.001)

### Multiplier Testing
- `--skip_multipliers` - Skip multiplier testing
- `--test_all_multipliers` - Test ALL multipliers (not just first 10)
- `--multiplier_dir PATH` (default: ./multipliers)

### Output
- `--log_dir PATH` (default: logs)

---

## Output Files

After running, you'll find in the log directory:
- `nas_run_TIMESTAMP.log` - Full detailed log
- `nas_results_TIMESTAMP.json` - Results in JSON format
- `standard_model_weights.h5` - Trained model weights

---

## Cleanup Recommendation

### Keep These:
```
main_reference.py
test_hardware_aware_nas.py
nas_reference.py
model_builder.py
dataloader.py
training.py
logger.py
multipliers/
logs/
```

### Archive/Delete These:
```
main.py (OLD)
nas.py (OLD)
version_0.py
test_*.py (except test_hardware_aware_nas.py)
debug_*.py
diagnose_*.py
architecture.py
operations.py
validation_utils.py
multiplier_verfication_code.py
```

You can create an `archive/` directory and move old files there:
```bash
mkdir archive
mv main.py nas.py version_0.py archive/
mv test_simple_mnist.py test_multiplier.py test_two_stage_pattern.py archive/
mv test_reference_architecture.py test_nas_quick.py archive/
mv test_all_multipliers_simple.py test_modelmul8u_2P7.py archive/
mv test_architecture_match.py test_architecture_complexity.py archive/
mv test_conv1x1_issue.py test_workflow.py archive/
mv debug_model_match.py diagnose_layers.py archive/
mv multiplier_verfication_code.py archive/
mv architecture.py operations.py validation_utils.py archive/
```
