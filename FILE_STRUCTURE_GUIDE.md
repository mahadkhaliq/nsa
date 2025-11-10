# File Structure Guide - Hardware-Aware NAS with RTAMT

## 🔴 ESSENTIAL FILES (Core System)

These are the **most important** files you need to run experiments:

### Main Entry Points
1. **`main_reference.py`** ⭐ **PRIMARY SCRIPT**
   - Main entry point for all experiments
   - Orchestrates: NAS → Training → Multiplier Testing → RTAMT Verification
   - Command-line interface for all configurations
   - **Use this to run experiments**

2. **`test_hardware_aware_nas.py`**
   - Quick test comparing standard vs hardware-aware NAS
   - 3 trials, ~10 minutes
   - Good for validation

3. **`test_rtamt_integration.py`** ⭐ **NEW - Test RTAMT**
   - Quick test of RTAMT formal verification
   - 3 trials with verification, ~3-5 minutes
   - Validates RTAMT integration

### Core NAS Engine
4. **`nas_reference.py`** ⭐ **CORE NAS**
   - Hardware-aware NAS implementation
   - Evaluates architectures with ALL multipliers
   - Evolutionary search with mutation
   - **NOW INCLUDES**: RTAMT formal verification

### Model Building
5. **`model_builder.py`** ⭐ **ARCHITECTURE**
   - Two model builders:
     - `build_model_for_training()` - Standard Conv2D for training
     - `build_model_for_evaluation()` - FakeApproxConv2D for testing
   - Multi-conv blocks (2 convs per block)
   - Weight transfer support

### Support Modules
6. **`dataloader.py`**
   - Dataset loading: MNIST, CIFAR-10, CIFAR-100, Fashion-MNIST
   - Automatic normalization and preprocessing

7. **`training.py`**
   - Model evaluation utilities
   - Accuracy calculation

8. **`logger.py`**
   - Experiment logging
   - Results tracking
   - Timestamped log files

### Custom Layers
9. **`layers.py`**
   - `FakeApproxConv2D` - Custom layer simulating 8-bit approximate multiplication
   - Loads .bin multiplier lookup tables

### 🆕 RTAMT Formal Verification (NEW)
10. **`rtamt_verifier.py`** ⭐ **NEW - FORMAL VERIFICATION**
    - `NASVerifier` class
    - Three STL specifications:
      - Training convergence
      - Multiplier robustness
      - Energy-accuracy tradeoff
    - Comprehensive verification reports

11. **`pareto_visualization.py`** ⭐ **NEW - PARETO FRONTIER**
    - Pareto frontier identification and visualization
    - Three plot types:
      - Pareto frontier plot
      - Scatter plot
      - Verification summary

### Documentation
12. **`RTAMT_INTEGRATION_GUIDE.md`** ⭐ **NEW - COMPLETE GUIDE**
    - Full RTAMT documentation
    - Usage examples
    - Threshold tuning
    - Troubleshooting

13. **`QUICK_START_RTAMT.md`** ⭐ **NEW - QUICK REFERENCE**
    - Quick start commands
    - Common use cases
    - Interpreting results

14. **`ESSENTIAL_FILES.md`**
    - Comprehensive experiment guide
    - Running different experiments
    - Command-line arguments

15. **`CIFAR10_GUIDE.md`**
    - CIFAR-10 specific guide
    - Hyperparameter recommendations
    - Expected performance

---

## 🟡 SUPPORTING FILES (Keep These)

### Configuration
- **`multipliers/`** - Directory with 36 .bin multiplier files
- **`logs/`** - Experiment results and logs

---

## 🟢 LEGACY/TEST FILES (Can Archive)

These files were used during development but are not needed for running experiments:

### Old Main Scripts (Replaced by main_reference.py)
- `main.py` - OLD main script ❌
- `nas.py` - OLD NAS implementation ❌
- `version_0.py` - Very old version ❌

### Old Test Scripts (Development/Debugging)
- `test_simple_mnist.py`
- `test_multiplier.py`
- `test_two_stage_pattern.py`
- `test_reference_architecture.py`
- `test_nas_quick.py`
- `test_all_multipliers_simple.py`
- `test_modelmul8u_2P7.py`
- `test_architecture_match.py`
- `test_architecture_complexity.py`
- `test_conv1x1_issue.py`
- `test_workflow.py`

### Debug/Diagnostic Files
- `debug_model_match.py`
- `diagnose_layers.py`
- `multiplier_verfication_code.py`

### Old Architecture Files
- `architecture.py` - OLD ❌
- `operations.py` - OLD ❌
- `validation_utils.py` - OLD ❌

---

## 📁 Recommended File Organization

### Keep These Active
```
nsa/
├── main_reference.py                    ⭐ Main script
├── test_hardware_aware_nas.py           ⭐ Quick test
├── test_rtamt_integration.py            ⭐ RTAMT test (NEW)
├── nas_reference.py                     ⭐ Core NAS
├── model_builder.py                     ⭐ Architecture
├── dataloader.py                        ⭐ Data loading
├── training.py                          ⭐ Evaluation
├── logger.py                            ⭐ Logging
├── layers.py                            ⭐ Custom layers
├── rtamt_verifier.py                    ⭐ Verification (NEW)
├── pareto_visualization.py              ⭐ Pareto plots (NEW)
├── RTAMT_INTEGRATION_GUIDE.md           📖 Complete guide (NEW)
├── QUICK_START_RTAMT.md                 📖 Quick start (NEW)
├── RTAMT_INTEGRATION_SUMMARY.md         📖 Summary (NEW)
├── ESSENTIAL_FILES.md                   📖 Experiment guide
├── CIFAR10_GUIDE.md                     📖 CIFAR-10 guide
├── FILE_STRUCTURE_GUIDE.md              📖 This file
├── multipliers/                         📂 Multiplier .bin files
└── logs/                                📂 Results
```

### Archive These (Optional)
```
nsa/archive/
├── main.py                              ❌ OLD
├── nas.py                               ❌ OLD
├── version_0.py                         ❌ OLD
├── test_simple_mnist.py                 ❌ Development
├── test_multiplier.py                   ❌ Development
├── test_two_stage_pattern.py            ❌ Development
├── test_reference_architecture.py       ❌ Development
├── test_nas_quick.py                    ❌ Development
├── test_all_multipliers_simple.py       ❌ Development
├── test_modelmul8u_2P7.py              ❌ Development
├── test_architecture_match.py           ❌ Development
├── test_architecture_complexity.py      ❌ Development
├── test_conv1x1_issue.py               ❌ Development
├── test_workflow.py                     ❌ Development
├── debug_model_match.py                 ❌ Debugging
├── diagnose_layers.py                   ❌ Debugging
├── multiplier_verfication_code.py       ❌ Debugging
├── architecture.py                      ❌ OLD
├── operations.py                        ❌ OLD
└── validation_utils.py                  ❌ OLD
```

---

## 🚀 How to Run Experiments

### 1. Quick RTAMT Test (3-5 minutes)
```bash
python test_rtamt_integration.py
```

### 2. Quick Hardware-Aware NAS Test (10 minutes)
```bash
python test_hardware_aware_nas.py
```

### 3. Full MNIST Experiment with RTAMT (2-3 hours)
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

### 4. Full CIFAR-10 Experiment with RTAMT (8-12 hours)
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

---

## 📊 What Gets Generated

After running an experiment, you'll find in `logs/`:

### Standard Outputs
- `nas_run_<timestamp>.log` - Detailed log
- `nas_results_<timestamp>.json` - Results in JSON
- `standard_model_weights.h5` - Trained model

### 🆕 RTAMT Outputs (NEW - when --use_rtamt is enabled)
- `verification_report_<timestamp>.json` - Verification results
- `pareto_frontier_<timestamp>.png` - Pareto frontier plot
- `accuracy_energy_scatter_<timestamp>.png` - Scatter plot
- `verification_summary_<timestamp>.png` - Verification summary

---

## 🎯 Key File Relationships

```
User runs: main_reference.py
    ↓
Calls: nas_reference.py (NAS engine)
    ↓
Uses: model_builder.py (builds architectures)
    ↓
Uses: layers.py (FakeApproxConv2D)
    ↓
Loads: multipliers/*.bin
    ↓
🆕 NEW: Verifies with rtamt_verifier.py (STL verification)
    ↓
🆕 NEW: Visualizes with pareto_visualization.py (Pareto frontier)
    ↓
Logs: logger.py → logs/
```

---

## 📝 Summary

**You only need these 11 core files to run experiments:**
1. `main_reference.py` - Run this
2. `nas_reference.py` - NAS engine
3. `model_builder.py` - Architecture
4. `dataloader.py` - Data loading
5. `training.py` - Evaluation
6. `logger.py` - Logging
7. `layers.py` - Custom layers
8. **NEW:** `rtamt_verifier.py` - Formal verification
9. **NEW:** `pareto_visualization.py` - Pareto plots
10. `test_hardware_aware_nas.py` - Quick test
11. **NEW:** `test_rtamt_integration.py` - RTAMT test

**Plus:**
- `multipliers/` directory (36 .bin files)
- Documentation files (.md)

**All other files** are legacy/development files that can be archived or deleted.

---

## 🆕 What's New (RTAMT Integration)

### New Files (6)
1. `rtamt_verifier.py` - Core verification engine
2. `pareto_visualization.py` - Visualization
3. `test_rtamt_integration.py` - Quick test
4. `RTAMT_INTEGRATION_GUIDE.md` - Complete guide
5. `QUICK_START_RTAMT.md` - Quick start
6. `RTAMT_INTEGRATION_SUMMARY.md` - Summary

### Modified Files (2)
1. `nas_reference.py` - Added RTAMT verification to each trial
2. `main_reference.py` - Added --use_rtamt flag and report generation

### New Features
✅ Formal verification using STL
✅ Training convergence verification
✅ Multiplier robustness verification
✅ Energy-accuracy tradeoff verification
✅ Pareto frontier generation
✅ Comprehensive visualizations
✅ Fitness penalties for failed verification

**To use**: Add `--use_rtamt` to any experiment command!
