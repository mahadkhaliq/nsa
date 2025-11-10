#!/bin/bash
# Hardware-Aware NAS Experiment Runner
# Usage: ./run_experiments.sh [experiment_name]

set -e  # Exit on error

echo "=========================================="
echo "Hardware-Aware NAS Experiment Runner"
echo "=========================================="
echo ""

# Check if experiment name is provided
if [ -z "$1" ]; then
    echo "Available experiments:"
    echo ""
    echo "  quick         - Quick test (3 trials, 5 multipliers, ~10 min)"
    echo "  test_all_muls - Test all 36 multipliers with default arch (~30 min)"
    echo "  standard_nas  - Standard NAS baseline (~2 hours)"
    echo "  hw_nas_10     - Hardware-aware NAS with 10 multipliers (~4 hours)"
    echo "  hw_nas_all    - Hardware-aware NAS with ALL 36 multipliers (~12 hours)"
    echo ""
    echo "CIFAR-10 experiments:"
    echo "  cifar10_quick      - Quick CIFAR-10 test (3 trials, ~45 min)"
    echo "  cifar10_test_muls  - Test all multipliers on CIFAR-10 (~1 hour)"
    echo "  cifar10_standard   - CIFAR-10 standard NAS (~8 hours)"
    echo "  cifar10_hw_nas     - CIFAR-10 hardware-aware NAS (~16 hours)"
    echo ""
    echo "Usage: ./run_experiments.sh [experiment_name]"
    echo "Example: ./run_experiments.sh quick"
    exit 1
fi

EXPERIMENT=$1

case $EXPERIMENT in
    quick)
        echo "Running Quick Test (3 trials, 5 multipliers)"
        echo "Expected time: ~10 minutes"
        echo "=========================================="
        python test_hardware_aware_nas.py
        ;;

    test_all_muls)
        echo "Testing ALL 36 multipliers with default architecture"
        echo "Expected time: ~30 minutes"
        echo "=========================================="
        python main_reference.py \
            --dataset mnist \
            --skip_nas \
            --test_all_multipliers \
            --train_samples 10000 \
            --val_samples 2000 \
            --final_epochs 30 \
            --log_dir logs/default_arch_all_multipliers
        ;;

    standard_nas)
        echo "Standard NAS (baseline, no multipliers)"
        echo "Expected time: ~2 hours"
        echo "=========================================="
        python main_reference.py \
            --dataset mnist \
            --nas_trials 20 \
            --nas_method evolutionary \
            --epochs_per_trial 15 \
            --final_epochs 30 \
            --train_samples 10000 \
            --val_samples 2000 \
            --test_all_multipliers \
            --log_dir logs/mnist_standard_nas
        ;;

    hw_nas_10)
        echo "Hardware-Aware NAS with 10 multipliers"
        echo "Expected time: ~4 hours"
        echo "=========================================="
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
            --log_dir logs/mnist_hw_nas_10
        ;;

    hw_nas_all)
        echo "Hardware-Aware NAS with ALL 36 multipliers"
        echo "Expected time: ~12 hours"
        echo "WARNING: This will take a long time!"
        echo "=========================================="
        read -p "Are you sure? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
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
                --log_dir logs/mnist_hw_nas_all
        else
            echo "Cancelled."
            exit 0
        fi
        ;;

    cifar10)
        echo "CIFAR-10 Hardware-Aware NAS"
        echo "Expected time: ~8 hours"
        echo "=========================================="
        python main_reference.py \
            --dataset cifar10 \
            --nas_trials 25 \
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
        ;;

    *)
        echo "ERROR: Unknown experiment '$EXPERIMENT'"
        echo "Run './run_experiments.sh' to see available experiments"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Experiment '$EXPERIMENT' completed!"
echo "Check logs/ directory for results"
echo "=========================================="
