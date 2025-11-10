"""
Pareto Frontier Visualization for Hardware-Aware NAS
Visualizes the accuracy-energy tradeoff across different multipliers
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import json
from typing import List, Dict, Any


def plot_pareto_frontier(multiplier_results: List[Dict[str, Any]],
                        output_file: str = 'pareto_frontier.png',
                        title: str = 'Accuracy-Energy Pareto Frontier',
                        show_labels: bool = True):
    """
    Plot Pareto frontier showing accuracy vs energy tradeoff

    Args:
        multiplier_results: List of dicts with 'name', 'accuracy', 'energy_ratio'
        output_file: Path to save the plot
        title: Plot title
        show_labels: Whether to show multiplier names on plot
    """
    if not multiplier_results:
        print("No multiplier results to plot")
        return

    # Extract data
    names = [r['name'] for r in multiplier_results]
    accuracies = np.array([r['accuracy'] for r in multiplier_results])
    energy_ratios = np.array([r.get('energy_ratio', 1.0) for r in multiplier_results])

    # Identify Pareto-optimal points
    # A point is Pareto-optimal if no other point dominates it
    # (higher accuracy AND lower energy)
    pareto_mask = np.zeros(len(multiplier_results), dtype=bool)

    for i in range(len(multiplier_results)):
        is_dominated = False
        for j in range(len(multiplier_results)):
            if i != j:
                # Point j dominates point i if it has both:
                # - Higher or equal accuracy
                # - Lower or equal energy
                # (and at least one strict inequality)
                if (accuracies[j] >= accuracies[i] and
                    energy_ratios[j] <= energy_ratios[i] and
                    (accuracies[j] > accuracies[i] or energy_ratios[j] < energy_ratios[i])):
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_mask[i] = True

    # Sort Pareto points by energy for line connection
    pareto_indices = np.where(pareto_mask)[0]
    pareto_points = [(energy_ratios[i], accuracies[i]) for i in pareto_indices]
    pareto_points.sort(key=lambda x: x[0])  # Sort by energy

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all points
    ax.scatter(energy_ratios[~pareto_mask], accuracies[~pareto_mask],
               c='lightblue', s=100, alpha=0.6, label='Non-Pareto', zorder=2)

    # Plot Pareto-optimal points
    ax.scatter(energy_ratios[pareto_mask], accuracies[pareto_mask],
               c='red', s=150, marker='*', label='Pareto-Optimal', zorder=3)

    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2, zorder=1)

    # Add labels for Pareto points
    if show_labels:
        for i in pareto_indices:
            ax.annotate(names[i].replace('.bin', ''),
                       (energy_ratios[i], accuracies[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

    # Formatting
    ax.set_xlabel('Energy Ratio (lower is better)', fontsize=12)
    ax.set_ylabel('Accuracy (higher is better)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add ideal region annotation
    if len(multiplier_results) > 0:
        max_acc = max(accuracies)
        min_energy = min(energy_ratios)
        ax.annotate('Ideal Region\n(High Acc, Low Energy)',
                   xy=(min_energy, max_acc),
                   xytext=(10, -30), textcoords='offset points',
                   fontsize=10, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Pareto frontier plot saved to: {output_file}")
    plt.close()

    # Return statistics
    pareto_count = np.sum(pareto_mask)
    stats = {
        'total_points': len(multiplier_results),
        'pareto_optimal_count': int(pareto_count),
        'pareto_points': [names[i] for i in pareto_indices],
        'best_accuracy': {
            'name': names[np.argmax(accuracies)],
            'accuracy': float(max(accuracies)),
            'energy_ratio': float(energy_ratios[np.argmax(accuracies)])
        },
        'best_energy': {
            'name': names[np.argmin(energy_ratios)],
            'accuracy': float(accuracies[np.argmin(energy_ratios)]),
            'energy_ratio': float(min(energy_ratios))
        }
    }

    return stats


def plot_accuracy_energy_scatter(multiplier_results: List[Dict[str, Any]],
                                 output_file: str = 'accuracy_energy_scatter.png'):
    """
    Create a scatter plot with accuracy and energy information

    Args:
        multiplier_results: List of dicts with 'name', 'accuracy', 'energy_ratio'
        output_file: Path to save the plot
    """
    if not multiplier_results:
        print("No multiplier results to plot")
        return

    names = [r['name'] for r in multiplier_results]
    accuracies = np.array([r['accuracy'] for r in multiplier_results])
    energy_ratios = np.array([r.get('energy_ratio', 1.0) for r in multiplier_results])

    # Create color map based on quality score
    # Quality = normalized_accuracy + (1 - normalized_energy)
    norm_acc = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min() + 1e-8)
    norm_energy = (energy_ratios - energy_ratios.min()) / (energy_ratios.max() - energy_ratios.min() + 1e-8)
    quality = norm_acc + (1 - norm_energy)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(range(len(names)), accuracies,
                        c=energy_ratios, s=200, alpha=0.6,
                        cmap='RdYlGn_r')

    # Color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy Ratio', rotation=270, labelpad=20)

    # Formatting
    ax.set_xlabel('Multiplier Index', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Multiplier Performance: Accuracy vs Energy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('.bin', '') for n in names], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_file}")
    plt.close()


def plot_nas_verification_summary(verification_report: Dict[str, Any],
                                  output_file: str = 'verification_summary.png'):
    """
    Create visualization of RTAMT verification results

    Args:
        verification_report: Verification report from NASVerifier
        output_file: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Convergence
    ax1 = axes[0, 0]
    training = verification_report['verification_results']['training']
    robustness = training['robustness']
    satisfied = training['satisfied']

    colors = ['green' if satisfied else 'red']
    ax1.bar(['Training\nConvergence'], [robustness], color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel('Robustness Value', fontsize=10)
    ax1.set_title('Training Convergence Verification', fontsize=11, fontweight='bold')
    ax1.set_ylim([min(robustness, -0.1), max(robustness, 0.1) + 0.05])
    ax1.grid(True, alpha=0.3, axis='y')
    status_text = '✓ PASS' if satisfied else '✗ FAIL'
    ax1.text(0, robustness, status_text, ha='center', va='bottom', fontweight='bold')

    # Plot 2: Robustness Distribution
    ax2 = axes[0, 1]
    robustness_results = verification_report['verification_results']['robustness']
    if robustness_results:
        rob_values = robustness_results['robustness_values']
        satisfactions = robustness_results['satisfactions']

        colors = ['green' if s else 'red' for s in satisfactions]
        x_pos = range(len(rob_values))
        ax2.bar(x_pos, rob_values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Multiplier Index', fontsize=10)
        ax2.set_ylabel('Robustness Value', fontsize=10)
        ax2.set_title(f'Multiplier Robustness (Pass Rate: {robustness_results["satisfaction_rate"]:.1%})',
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Pareto Points
    ax3 = axes[1, 0]
    pareto_results = verification_report['verification_results']['pareto']
    if pareto_results:
        pareto_names = [p['multiplier'].replace('.bin', '')[:10] for p in pareto_results]
        pareto_accs = [p['accuracy'] for p in pareto_results]
        pareto_satisfied = [p['satisfied'] for p in pareto_results]

        colors = ['green' if s else 'red' for s in pareto_satisfied]
        x_pos = range(len(pareto_names))
        ax3.bar(x_pos, pareto_accs, color=colors, alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(pareto_names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Accuracy', fontsize=10)
        ax3.set_title('Pareto Point Verification', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Overall Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    verdict = verification_report['overall_verdict']
    summary_text = f"""
VERIFICATION SUMMARY

Overall Status: {'✓ PASSED' if verdict['passed'] else '✗ FAILED'}

Training Converged: {verdict['training_converged']}
Robustness Rate: {verdict['robustness_rate']:.1%}
Has Good Pareto Point: {verdict['has_good_pareto_point']}

{'━' * 40}

Properties Verified:
• Training convergence (STL)
• Multiplier robustness (STL)
• Energy-accuracy tradeoff (STL)
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Verification summary plot saved to: {output_file}")
    plt.close()


if __name__ == '__main__':
    # Example usage with dummy data
    print("Pareto Frontier Visualization Module")
    print("=" * 80)

    # Example multiplier results
    example_results = [
        {'name': 'mul8u_2P7.bin', 'accuracy': 0.88, 'energy_ratio': 0.65},
        {'name': 'mul8u_3P8.bin', 'accuracy': 0.85, 'energy_ratio': 0.58},
        {'name': 'mul8u_5NG.bin', 'accuracy': 0.75, 'energy_ratio': 0.52},
        {'name': 'mul8u_6KH.bin', 'accuracy': 0.82, 'energy_ratio': 0.60},
        {'name': 'standard.bin', 'accuracy': 0.90, 'energy_ratio': 1.00},
    ]

    # Generate Pareto frontier plot
    stats = plot_pareto_frontier(example_results, 'example_pareto.png')
    print(f"\nPareto Statistics:")
    print(f"  Total points: {stats['total_points']}")
    print(f"  Pareto-optimal: {stats['pareto_optimal_count']}")
    print(f"  Pareto points: {stats['pareto_points']}")
    print(f"  Best accuracy: {stats['best_accuracy']}")
    print(f"  Best energy: {stats['best_energy']}")

    # Generate scatter plot
    plot_accuracy_energy_scatter(example_results, 'example_scatter.png')

    print("\n" + "=" * 80)
    print("Example plots generated successfully!")
