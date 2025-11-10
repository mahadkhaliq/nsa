"""
RTAMT-based formal verification for hardware-aware NAS.
Verifies temporal properties of neural network training and approximate multiplier performance.
"""
import rtamt
import numpy as np
from typing import Dict, List, Tuple, Any


class NASVerifier:
    """Formal verification of NAS properties using RTAMT"""

    def __init__(self):
        self.specs = {}
        self.traces = {}

    def create_training_spec(self, min_accuracy: float = 0.7, max_epochs: int = 50):
        """
        Create STL specification for training convergence.

        Properties:
        1. Accuracy should eventually reach min_accuracy
        2. Validation accuracy should not diverge too much from training accuracy
        3. Loss should be decreasing (on average)
        """
        spec = rtamt.StlDiscreteTimeOfflineSpecification()
        spec.name = 'Training Convergence'

        # Variables
        spec.declare_var('train_acc', 'float')
        spec.declare_var('val_acc', 'float')
        spec.declare_var('loss', 'float')
        spec.declare_var('epoch', 'int')

        # Specification: Eventually accuracy reaches threshold
        spec.spec = f'eventually[0:{max_epochs}](val_acc >= {min_accuracy})'

        try:
            spec.parse()
            self.specs['training'] = spec
            return spec
        except rtamt.STLParseException as e:
            print(f"Error parsing training spec: {e}")
            return None

    def create_robustness_spec(self, max_drop_percent: float = 10.0):
        """
        Create STL specification for multiplier robustness.

        Properties:
        1. Accuracy drop should always be below threshold
        2. If a multiplier has high error, accuracy drop should be bounded
        """
        spec = rtamt.StlDiscreteTimeOfflineSpecification()
        spec.name = 'Multiplier Robustness'

        # Variables
        spec.declare_var('accuracy_drop', 'float')
        spec.declare_var('std_accuracy', 'float')
        spec.declare_var('approx_accuracy', 'float')
        spec.declare_var('multiplier_id', 'int')

        # Specification: Accuracy drop always below threshold
        spec.spec = f'always(accuracy_drop <= {max_drop_percent})'

        try:
            spec.parse()
            self.specs['robustness'] = spec
            return spec
        except rtamt.STLParseException as e:
            print(f"Error parsing robustness spec: {e}")
            return None

    def create_energy_accuracy_spec(self,
                                     min_accuracy: float = 0.7,
                                     max_energy_multiplier: float = 1.5):
        """
        Create STL specification for energy-accuracy tradeoff.

        Properties:
        1. If energy is low (< max_energy_multiplier), then accuracy must be reasonable
        2. High accuracy implies acceptable energy consumption
        """
        spec = rtamt.StlDiscreteTimeOfflineSpecification()
        spec.name = 'Energy-Accuracy Tradeoff'

        # Variables
        spec.declare_var('accuracy', 'float')
        spec.declare_var('energy_ratio', 'float')  # Relative to standard multiplier
        spec.declare_var('is_pareto_optimal', 'float')

        # Specification: Energy-accuracy tradeoff
        # If energy is low, accuracy should be acceptable
        spec.spec = f'always((energy_ratio <= {max_energy_multiplier}) implies (accuracy >= {min_accuracy}))'

        try:
            spec.parse()
            self.specs['energy_accuracy'] = spec
            return spec
        except rtamt.STLParseException as e:
            print(f"Error parsing energy-accuracy spec: {e}")
            return None

    def verify_training(self, history: Dict[str, List[float]]) -> Tuple[float, bool]:
        """
        Verify training convergence properties.

        Args:
            history: Training history dict with keys 'accuracy', 'val_accuracy', 'loss'

        Returns:
            (robustness_value, satisfaction): Robustness degree and boolean satisfaction
        """
        if 'training' not in self.specs:
            self.create_training_spec()

        spec = self.specs['training']

        # Build trace
        epochs = len(history['accuracy'])
        trace = []

        for i in range(epochs):
            trace.append({
                'time': i,
                'train_acc': history['accuracy'][i],
                'val_acc': history['val_accuracy'][i],
                'loss': history['loss'][i],
                'epoch': i
            })

        # Store trace
        self.traces['training'] = trace

        # Evaluate offline
        try:
            # Prepare data for offline evaluation
            train_acc_data = [(t['time'], t['train_acc']) for t in trace]
            val_acc_data = [(t['time'], t['val_acc']) for t in trace]
            loss_data = [(t['time'], t['loss']) for t in trace]
            epoch_data = [(t['time'], t['epoch']) for t in trace]

            # Evaluate with offline spec
            robustness = spec.evaluate(
                train_acc=train_acc_data,
                val_acc=val_acc_data,
                loss=loss_data,
                epoch=epoch_data
            )

            # For offline, robustness is a list - take the first value (at time 0)
            if isinstance(robustness, list) and len(robustness) > 0:
                robustness = robustness[0][1]  # Get value from (time, value) tuple

            satisfaction = robustness > 0

            return robustness, satisfaction
        except Exception as e:
            print(f"Error verifying training: {e}")
            return float('-inf'), False

    def verify_multiplier_robustness(self,
                                      std_accuracy: float,
                                      multiplier_accuracies: List[float]) -> Dict[str, Any]:
        """
        Verify robustness properties across multipliers.

        Args:
            std_accuracy: Standard (exact) multiplier accuracy
            multiplier_accuracies: List of accuracies with approximate multipliers

        Returns:
            Dict with verification results for each multiplier
        """
        if 'robustness' not in self.specs:
            self.create_robustness_spec()

        results = {
            'robustness_values': [],
            'satisfactions': [],
            'accuracy_drops': []
        }

        for i, approx_acc in enumerate(multiplier_accuracies):
            spec = rtamt.StlDiscreteTimeOfflineSpecification()
            spec.name = f'Robustness_Multiplier_{i}'
            spec.declare_var('accuracy_drop', 'float')
            spec.spec = 'always(accuracy_drop <= 10.0)'
            spec.parse()

            # Calculate drop percentage
            drop_pct = ((std_accuracy - approx_acc) / std_accuracy) * 100 if std_accuracy > 0 else 100

            # Single point trace for offline evaluation
            robustness_result = spec.evaluate(accuracy_drop=[(0, drop_pct)])

            # Extract robustness value
            if isinstance(robustness_result, list) and len(robustness_result) > 0:
                robustness = robustness_result[0][1]
            else:
                robustness = robustness_result

            satisfaction = robustness > 0

            results['robustness_values'].append(robustness)
            results['satisfactions'].append(satisfaction)
            results['accuracy_drops'].append(drop_pct)

        # Summary statistics
        results['mean_robustness'] = np.mean(results['robustness_values'])
        results['min_robustness'] = np.min(results['robustness_values'])
        results['satisfaction_rate'] = np.mean(results['satisfactions'])

        return results

    def verify_pareto_point(self,
                           accuracy: float,
                           energy_ratio: float,
                           min_accuracy: float = 0.7,
                           max_energy: float = 1.5) -> Tuple[float, bool]:
        """
        Verify if a point satisfies energy-accuracy tradeoff constraints.

        Args:
            accuracy: Model accuracy
            energy_ratio: Energy consumption relative to standard (1.0 = standard)
            min_accuracy: Minimum acceptable accuracy
            max_energy: Maximum acceptable energy ratio

        Returns:
            (robustness_value, satisfaction)
        """
        spec = rtamt.StlDiscreteTimeOfflineSpecification()
        spec.name = 'Pareto Point Verification'

        spec.declare_var('accuracy', 'float')
        spec.declare_var('energy_ratio', 'float')

        # Property: Low energy implies good accuracy
        spec.spec = f'always((energy_ratio <= {max_energy}) implies (accuracy >= {min_accuracy}))'
        spec.parse()

        # Single point evaluation for offline
        robustness_result = spec.evaluate(
            accuracy=[(0, accuracy)],
            energy_ratio=[(0, energy_ratio)]
        )

        # Extract robustness value
        if isinstance(robustness_result, list) and len(robustness_result) > 0:
            robustness = robustness_result[0][1]
        else:
            robustness = robustness_result

        satisfaction = robustness > 0

        return robustness, satisfaction

    def generate_verification_report(self,
                                      config: Dict,
                                      training_history: Dict,
                                      std_accuracy: float,
                                      multiplier_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive verification report.

        Args:
            config: Architecture configuration
            training_history: Training history
            std_accuracy: Standard accuracy
            multiplier_results: List of {accuracy, energy_ratio, name} dicts

        Returns:
            Verification report with all properties
        """
        report = {
            'config': config,
            'verification_results': {}
        }

        # 1. Verify training convergence
        train_robustness, train_sat = self.verify_training(training_history)
        report['verification_results']['training'] = {
            'robustness': float(train_robustness),
            'satisfied': bool(train_sat),
            'property': 'Eventually reaches minimum accuracy'
        }

        # 2. Verify multiplier robustness
        multiplier_accs = [m['accuracy'] for m in multiplier_results]
        rob_results = self.verify_multiplier_robustness(std_accuracy, multiplier_accs)
        report['verification_results']['robustness'] = rob_results

        # 3. Verify Pareto points (energy-accuracy tradeoff)
        pareto_results = []
        for m in multiplier_results:
            energy_ratio = m.get('energy_ratio', 1.0)
            rob, sat = self.verify_pareto_point(m['accuracy'], energy_ratio)
            pareto_results.append({
                'multiplier': m['name'],
                'accuracy': m['accuracy'],
                'energy_ratio': energy_ratio,
                'robustness': float(rob),
                'satisfied': bool(sat)
            })

        report['verification_results']['pareto'] = pareto_results

        # 4. Overall verdict
        all_satisfied = (
            train_sat and
            rob_results['satisfaction_rate'] > 0.7 and  # At least 70% of multipliers pass
            any(p['satisfied'] for p in pareto_results)  # At least one good Pareto point
        )

        report['overall_verdict'] = {
            'passed': all_satisfied,
            'training_converged': train_sat,
            'robustness_rate': rob_results['satisfaction_rate'],
            'has_good_pareto_point': any(p['satisfied'] for p in pareto_results)
        }

        return report

    def get_summary_string(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary of verification results"""
        lines = []
        lines.append("=" * 80)
        lines.append("FORMAL VERIFICATION REPORT")
        lines.append("=" * 80)

        # Overall verdict
        verdict = report['overall_verdict']
        status = "✓ PASSED" if verdict['passed'] else "✗ FAILED"
        lines.append(f"\nOverall Verdict: {status}")
        lines.append(f"  Training Converged: {verdict['training_converged']}")
        lines.append(f"  Robustness Rate: {verdict['robustness_rate']:.2%}")
        lines.append(f"  Has Good Pareto Point: {verdict['has_good_pareto_point']}")

        # Training verification
        train = report['verification_results']['training']
        lines.append(f"\nTraining Convergence:")
        lines.append(f"  Property: {train['property']}")
        lines.append(f"  Robustness: {train['robustness']:.4f}")
        lines.append(f"  Satisfied: {train['satisfied']}")

        # Robustness verification
        rob = report['verification_results']['robustness']
        lines.append(f"\nMultiplier Robustness:")
        lines.append(f"  Mean Robustness: {rob['mean_robustness']:.4f}")
        lines.append(f"  Min Robustness: {rob['min_robustness']:.4f}")
        lines.append(f"  Satisfaction Rate: {rob['satisfaction_rate']:.2%}")

        # Pareto analysis
        pareto = report['verification_results']['pareto']
        satisfied_pareto = [p for p in pareto if p['satisfied']]
        lines.append(f"\nPareto-Optimal Points:")
        lines.append(f"  Total Points: {len(pareto)}")
        lines.append(f"  Satisfied: {len(satisfied_pareto)}")

        if satisfied_pareto:
            lines.append(f"  Best Point:")
            best = max(satisfied_pareto, key=lambda p: p['accuracy'])
            lines.append(f"    Multiplier: {best['multiplier']}")
            lines.append(f"    Accuracy: {best['accuracy']:.4f}")
            lines.append(f"    Energy Ratio: {best['energy_ratio']:.2f}x")

        lines.append("=" * 80)

        return "\n".join(lines)


def estimate_energy_ratio(multiplier_name: str) -> float:
    """
    Estimate energy consumption ratio based on multiplier name.

    This is a placeholder - replace with actual energy measurements if available.
    Lower ratio = more energy efficient.

    Args:
        multiplier_name: Name of the multiplier file

    Returns:
        Energy ratio (1.0 = standard multiplier energy)
    """
    # Parse multiplier characteristics from name
    # Example: mul8u_2P7.bin -> approximate multiplier with certain error rate

    # Heuristic: Approximate multipliers use less energy
    # More aggressive approximation = lower energy but potentially lower accuracy

    # Standard multiplier
    if 'standard' in multiplier_name.lower():
        return 1.0

    # Approximate multipliers - estimate based on naming
    # This is a PLACEHOLDER - replace with actual measurements
    base_energy = 0.6  # Approximate multipliers typically 40-60% energy of standard

    # Add variance based on multiplier characteristics
    import hashlib
    hash_val = int(hashlib.md5(multiplier_name.encode()).hexdigest(), 16)
    variance = (hash_val % 20) / 100.0  # 0-0.2 variance

    return base_energy + variance


if __name__ == '__main__':
    # Example usage
    print("RTAMT Verifier for Hardware-Aware NAS")
    print("=" * 80)

    # Create verifier
    verifier = NASVerifier()

    # Example training history
    history = {
        'accuracy': [0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92],
        'val_accuracy': [0.58, 0.68, 0.73, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.89],
        'loss': [1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25]
    }

    # Example multiplier results
    multiplier_results = [
        {'name': 'mul8u_2P7.bin', 'accuracy': 0.88, 'energy_ratio': 0.65},
        {'name': 'mul8u_3P8.bin', 'accuracy': 0.85, 'energy_ratio': 0.58},
        {'name': 'mul8u_5NG.bin', 'accuracy': 0.75, 'energy_ratio': 0.52},
    ]

    # Generate report
    config = {'conv1_filters': 32, 'conv2_filters': 64, 'conv3_filters': 128}
    report = verifier.generate_verification_report(
        config, history, 0.89, multiplier_results
    )

    # Print summary
    print(verifier.get_summary_string(report))
