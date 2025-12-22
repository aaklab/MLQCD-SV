#!/usr/bin/env python3
"""
Additional Physics Rules for Lattice QCD PINNS
"""

import numpy as np
import torch
import torch.nn as nn

class AdvancedPhysicsConstraints:
    """
    Collection of physics rules for Lattice QCD correlators
    """
    
    @staticmethod
    def positivity_constraint(predictions):
        """
        RULE 1: Positivity - C(t) ≥ 0 for all t
        Physics: Correlators are positive definite (reflection positivity)
        """
        violation = torch.relu(-predictions)  # Penalize negative values
        return violation.mean()
    
    @staticmethod
    def monotonic_decay_constraint(predictions, t_values, t_threshold=5):
        """
        RULE 2: Monotonic Decay - C(t+1) ≤ C(t) for t > t_threshold
        Physics: Exponential decay dominates at large times
        """
        late_times = t_values > t_threshold
        if late_times.sum() > 1:
            C_t = predictions[late_times[:-1]]
            C_t_plus_1 = predictions[late_times[1:]]
            violations = torch.relu(C_t_plus_1 - C_t)
            return violations.mean()
        return torch.tensor(0.0)
    
    @staticmethod
    def energy_ordering_constraint(E0, E1, E2=None):
        """
        RULE 3: Energy Ordering - E₂ > E₁ > E₀ > 0
        Physics: Excited states have higher energies
        """
        loss = torch.relu(E0 - 0.01)  # E0 > 0
        loss += torch.relu(E0 - E1 + 0.01)  # E1 > E0
        if E2 is not None:
            loss += torch.relu(E1 - E2 + 0.01)  # E2 > E1
        return loss
    
    @staticmethod
    def effective_mass_plateau_constraint(predictions, t_values, target_E0):
        """
        RULE 4: Effective Mass Plateau - m_eff(t) → E₀ for large t
        Physics: Effective mass should approach ground state energy
        """
        if len(predictions) > 2:
            # Effective mass: m_eff(t) = ln(C(t)/C(t+1))
            C_t = predictions[:-1]
            C_t_plus_1 = predictions[1:]
            
            # Avoid log(0) and negative ratios
            ratio = torch.clamp(C_t / (C_t_plus_1 + 1e-10), min=1e-10)
            m_eff = torch.log(ratio)
            
            # For large t, m_eff should approach E0
            late_times = t_values[:-1] > 8
            if late_times.sum() > 0:
                m_eff_late = m_eff[late_times]
                target = target_E0.expand_as(m_eff_late)
                return nn.MSELoss()(m_eff_late, target)
        
        return torch.tensor(0.0)
    
    @staticmethod
    def multi_exponential_constraint(predictions, t_values, A0, A1, E0, E1):
        """
        RULE 5: Multi-Exponential Form - C(t) = A₀e^(-E₀t) + A₁e^(-E₁t)
        Physics: Two-state spectral decomposition
        """
        # Ensure positive parameters
        A0_pos = torch.exp(A0)  # log-parameterization
        A1_pos = torch.exp(A1)
        E0_pos = torch.exp(E0)
        E1_pos = torch.exp(E1)
        
        # Analytical form
        analytical = A0_pos * torch.exp(-E0_pos * t_values) + A1_pos * torch.exp(-E1_pos * t_values)
        
        return nn.MSELoss()(predictions, analytical)
    
    @staticmethod
    def parity_constraint(predictions, parity=1):
        """
        RULE 6: Parity - C(-t) = P·C(t) where P = ±1
        Physics: Parity symmetry for certain operators
        """
        n = len(predictions)
        if n > 2:
            # Compare C(t) with P·C(T-t)
            forward = predictions[:n//2]
            backward = parity * predictions[n-1:n//2-1:-1] if n//2 > 0 else predictions[n//2:]
            
            min_len = min(len(forward), len(backward))
            if min_len > 0:
                return nn.MSELoss()(forward[:min_len], backward[:min_len])
        
        return torch.tensor(0.0)
    
    @staticmethod
    def causality_constraint(predictions, t_values, v_max=1.0):
        """
        RULE 7: Causality - Information cannot propagate faster than light
        Physics: Relativistic causality in lattice spacing units
        """
        # For 3-point functions: correlations should decay with distance
        # This is more complex and depends on the specific correlator geometry
        # Simplified: rapid changes should be penalized
        if len(predictions) > 1:
            derivatives = torch.abs(predictions[1:] - predictions[:-1])
            max_allowed_change = v_max / (t_values[1:] - t_values[:-1] + 1e-10)
            violations = torch.relu(derivatives - max_allowed_change)
            return violations.mean()
        
        return torch.tensor(0.0)
    
    @staticmethod
    def chiral_symmetry_constraint(predictions_vector, predictions_axial):
        """
        RULE 8: Chiral Symmetry - Relations between vector and axial correlators
        Physics: Chiral symmetry breaking patterns
        """
        # Example: In the chiral limit, certain relations hold
        # This would require paired vector/axial correlator data
        return nn.MSELoss()(predictions_vector, predictions_axial)
    
    @staticmethod
    def ward_identity_constraint(predictions, source_type):
        """
        RULE 9: Ward Identities - Conservation laws
        Physics: Current conservation, PCAC relations
        """
        # Example: Partially Conserved Axial Current (PCAC)
        # ∂μ A_μ = 2m_q P (axial current divergence)
        # This requires specific correlator combinations
        return torch.tensor(0.0)  # Placeholder
    
    @staticmethod
    def finite_volume_constraint(predictions, L_spatial):
        """
        RULE 10: Finite Volume Effects - Lüscher corrections
        Physics: Finite box size affects energy levels
        """
        # Lüscher formula corrections for finite volume
        # E_L = E_∞ + corrections(L, E_∞)
        # This is quite complex and requires specific implementations
        return torch.tensor(0.0)  # Placeholder


def create_comprehensive_physics_loss(model, predictions, targets, t_values, weights=None):
    """
    Combine multiple physics constraints with configurable weights
    """
    if weights is None:
        weights = {
            'data': 1.0,
            'positivity': 0.5,
            'monotonic_decay': 0.2,
            'energy_ordering': 0.3,
            'effective_mass': 0.2,
            'multi_exponential': 0.4,
            'parity': 0.1,
            'causality': 0.1,
        }
    
    physics = AdvancedPhysicsConstraints()
    
    # Standard data loss
    data_loss = nn.MSELoss()(predictions, targets)
    
    # Physics constraints
    losses = {
        'data_loss': data_loss,
        'positivity': physics.positivity_constraint(predictions),
        'monotonic_decay': physics.monotonic_decay_constraint(predictions, t_values),
        'parity': physics.parity_constraint(predictions, parity=1),
        'causality': physics.causality_constraint(predictions, t_values),
    }
    
    # Add energy ordering if model has learnable parameters
    if hasattr(model, 'log_E0') and hasattr(model, 'log_E1'):
        E0 = torch.exp(model.log_E0)
        E1 = torch.exp(model.log_E1)
        losses['energy_ordering'] = physics.energy_ordering_constraint(E0, E1)
        
        # Add effective mass constraint
        losses['effective_mass'] = physics.effective_mass_plateau_constraint(predictions, t_values, E0)
        
        # Add multi-exponential constraint
        if hasattr(model, 'log_A0') and hasattr(model, 'log_A1'):
            losses['multi_exponential'] = physics.multi_exponential_constraint(
                predictions, t_values, model.log_A0, model.log_A1, model.log_E0, model.log_E1
            )
    
    # Combine with weights
    total_loss = weights['data'] * data_loss
    
    for loss_name, loss_value in losses.items():
        if loss_name != 'data_loss' and loss_name in weights:
            total_loss += weights[loss_name] * loss_value
    
    return total_loss, {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}


# Configuration for different physics rule combinations
PHYSICS_RULE_SETS = {
    'basic': {
        'rules': ['positivity', 'monotonic_decay'],
        'description': 'Basic physical constraints',
        'weights': {'positivity': 0.5, 'monotonic_decay': 0.2}
    },
    
    'spectral': {
        'rules': ['positivity', 'energy_ordering', 'multi_exponential', 'effective_mass'],
        'description': 'Spectral decomposition constraints',
        'weights': {'positivity': 0.3, 'energy_ordering': 0.4, 'multi_exponential': 0.5, 'effective_mass': 0.3}
    },
    
    'symmetry': {
        'rules': ['positivity', 'parity', 'monotonic_decay'],
        'description': 'Symmetry-based constraints',
        'weights': {'positivity': 0.4, 'parity': 0.3, 'monotonic_decay': 0.2}
    },
    
    'comprehensive': {
        'rules': ['positivity', 'monotonic_decay', 'energy_ordering', 'effective_mass', 
                 'multi_exponential', 'parity', 'causality'],
        'description': 'All available physics constraints',
        'weights': {
            'positivity': 0.3, 'monotonic_decay': 0.2, 'energy_ordering': 0.4,
            'effective_mass': 0.2, 'multi_exponential': 0.4, 'parity': 0.1, 'causality': 0.1
        }
    }
}

if __name__ == "__main__":
    print("ADDITIONAL PHYSICS RULES FOR LATTICE QCD PINNS")
    print("=" * 60)
    
    rules = [
        ("Positivity", "C(t) ≥ 0", "Reflection positivity"),
        ("Monotonic Decay", "C(t+1) ≤ C(t) for large t", "Exponential decay dominance"),
        ("Energy Ordering", "E₂ > E₁ > E₀ > 0", "Excited states hierarchy"),
        ("Effective Mass Plateau", "m_eff(t) → E₀", "Ground state dominance"),
        ("Multi-Exponential", "C(t) = ΣAᵢe^(-Eᵢt)", "Spectral decomposition"),
        ("Parity", "C(-t) = P·C(t)", "Parity symmetry"),
        ("Causality", "No superluminal propagation", "Relativistic causality"),
        ("Chiral Symmetry", "Vector-axial relations", "Chiral symmetry breaking"),
        ("Ward Identities", "Current conservation", "Gauge invariance"),
        ("Finite Volume", "Lüscher corrections", "Box size effects"),
    ]
    
    for i, (name, formula, physics) in enumerate(rules, 1):
        print(f"{i:2d}. {name:<20} | {formula:<25} | {physics}")
    
    print("\n" + "=" * 60)
    print("RULE SET RECOMMENDATIONS:")
    for name, config in PHYSICS_RULE_SETS.items():
        print(f"\n{name.upper()}:")
        print(f"  Rules: {', '.join(config['rules'])}")
        print(f"  Use case: {config['description']}")
    print("=" * 60)