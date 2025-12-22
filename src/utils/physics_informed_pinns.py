#!/usr/bin/env python3
"""
Enhanced Physics-Informed Neural Networks for Lattice QCD
This implements explicit physics constraints as loss functions
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

class PhysicsInformedPINNS(nn.Module):
    """
    True Physics-Informed Neural Network for Lattice QCD correlators
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 32]):
        super().__init__()
        
        # Standard neural network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),  # Good for physics problems
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable physics parameters
        self.log_A0 = nn.Parameter(torch.tensor(-2.0))  # log(amplitude_0)
        self.log_A1 = nn.Parameter(torch.tensor(-3.0))  # log(amplitude_1)
        self.log_E0 = nn.Parameter(torch.tensor(-2.0))  # log(energy_0)
        self.log_E1 = nn.Parameter(torch.tensor(-1.0))  # log(energy_1)
        
    def forward(self, x):
        """Forward pass through network"""
        return self.network(x)
    
    def physics_correlator(self, t):
        """
        Analytical correlator form: C(t) = A0*exp(-E0*t) + A1*exp(-E1*t)
        """
        A0 = torch.exp(self.log_A0)  # Ensure positive
        A1 = torch.exp(self.log_A1)  # Ensure positive
        E0 = torch.exp(self.log_E0)  # Ensure positive
        E1 = torch.exp(self.log_E1)  # Ensure positive
        
        # Two-state exponential form
        C_physics = A0 * torch.exp(-E0 * t) + A1 * torch.exp(-E1 * t)
        return C_physics
    
    def get_physics_parameters(self):
        """Extract physics parameters"""
        return {
            'A0': torch.exp(self.log_A0).item(),
            'A1': torch.exp(self.log_A1).item(), 
            'E0': torch.exp(self.log_E0).item(),
            'E1': torch.exp(self.log_E1).item(),
        }


def physics_loss_functions(model, predictions, targets, t_values):
    """
    Comprehensive physics loss functions for Lattice QCD
    """
    losses = {}
    
    # 1. EXPONENTIAL DECAY CONSTRAINT
    # Correlators should decay exponentially: C(t+1) ≤ C(t) for large t
    def exponential_decay_loss():
        # Check monotonic decrease for t > 5 (avoid early-time oscillations)
        late_times = t_values > 5
        if late_times.sum() > 1:
            C_t = predictions[late_times[:-1]]
            C_t_plus_1 = predictions[late_times[1:]]
            
            # Penalize increases: max(0, C(t+1) - C(t))
            violations = torch.relu(C_t_plus_1 - C_t)
            return violations.mean()
        return torch.tensor(0.0)
    
    # 2. POSITIVITY CONSTRAINT  
    # Correlators should be positive: C(t) ≥ 0
    def positivity_loss():
        negative_values = torch.relu(-predictions)
        return negative_values.mean()
    
    # 3. ENERGY ORDERING CONSTRAINT
    # Excited state energy should be higher: E1 > E0
    def energy_ordering_loss():
        params = model.get_physics_parameters()
        E0, E1 = params['E0'], params['E1']
        
        # Penalize E1 ≤ E0
        violation = torch.relu(torch.tensor(E0 - E1 + 0.01))  # Small margin
        return violation
    
    # 4. SPECTRAL FORM CONSTRAINT
    # Predictions should match analytical two-state form
    def spectral_form_loss():
        physics_pred = model.physics_correlator(t_values)
        return nn.MSELoss()(predictions, physics_pred)
    
    # 5. SYMMETRY CONSTRAINT (for appropriate correlators)
    # Some correlators have C(t) = C(T-t) symmetry
    def symmetry_loss():
        T = len(predictions)
        if T > 10:  # Only for reasonable time extents
            # Compare C(t) with C(T-1-t)
            forward = predictions[:T//2]
            backward = predictions[T-1:T//2-1:-1] if T//2 > 0 else predictions[T//2:]
            
            min_len = min(len(forward), len(backward))
            if min_len > 0:
                return nn.MSELoss()(forward[:min_len], backward[:min_len])
        return torch.tensor(0.0)
    
    # 6. EFFECTIVE MASS CONSTRAINT
    # Effective mass should approach ground state: m_eff(t) → E0 for large t
    def effective_mass_loss():
        if len(predictions) > 2:
            # Effective mass: m_eff(t) = ln(C(t)/C(t+1))
            C_t = predictions[:-1]
            C_t_plus_1 = predictions[1:]
            
            # Avoid log(0) and negative values
            ratio = torch.clamp(C_t / (C_t_plus_1 + 1e-10), min=1e-10)
            m_eff = torch.log(ratio)
            
            # For large t, m_eff should approach E0
            E0 = torch.exp(model.log_E0)
            late_times = t_values[:-1] > 8  # Focus on late times
            
            if late_times.sum() > 0:
                m_eff_late = m_eff[late_times]
                target_mass = E0.expand_as(m_eff_late)
                return nn.MSELoss()(m_eff_late, target_mass)
        
        return torch.tensor(0.0)
    
    # Compute all losses
    losses['exponential_decay'] = exponential_decay_loss()
    losses['positivity'] = positivity_loss()
    losses['energy_ordering'] = energy_ordering_loss()
    losses['spectral_form'] = spectral_form_loss()
    losses['symmetry'] = symmetry_loss()
    losses['effective_mass'] = effective_mass_loss()
    
    return losses


def combined_physics_loss(model, predictions, targets, t_values, weights=None):
    """
    Combine all physics losses with configurable weights
    """
    if weights is None:
        weights = {
            'data': 1.0,              # Standard data fitting loss
            'exponential_decay': 0.1,  # Monotonic decrease
            'positivity': 0.5,         # Must be positive
            'energy_ordering': 0.2,    # E1 > E0
            'spectral_form': 0.3,      # Match analytical form
            'symmetry': 0.1,           # Correlator symmetry
            'effective_mass': 0.2,     # Effective mass behavior
        }
    
    # Standard data loss
    data_loss = nn.MSELoss()(predictions, targets)
    
    # Physics losses
    physics_losses = physics_loss_functions(model, predictions, targets, t_values)
    
    # Combine with weights
    total_loss = weights['data'] * data_loss
    
    for loss_name, loss_value in physics_losses.items():
        if loss_name in weights:
            total_loss += weights[loss_name] * loss_value
    
    return total_loss, {
        'data_loss': data_loss.item(),
        **{k: v.item() for k, v in physics_losses.items()}
    }


# Example usage and configuration
PHYSICS_CONSTRAINTS_CONFIG = {
    'exponential_decay': {
        'weight': 0.1,
        'description': 'Enforce C(t+1) ≤ C(t) for large t',
        'physics': 'Exponential decay of correlators'
    },
    'positivity': {
        'weight': 0.5,
        'description': 'Enforce C(t) ≥ 0 for all t',
        'physics': 'Correlators are positive definite'
    },
    'energy_ordering': {
        'weight': 0.2,
        'description': 'Enforce E₁ > E₀',
        'physics': 'Excited states have higher energy'
    },
    'spectral_form': {
        'weight': 0.3,
        'description': 'Match two-state exponential form',
        'physics': 'C(t) = A₀e^(-E₀t) + A₁e^(-E₁t)'
    },
    'symmetry': {
        'weight': 0.1,
        'description': 'Enforce C(t) = C(T-t) if applicable',
        'physics': 'Time-reversal symmetry'
    },
    'effective_mass': {
        'weight': 0.2,
        'description': 'Effective mass approaches E₀',
        'physics': 'mₑff(t) = ln(C(t)/C(t+1)) → E₀'
    }
}

if __name__ == "__main__":
    print("Physics-Informed Neural Network Constraints for Lattice QCD")
    print("=" * 60)
    
    for constraint, config in PHYSICS_CONSTRAINTS_CONFIG.items():
        print(f"\n{constraint.upper()}:")
        print(f"  Weight: {config['weight']}")
        print(f"  Rule: {config['description']}")
        print(f"  Physics: {config['physics']}")
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION STATUS:")
    print("  Current: Basic GBR + MLP (minimal physics)")
    print("  Enhanced: All 6 physics constraints above")
    print("  Benefit: True physics-informed learning")
    print("=" * 60)