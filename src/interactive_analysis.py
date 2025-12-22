#!/usr/bin/env python3
"""
Interactive MLQCD-SV Analysis Tool

This script provides a menu-driven interface to run analysis on individual datasets.
You can select from all available experiments and choose which models to run.
"""

import sys
import os
from pathlib import Path
import config
import lattice_qcd_analysis

def display_header():
    """Display the application header."""
    print("=" * 80)
    print("MLQCD-SV: Interactive Analysis Tool")
    print("=" * 80)
    print("Physics-Informed Machine Learning for Lattice QCD Correlators")
    print()

def get_available_experiments():
    """Get all available experiments from config."""
    experiments = []
    for exp_id, exp_config in config.EXPERIMENTS.items():
        experiments.append({
            'id': exp_id,
            'label': exp_config['label'],
            'type': exp_config['type'],
            'input_file': Path(exp_config['input_file']).name,
            'target_file': Path(exp_config['target_file']).name
        })
    
    # Sort by experiment type and then by label
    experiments.sort(key=lambda x: (x['type'], x['label']))
    return experiments

def display_experiments_menu(experiments):
    """Display the experiments menu."""
    print("Available Experiments:")
    print("-" * 80)
    
    current_type = None
    for i, exp in enumerate(experiments, 1):
        # Group by experiment type
        if exp['type'] != current_type:
            if current_type is not None:
                print()
            print(f"  {exp['type'].upper()}-POINT CORRELATORS:")
            current_type = exp['type']
        
        print(f"  [{i:2d}] {exp['label']:<35} ({exp['input_file']} → {exp['target_file']})")
    
    print()
    print(f"  [ 0] Exit")
    print("-" * 80)

def get_available_models():
    """Get available ML models."""
    models = []
    
    # Always available models
    base_models = ["GBR", "MLP", "RIDGE", "DTREE"]
    models.extend(base_models)
    
    # Check if PyTorch is available by trying to import
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
    
    # PyTorch-dependent models
    if torch_available:
        models.extend(["CNN", "TRANSFORMER"])
    
    # GBR+PINNS if enabled (simplified version doesn't need PyTorch)
    if getattr(config, 'ENABLE_GBR_PINNS', False):
        models.append("GBR_PINNS")
    
    return models

def display_models_menu(models):
    """Display the models selection menu."""
    print("\nAvailable ML Models:")
    print("-" * 50)
    
    for i, model in enumerate(models, 1):
        description = {
            "GBR": "Gradient Boosting Regressor",
            "MLP": "Multi-Layer Perceptron", 
            "RIDGE": "Ridge Regression",
            "DTREE": "Decision Tree",
            "CNN": "Convolutional Neural Network",
            "TRANSFORMER": "Transformer Model",
            "GBR_PINNS": "GBR + Physics-Informed Neural Networks"
        }.get(model, model)
        
        print(f"  [{i}] {model:<12} - {description}")
    
    print(f"  [A] ALL MODELS   - Run all available models")
    print(f"  [Q] QUICK TEST   - Run GBR only (fastest)")
    print("-" * 50)

def select_experiment(experiments):
    """Let user select an experiment."""
    while True:
        try:
            choice = input("\nSelect experiment (number or 0 to exit): ").strip()
            
            if choice == '0':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(experiments):
                return experiments[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(experiments)}, or 0 to exit.")
                
        except ValueError:
            print("Please enter a valid number.")

def select_models(available_models):
    """Let user select which models to run."""
    while True:
        choice = input("\nSelect models (number, A for all, Q for quick, or comma-separated): ").strip().upper()
        
        if choice == 'Q':
            return ["GBR"]
        elif choice == 'A':
            return available_models
        elif choice.isdigit():
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    return [available_models[choice_num - 1]]
                else:
                    print(f"Please enter a number between 1 and {len(available_models)}.")
                    continue
            except ValueError:
                pass
        else:
            # Handle comma-separated choices
            try:
                choices = [c.strip() for c in choice.split(',')]
                selected_models = []
                
                for c in choices:
                    if c.isdigit():
                        choice_num = int(c)
                        if 1 <= choice_num <= len(available_models):
                            selected_models.append(available_models[choice_num - 1])
                        else:
                            print(f"Invalid choice: {c}")
                            break
                    else:
                        print(f"Invalid choice: {c}")
                        break
                else:
                    if selected_models:
                        return selected_models
                
            except ValueError:
                pass
        
        print("Invalid selection. Please try again.")

def confirm_selection(experiment, models):
    """Confirm the user's selection."""
    print("\n" + "=" * 60)
    print("ANALYSIS CONFIGURATION")
    print("=" * 60)
    print(f"Experiment: {experiment['label']}")
    print(f"Type:       {experiment['type']}-point correlator")
    print(f"Input:      {experiment['input_file']}")
    print(f"Target:     {experiment['target_file']}")
    print(f"Models:     {', '.join(models)}")
    print("=" * 60)
    
    while True:
        confirm = input("\nProceed with this analysis? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return True
        elif confirm in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")

def run_analysis(experiment, models):
    """Run the analysis with selected parameters."""
    print(f"\nStarting analysis for: {experiment['label']}")
    print(f"Models: {', '.join(models)}")
    print("-" * 60)
    
    # Temporarily update config with selected models
    original_models = config.RUN_MODELS
    config.RUN_MODELS = models
    
    try:
        # Set up sys.argv for the analysis
        sys.argv = ["interactive_analysis.py", experiment['label']]
        
        # Run the analysis
        lattice_qcd_analysis.main()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: results/batch/")
        print("Check the output directory for:")
        print("  - spectral_fit_parameters.txt")
        print("  - summary_plots_*.pdf")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Analysis failed with error: {e}")
        print("Check the error message above for details.")
        
    finally:
        # Restore original config
        config.RUN_MODELS = original_models

def main():
    """Main interactive loop."""
    display_header()
    
    # Get available experiments and models
    experiments = get_available_experiments()
    available_models = get_available_models()
    
    if not experiments:
        print("ERROR: No experiments found in config.py")
        return
    
    print(f"Found {len(experiments)} experiments and {len(available_models)} ML models.")
    
    while True:
        print("\n")
        display_experiments_menu(experiments)
        
        # Select experiment
        selected_experiment = select_experiment(experiments)
        if selected_experiment is None:
            print("\nGoodbye!")
            break
        
        # Display and select models
        display_models_menu(available_models)
        selected_models = select_models(available_models)
        
        # Confirm selection
        if confirm_selection(selected_experiment, selected_models):
            run_analysis(selected_experiment, selected_models)
            
            # Ask if user wants to run another analysis
            while True:
                another = input("\nRun another analysis? (y/n): ").strip().lower()
                if another in ['y', 'yes']:
                    break
                elif another in ['n', 'no']:
                    print("\nGoodbye!")
                    return
                else:
                    print("Please enter 'y' or 'n'.")
        else:
            print("Analysis cancelled.")

if __name__ == "__main__":
    main()