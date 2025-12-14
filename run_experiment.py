# run_experiment.py
import sys
import importlib
import lattice_qcd_analysis

def run(exp_label):
    # dynamically update the config
    import config
    config.EXPERIMENT = exp_label

    print(f"Running experiment: {exp_label}")
    lattice_qcd_analysis.main()

if __name__ == "__main__":
    run(sys.argv[1])
