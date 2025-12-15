# run_experiment.py
import sys
import lattice_qcd_analysis

def run(exp_label):
    print(f"Running experiment: {exp_label}")
    # Pass the experiment label via sys.argv so main() can pick it up
    sys.argv = ["lattice_qcd_analysis.py", exp_label]
    lattice_qcd_analysis.main()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        print("Usage: python run_experiment.py <experiment_label>")
        print("Available experiments:")
        import config
        for exp_id, exp_cfg in config.EXPERIMENTS.items():
            print(f"  {exp_cfg['label']}")
        sys.exit(1)
