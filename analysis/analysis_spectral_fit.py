# analysis_spectral_fit.py  (simple: only plot fitted curves)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT_DIR / "data" / "predictions"
PLOT_DIR = ROOT_DIR / "plots" / "spectral_fits"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def fit_model(t, a0, a1, dE0, dE1):
    t = np.asarray(t, dtype=float)
    return a0 * np.exp(-dE0 * t) + a1 * np.exp(-dE1 * t)


def load_fit_params(path: Path):
    """
    Parse one 'SPECTRAL FIT PARAMETERS' file and return
    (params_truth, params_gbr, params_mlp) as dicts.

    Robust to lines like:
    TRUTH  0.1013 (0.0020)  9.0827 (20.8260)  0.4724 (0.0026)  2.5276 (1.3662)  1.205  0.192
    TRUTH  0.1000 (0.0022)  9.1410(41.5563)   0.3892(0.0264)   2.6188(2.7131)   2.190  0.000
    """
    params = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(("TRUTH", "GBR", "MLP")):
                # method name is the first word
                method = line.split()[0]

                # extract all numbers on the line (integers or decimals)
                # e.g. ['0.1013', '0.0020', '9.0827', '20.8260', '0.4724', ...]
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                if len(nums) < 4:
                    raise ValueError(
                        f"Could not find 4 parameters in line:\n{line}\n"
                        f"Extracted numbers: {nums}"
                    )

                # first four numbers are a0, a1, dE0, dE1
                a0, a1, dE0, dE1 = map(float, nums[:4])
                params[method] = dict(a0=a0, a1=a1, dE0=dE0, dE1=dE1)

    # Expect all three entries to be present
    return params["TRUTH"], params["GBR"], params["MLP"]


def plot_channel(stem, title, nt, outfile):
    """
    stem   : base filename (without extension) in data/predictions
    title  : LaTeX title for the plot
    nt     : number of time slices to plot (0 .. nt-1)
    outfile: PDF path in plots/spectral_fits
    """
    # accept either .csv or .txt
    cand_csv = PRED_DIR / f"{stem}.csv"
    cand_txt = PRED_DIR / f"{stem}.txt"
    if cand_csv.exists():
        path = cand_csv
    elif cand_txt.exists():
        path = cand_txt
    else:
        print(f"WARNING: no parameter file found for {stem}")
        return

    p_truth, p_gbr, p_mlp = load_fit_params(path)

    t = np.arange(nt)

    C_truth = fit_model(t, **p_truth)
    C_gbr   = fit_model(t, **p_gbr)
    C_mlp   = fit_model(t, **p_mlp)

    plt.figure(figsize=(7, 5))
    plt.yscale("log")

    plt.plot(t, C_truth, "o-", color="black",      label="TRUTH fit")
    plt.plot(t, C_gbr,   "^-", color="tab:green",  label="GBR fit")
    plt.plot(t, C_mlp,   "s-", color="tab:orange", label="MLP fit")

    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t)$")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"Saved {outfile}")


def main():
    channels = [
        # stem,                          title,                                               nt
        ("K_ll_to_qsq0",                 r"$K$ two-point, $q^{2}=0$",                         25),
        ("K_ll_to_2qsqmaxby3",           r"$K$ two-point, $q^{2}=2q_{\mathrm{max}}^{2}/3$",   25),
        ("localscalar_T16_to_qsq0",      r"Local scalar three-point, $T=16$, $q^{2}=0$",      17),
        ("localscalar_T19_to_qsqmaxby3", r"Local scalar three-point, $T=19$, $q^{2}=2q_{\max}^2/3$", 20),
        ("localscalar_T22_to_2qsqmaxby3",r"Local scalar three-point, $T=22$, $q^{2}=2q_{\max}^2/3$", 23),
        ("localtempvector_T16_to_qsq0",  r"Local temporal vector three-point, $T=16$, $q^{2}=0$",    17),
        ("localtempvector_T22_to_2qsqmaxby3",
                                         r"Local temporal vector three-point, $T=22$, $q^{2}=2q_{\max}^2/3$", 23),
    ]

    for stem, title, nt in channels:
        outfile = PLOT_DIR / f"spectral_fit_overlay_{stem}.pdf"
        plot_channel(stem, title, nt, outfile)


if __name__ == "__main__":
    main()
