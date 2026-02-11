import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


FONT_PATH = "/Users/igorzagubien/Library/Fonts/Karla-VariableFont_wght.ttf"
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError("Karla font not found at: " + FONT_PATH)
PROP = fm.FontProperties(fname=FONT_PATH)

UPB_ULTRABLAU   = "#0025AA"
UPB_IRISVIOLETT = "#7E3FA8"
UPB_MEERBLAU    = "#23A9C9"
UPB_GRANATPINK  = "#EF3A84"

RUN26_COLOR = "#111111"
RUN27_COLOR = "#666666"

SCRIPT_DIR = Path(__file__).parent
RESULTS_ROOT = SCRIPT_DIR / "results"

LABELS = {
    "18": ("w2v2-base + Transformer", UPB_IRISVIOLETT),
    "19": ("w2v2-large + Transformer", UPB_ULTRABLAU),
    "23": ("w2v2-large + BiLSTM", UPB_MEERBLAU),
    "24": ("w2v2-base + BiLSTM", UPB_GRANATPINK),
    "26": ("WavLM-base + BiLSTM", RUN26_COLOR),
    "27": ("WavLM-large + BiLSTM", RUN27_COLOR),
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"elapsed_s", "mos_running_mean"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values("elapsed_s").reset_index(drop=True)


def extract_run_id(path: Path) -> str | None:
    # supports: results_running_mean_27.csv, results_running_mean_27_v1.csv
    m = re.search(r"results_running_mean_(\d+)(?:_v\d+)?$", path.stem)
    if not m:
        return None
    return m.group(1)


def plot_folder(folder: Path, csv_paths: list[Path]) -> None:
    # unique per run_id, keep latest (lexicographically) if duplicates exist
    by_run = {}
    for p in csv_paths:
        run_id = extract_run_id(p)
        if run_id is None:
            continue
        by_run[run_id] = max(by_run.get(run_id, p), p)

    if not by_run:
        return

    plt.figure(figsize=(12, 6))

    for run_id in sorted(by_run.keys(), key=lambda x: int(x)):
        p = by_run[run_id]
        label, color = LABELS.get(run_id, (p.stem, "#000000"))
        df = load_csv(p)
        plt.plot(
            df["elapsed_s"],
            df["mos_running_mean"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

    plt.title("MOS vs. Audio Length (running mean over sentences)", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length/s", fontproperties=PROP, fontsize=22)
    plt.ylabel("Predicted MOS (running mean)", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.xticks(fontproperties=PROP, fontsize=16)
    plt.yticks(fontproperties=PROP, fontsize=16)
    plt.legend(prop=PROP, fontsize=16)
    plt.tight_layout()

    out_path = folder / "compare_mos_running_mean.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] {folder.name} -> {out_path.name}")


def main():
    if not RESULTS_ROOT.exists():
        raise RuntimeError(f"RESULTS_ROOT does not exist: {RESULTS_ROOT}")

    all_csv = sorted(RESULTS_ROOT.rglob("results_running_mean_*.csv"))
    if not all_csv:
        raise RuntimeError(f"No running-mean CSVs found under: {RESULTS_ROOT}")


    groups = {}
    for p in all_csv:
        groups.setdefault(p.parent, []).append(p)

    
    wrote = 0
    for folder in sorted(groups.keys()):
        try:
            plot_folder(folder, groups[folder])
            wrote += 1
        except Exception as e:
            print(f"[WARN] skipping {folder}: {e}")

    if wrote == 0:
        raise RuntimeError("Found running-mean CSVs but no plots were written (format mismatch?).")


if __name__ == "__main__":
    main()
