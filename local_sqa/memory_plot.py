import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


# FONT
FONT_PATH = "/Users/igorzagubien/Library/Fonts/Karla-VariableFont_wght.ttf"
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError("Karla font not found at: " + FONT_PATH)
PROP = fm.FontProperties(fname=FONT_PATH)


# UPB COLORS
UPB_ULTRABLAU   = "#0025AA"
UPB_IRISVIOLETT = "#7E3FA8"
UPB_MEERBLAU    = "#23A9C9"
UPB_GRANATPINK  = "#EF3A84"


COLOR_NOISE     = "#888888"
COLOR_BP        = "#55AA55"

SCRIPT_DIR = Path(__file__).parent

RESULTS_ROOT = SCRIPT_DIR / "results"


CHAPTER_DIRS = [
    "47_122796",
    "47_122796_filter",
    "47_122796_N4",
    "49_121052",
    "49_121052__47_122796",
]

RUNS = [
    ("results_19.csv", "w2v2-large + Transformer", UPB_ULTRABLAU),
    ("results_18.csv", "w2v2-base + Transformer",  UPB_IRISVIOLETT),
    ("results_23.csv", "w2v2-large + BiLSTM",      UPB_MEERBLAU),
    ("results_24.csv", "w2v2-base + BiLSTM",       UPB_GRANATPINK),

    # 27 = wav2vec2 large + conv decoder
    ("results_27.csv", "w2v2-large + Conv",        "#111111"),

    # 26 = wav2vec2 base + conv decoder
    ("results_26.csv", "w2v2-base + Conv",         "#666666"),
]


def load_run_data(csv_dir: Path, filename: str):
    csv_path = csv_dir / filename
    if not csv_path.exists():
        print(f"[WARN] CSV not found, skipping: {csv_path}")
        return None

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # CSV format:
    has_new_format = "mos_clean" in df.columns

    if has_new_format:
        print("Detected NEW FORMAT (clean/noise/bp)")
        df_clean = df[df["success_clean"] == True]
        df_noise = df[df["success_noise"] == True]
        df_bp    = df[df["success_bp"] == True]

        return {
            "mode": "new",
            "df_clean": df_clean,
            "df_noise": df_noise,
            "df_bp": df_bp,
        }

    else:
        print("Detected OLD FORMAT (clean only)")
        if "success" not in df.columns:
            raise ValueError(f"'success' column missing in {csv_path}")

        df_success = df[df["success"] == True].copy()
        df_runtime = df_success.dropna(subset=["runtime_s"]).copy()
        df_memory = df_success.dropna(subset=["memory_mb"]).copy()

        return {
            "mode": "old",
            "df_success": df_success,
            "df_runtime": df_runtime,
            "df_memory": df_memory,
        }


def add_endpoint_label(df, value_col, text, color):
    """Write 'WN' or 'BP' slightly to the right of the last sample of a curve."""
    x = df["seconds"].iloc[-1]
    y = df[value_col].iloc[-1]

    plt.text(
        x + 12, y,
        text,
        fontsize=11,
        color=color,
        fontproperties=PROP,
        va="center",
    )


def plot_one_dir(csv_dir: Path):
    runs_data = []

    for filename, label, color in RUNS:
        data = load_run_data(csv_dir, filename)
        if data is None:
            continue
        data["label"] = label
        data["color"] = color
        data["filename"] = filename
        runs_data.append(data)

    if not runs_data:
        print(f"[WARN] No CSVs loaded in {csv_dir} -> skipping plots")
        return



    plt.figure(figsize=(12, 6))
    for run in runs_data:
        if run["mode"] == "old":
            df = run["df_runtime"]
            if df.empty:
                continue
            plt.plot(
                df["seconds"], df["runtime_s"],
                marker="o", linewidth=2, color=run["color"], label=run["label"]
            )

        else:
            df = run["df_clean"].dropna(subset=["runtime_clean_s"])
            if df.empty:
                continue
            plt.plot(
                df["seconds"], df["runtime_clean_s"],
                marker="o", linewidth=2, color=run["color"], label=run["label"]
            )

    plt.title("Runtime vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length/s", fontproperties=PROP, fontsize=22)
    plt.ylabel("runtime/s", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.legend(prop=PROP, fontsize=18)
    plt.tight_layout()
    plt.savefig(csv_dir / "compare_runtime.png", dpi=300)
    plt.close()



    plt.figure(figsize=(12, 6))
    for run in runs_data:
        if run["mode"] == "old":
            df = run["df_memory"]
            if df.empty:
                continue
            plt.plot(
                df["seconds"], df["memory_mb"] / 1024.0,
                marker="o", linewidth=2, color=run["color"], label=run["label"]
            )
        else:
            df = run["df_clean"].dropna(subset=["memory_clean_mb"])
            if df.empty:
                continue
            plt.plot(
                df["seconds"], df["memory_clean_mb"] / 1024.0,
                marker="o", linewidth=2, color=run["color"], label=run["label"]
            )

    plt.title("Peak Memory vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length/s", fontproperties=PROP, fontsize=22)
    plt.ylabel("memory [GB]", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.legend(prop=PROP, fontsize=18)
    plt.tight_layout()
    plt.savefig(csv_dir / "compare_memory.png", dpi=300)
    plt.close()



    plt.figure(figsize=(12, 6))

    for run in runs_data:
        base = run["color"]

        if run["mode"] == "old":
            df = run["df_success"]
            plt.plot(
                df["seconds"], df["mos"],
                marker="o", linewidth=2, color=base, label=run["label"]
            )
            continue

        dfc = run["df_clean"]
        plt.plot(
            dfc["seconds"], dfc["mos_clean"],
            marker="o", linewidth=2, linestyle="-",
            color=base, label=run["label"] + " (clean)"
        )

        dfn = run["df_noise"]
        if not dfn.empty:
            plt.plot(
                dfn["seconds"], dfn["mos_noise"],
                marker="o", linewidth=2, linestyle="--",
                color=base
            )
            add_endpoint_label(dfn, "mos_noise", "WN", base)

        dfb = run["df_bp"]
        if not dfb.empty:
            plt.plot(
                dfb["seconds"], dfb["mos_bp"],
                marker="o", linewidth=2, linestyle=":",
                color=base
            )
            add_endpoint_label(dfb, "mos_bp", "BP", base)

    plt.title("MOS vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length/s", fontproperties=PROP, fontsize=22)
    plt.ylabel("Predicted MOS", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.legend(prop=PROP, fontsize=16)
    plt.tight_layout()
    plt.savefig(csv_dir / "compare_mos.png", dpi=300)
    plt.close()

    print(f"[OK] wrote plots in: {csv_dir}")


def main():
    for name in CHAPTER_DIRS:
        csv_dir = RESULTS_ROOT / name
        if not csv_dir.exists():
            print(f"[WARN] missing folder, skipping: {csv_dir}")
            continue
        print(f"\n=== processing: {csv_dir} ===")
        plot_one_dir(csv_dir)


if __name__ == "__main__":
    main()
