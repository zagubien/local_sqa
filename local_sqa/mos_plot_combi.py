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
    ("19", "w2v2-large + Transformer", UPB_ULTRABLAU),
    ("18", "w2v2-base + Transformer",  UPB_IRISVIOLETT),
    ("23", "w2v2-large + BiLSTM",      UPB_MEERBLAU),
    ("24", "w2v2-base + BiLSTM",       UPB_GRANATPINK),

    # 27 = wav2vec2 large + conv decoder
    ("27", "w2v2-large + Conv",        "#111111"),

    # 26 = wav2vec2 base + conv decoder
    ("26", "w2v2-base + Conv",         "#666666"),
]


def pick_latest_version(csv_dir: Path, pattern: str) -> Path | None:
    files = sorted(csv_dir.glob(pattern))
    if not files:
        return None

    def vnum(p: Path) -> int:
        m = re.search(r"_v(\d+)$", p.stem)
        return int(m.group(1)) if m else -1

    files.sort(key=vnum)
    return files[-1]


def resolve_concat_csv(csv_dir: Path, run_id: str) -> Path | None:
    base = csv_dir / f"results_{run_id}.csv"
    if base.exists():
        return base
    return pick_latest_version(csv_dir, f"results_{run_id}_v*.csv")


def resolve_running_mean_csv(csv_dir: Path, run_id: str) -> Path | None:
    base = csv_dir / f"results_running_mean_{run_id}.csv"
    if base.exists():
        return base
    return pick_latest_version(csv_dir, f"results_running_mean_{run_id}_v*.csv")


def load_concat(df_path: Path):
    df = pd.read_csv(df_path)

    if "mos_clean" in df.columns:
        dfc = df[df["success_clean"] == True].copy()
        mos_col = "mos_clean"
        rt_col = "runtime_clean_s"
        mem_col = "memory_clean_mb"
    else:
        if "success" not in df.columns:
            raise ValueError(f"'success' column missing in {df_path}")
        dfc = df[df["success"] == True].copy()
        mos_col = "mos"
        rt_col = "runtime_s"
        mem_col = "memory_mb"

    dfc = dfc.sort_values("seconds").reset_index(drop=True)

    df_rt = dfc.dropna(subset=[rt_col]).copy() if rt_col in dfc.columns else dfc.iloc[0:0]
    df_mem = dfc.dropna(subset=[mem_col]).copy() if mem_col in dfc.columns else dfc.iloc[0:0]

    return dfc, mos_col, df_rt, rt_col, df_mem, mem_col


def load_running_mean(df_path: Path):
    df = pd.read_csv(df_path)

    mos_candidates = [
        "mos_running_mean",
        "mos_pred_rm_durw",
        "mos_rm",
        "mos",
    ]
    mos_col = None
    for c in mos_candidates:
        if c in df.columns:
            mos_col = c
            break

    if "elapsed_s" not in df.columns or mos_col is None:
        raise ValueError(
            f"{df_path} missing required columns. "
            f"Need 'elapsed_s' and one of {mos_candidates}, got {list(df.columns)}"
        )

    df = df.sort_values("elapsed_s").reset_index(drop=True)
    return df, mos_col


def plot_one_dir(csv_dir: Path):
    if not csv_dir.exists():
        print(f"[WARN] CSV_DIR does not exist: {csv_dir}")
        return

    runs_data = []
    for run_id, label, color in RUNS:
        concat_path = resolve_concat_csv(csv_dir, run_id)
        rm_path = resolve_running_mean_csv(csv_dir, run_id)

        if concat_path is None and rm_path is None:
            print(f"[WARN] no CSVs found for run_id={run_id} in {csv_dir.name}")
            continue

        runs_data.append({
            "id": run_id,
            "label": label,
            "color": color,
            "concat_path": concat_path,
            "rm_path": rm_path,
        })

    if not runs_data:
        print(f"[WARN] No runs found in {csv_dir} -> skipping")
        return

    # runtime (concat only)
    plt.figure(figsize=(12, 6))
    for run in runs_data:
        if run["concat_path"] is None:
            continue
        dfc, mos_col, df_rt, rt_col, df_mem, mem_col = load_concat(run["concat_path"])
        if df_rt.empty or rt_col not in df_rt.columns:
            continue
        plt.plot(df_rt["seconds"], df_rt[rt_col], marker="o", linewidth=2, color=run["color"], label=run["label"])

    plt.title("Runtime vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length [s]", fontproperties=PROP, fontsize=22)
    plt.ylabel("runtime [s]", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.legend(prop=PROP, fontsize=18)
    plt.tight_layout()
    plt.savefig(csv_dir / "compare_runtime.png", dpi=300)
    plt.close()

    # memory (concat only)
    plt.figure(figsize=(12, 6))
    for run in runs_data:
        if run["concat_path"] is None:
            continue
        dfc, mos_col, df_rt, rt_col, df_mem, mem_col = load_concat(run["concat_path"])
        if df_mem.empty or mem_col not in df_mem.columns:
            continue
        plt.plot(df_mem["seconds"], df_mem[mem_col], marker="o", linewidth=2, color=run["color"], label=run["label"])

    plt.title("Peak Memory vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length [s]", fontproperties=PROP, fontsize=22)
    plt.ylabel("memory [MB]", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.legend(prop=PROP, fontsize=18)
    plt.tight_layout()
    plt.savefig(csv_dir / "compare_memory.png", dpi=300)
    plt.close()

    # MOS (concat vs running mean)
    plt.figure(figsize=(12, 6))
    for run in runs_data:
        color = run["color"]

    
        if run["concat_path"] is not None:
            dfc, mos_col, *_ = load_concat(run["concat_path"])
            if not dfc.empty and mos_col in dfc.columns:
                plt.plot(
                    dfc["seconds"], dfc[mos_col],
                    marker="o", markersize=6,
                    linewidth=2.5, linestyle="-",
                    color=color, label=run["label"]
                )


        if run["rm_path"] is not None:
            dfr, rm_mos_col = load_running_mean(run["rm_path"])
            if not dfr.empty:
                plt.plot(
                    dfr["elapsed_s"], dfr[rm_mos_col],
                    linewidth=1.2, linestyle="-",
                    color=color, alpha=0.9
                )

                x = dfr["elapsed_s"].iloc[-1]
                y = dfr[rm_mos_col].iloc[-1]
                plt.text(
                    x + 12, y, "RM",
                    fontsize=11,
                    color=color,
                    fontproperties=PROP,
                    va="center",
                )

    plt.title("MOS vs. Audio Length", fontproperties=PROP, fontsize=26)
    plt.xlabel("audio length [s]", fontproperties=PROP, fontsize=22)
    plt.ylabel("Predicted MOS", fontproperties=PROP, fontsize=22)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)
    plt.legend(prop=PROP, fontsize=16)
    plt.tight_layout()
    plt.savefig(csv_dir / "compare_mos_concat_vs_running_mean.png", dpi=300)
    plt.close()

    print(f"[OK] wrote plots in: {csv_dir}")


def main():
    for name in CHAPTER_DIRS:
        csv_dir = RESULTS_ROOT / name
        print(f"\n=== processing: {csv_dir} ===")
        plot_one_dir(csv_dir)


if __name__ == "__main__":
    main()
