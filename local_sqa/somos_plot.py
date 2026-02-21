import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


DPI = 300

FONT_PATH = "/Users/igorzagubien/Library/Fonts/Karla-VariableFont_wght.ttf"
PROP = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

COL_CONCAT = "#EF3A84"
COL_RM     = "#0025AA"
COL_TGT    = "#111111"

CSV_CONCAT_NAME = "global_mos_concat.csv"
CSV_RM_NAME     = "global_mos_running_mean.csv"

SKIP_DIR_NAMES = {"packets"}
SKIP_PREFIXES = ("_",)

MODEL_INFO_MAP = {
    "SOMOS_24": "base + BiLSTM (24)",
    "SOMOS_27": "large + Conv (27)",
}


def fkw():
    return {"fontproperties": PROP} if PROP else {}


def pick_first_existing(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing column for {what}. have={list(df.columns)} candidates={candidates}")


def pretty_axes(ax):
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def is_dur_weighted(col_name: str) -> bool:
    s = col_name.lower()
    return ("durw" in s) or ("dur_w" in s) or ("dur-weight" in s) or ("durweight" in s)


def _is_debug_path(p: Path) -> bool:
    return "_debug" in p.parts


def model_info_from_path(p: Path) -> str:
    for part in p.parts:
        if part in MODEL_INFO_MAP:
            return MODEL_INFO_MAP[part]
        if re.fullmatch(r"SOMOS_\d+", part):
            return part
    return "unknown"


def find_somos_roots() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    if not results_dir.exists():
        return []

    roots = [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("SOMOS_")]
    roots = sorted(roots, key=lambda p: int(re.findall(r"\d+", p.name)[0]) if re.findall(r"\d+", p.name) else 10**9)
    return roots


def iter_result_dirs(root: Path):
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if d.name in SKIP_DIR_NAMES:
            continue
        if d.name.startswith(SKIP_PREFIXES):
            continue
        if _is_debug_path(d):
            continue

        c1 = d / CSV_CONCAT_NAME
        c2 = d / CSV_RM_NAME
        if c1.exists() and c2.exists():
            yield d


def plot_one(csv_dir: Path, root: Path, model_info: str) -> Path:
    c_global_concat = csv_dir / CSV_CONCAT_NAME
    c_global_rm     = csv_dir / CSV_RM_NAME

    df_gc = pd.read_csv(c_global_concat)
    df_gr = pd.read_csv(c_global_rm)

    x_gc = pick_first_existing(df_gc, ["seconds", "elapsed_s", "time_s"], "global concat x-axis")
    y_gc = pick_first_existing(df_gc, ["mos_pred_concat", "mos_pred", "mos"], "global concat predicted MOS")
    tgt_gc = pick_first_existing(
        df_gc,
        ["mos_target_concat_durw", "mos_target_concat_mean", "mos_target_concat_avg", "mos_target_concat", "mos_target"],
        "global concat target MOS",
    )

    x_gr = pick_first_existing(df_gr, ["elapsed_s", "seconds", "time_s"], "global RM x-axis")
    y_gr = pick_first_existing(
        df_gr,
        ["mos_pred_rm_durw", "mos_pred_rm_mean", "mos_pred_rm", "mos_running_mean", "mos_pred", "mos"],
        "global RM predicted MOS",
    )
    tgt_gr = pick_first_existing(
        df_gr,
        ["mos_target_rm_durw", "mos_target_rm_mean", "mos_target_rm", "mos_target"],
        "global RM target MOS",
    )

    df_gc = df_gc.sort_values(x_gc).reset_index(drop=True)
    df_gr = df_gr.sort_values(x_gr).reset_index(drop=True)

    tgt_prefix_label = "target MOS (prefix)"
    tgt_rm_label = "target MOS (running)"
    tgt_prefix_label += " (dur-weighted)" if is_dur_weighted(tgt_gc) else " (unweighted)"
    tgt_rm_label     += " (dur-weighted)" if is_dur_weighted(tgt_gr) else " (unweighted)"

    try:
        rel = str(csv_dir.relative_to(root))
    except Exception:
        rel = str(csv_dir)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.plot(
        df_gc[x_gc], df_gc[y_gc],
        color=COL_CONCAT,
        linewidth=3.0,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2.0,
        label="concat (predicted, global)",
    )

    ax.plot(
        df_gr[x_gr], df_gr[y_gr],
        color=COL_RM,
        linewidth=1.8,
        linestyle="--",
        label="running mean (predicted, global)",
    )

    ax.plot(
        df_gc[x_gc], df_gc[tgt_gc],
        color=COL_TGT,
        linewidth=1.8,
        linestyle="-",
        alpha=0.85,
        label=tgt_prefix_label,
    )

    ax.plot(
        df_gr[x_gr], df_gr[tgt_gr],
        color=COL_TGT,
        linewidth=1.2,
        linestyle="--",
        alpha=0.55,
        label=tgt_rm_label,
    )

    ax.set_title(
        f"SOMOS ({model_info}) — global MOS: concat vs running mean\n{rel}",
        fontsize=18,
        **fkw(),
    )
    ax.set_xlabel("audio length [s]", fontsize=20, **fkw())
    ax.set_ylabel("MOS", fontsize=20, **fkw())
    pretty_axes(ax)

    leg = ax.legend(loc="upper right", fontsize=12, frameon=True)
    leg.get_frame().set_alpha(0.95)

    fig.tight_layout()
    out_path = csv_dir / "plot_global_concat_vs_running_mean.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    return out_path


def main():
    roots = find_somos_roots()
    if not roots:
        raise FileNotFoundError("could not find results/SOMOS_* next to this script")

    ok, fail = 0, 0

    for root in roots:
        dirs = sorted(set(iter_result_dirs(root)), key=lambda p: str(p))
        if not dirs:
            print(f"[WARN] no result dirs found under {root}")
            continue

        for d in dirs:
            try:
                model_info = model_info_from_path(d)
                out = plot_one(d, root, model_info)
                print(f"[OK]  {d.relative_to(root)} -> {out.name}")
                ok += 1
            except Exception as e:
                try:
                    rel = d.relative_to(root)
                except Exception:
                    rel = d
                print(f"[FAIL] {rel}: {e}")
                fail += 1

    print(f"\nDONE: ok={ok} fail={fail} roots={[str(r) for r in roots]}")


if __name__ == "__main__":
    main()
