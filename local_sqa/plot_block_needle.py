from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


DPI = 300

FONT_PATH = str(Path.home() / "Library" / "Fonts" / "Karla-VariableFont_wght.ttf")
PROP = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

COL_CONCAT = "#EF3A84"
COL_RM     = "#0025AA"
COL_TGT    = "#111111"

CSV_GLOBAL_CONCAT = "global_mos_concat.csv"
CSV_GLOBAL_RM     = "global_mos_running_mean.csv"
CSV_BLOCK_CONCAT  = "block_mos_concat.csv"
CSV_BLOCK_RM      = "block_mos_running_mean.csv"

# visible step "towers" without fill
STEP_ALPHA_CONCAT = 0.85
STEP_ALPHA_RM     = 0.80
STEP_W_CONCAT     = 2.6
STEP_W_RM         = 2.2

MODES = {"forward", "reverse", "random"}


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


def find_results_tts_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for cand in [
        script_dir / "results_tts",
        script_dir.parent / "results_tts",
        Path.cwd() / "results_tts",
    ]:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError("could not find results_tts next to this script / parent / cwd")


def find_block_roots(results_tts: Path) -> list[Path]:
    # supports:
    #   results_tts/block_5
    #   results_tts/block_20
    # (and also results_tts/block for backwards compatibility)
    roots = []
    for p in sorted(results_tts.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if p.name == "block" or p.name.startswith("block_"):
            roots.append(p)
    if not roots:
        raise FileNotFoundError(f"no block roots found under {results_tts} (expected block_5/block_20)")
    return roots


def iter_leaf_dirs(block_root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    """
    layout:
      results_tts/block_X/<dataset>/<system>/<mode>/*.csv
    """
    for dataset_dir in sorted([p for p in block_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if dataset_dir.name not in {"bvcc", "somos"}:
            continue
        for system_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            for mode_dir in sorted([p for p in system_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                if mode_dir.name not in MODES:
                    continue
                has_global = (mode_dir / CSV_GLOBAL_CONCAT).exists() and (mode_dir / CSV_GLOBAL_RM).exists()
                has_block  = (mode_dir / CSV_BLOCK_CONCAT).exists() and (mode_dir / CSV_BLOCK_RM).exists()
                if has_global or has_block:
                    yield dataset_dir.name, system_dir.name, mode_dir.name, mode_dir


def _ensure_mid_start_end(df: pd.DataFrame, what: str) -> tuple[pd.DataFrame, str, str, str]:
    # accept either explicit block_mid_s or compute it from start/end
    if "block_start_s" in df.columns and "block_end_s" in df.columns:
        s_col, e_col = "block_start_s", "block_end_s"
    else:
        s_col = pick_first_existing(df, ["block_start_s", "start_s", "start"], f"{what} start")
        e_col = pick_first_existing(df, ["block_end_s", "end_s", "end"], f"{what} end")

    if "block_mid_s" not in df.columns:
        df = df.copy()
        df["block_mid_s"] = 0.5 * (df[s_col].astype(float) + df[e_col].astype(float))
    return df, s_col, e_col, "block_mid_s"


def _stairs_from_blocks(df: pd.DataFrame, start_col: str, end_col: str, y_col: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for _, r in df.iterrows():
        s = float(r[start_col])
        e = float(r[end_col])
        y = float(r[y_col])
        xs.extend([s, e])
        ys.extend([y, y])
    return xs, ys


def plot_global(csv_dir: Path, dataset: str, system: str, mode: str, block_tag: str) -> Path:
    df_gc = pd.read_csv(csv_dir / CSV_GLOBAL_CONCAT)
    df_gr = pd.read_csv(csv_dir / CSV_GLOBAL_RM)

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
        zorder=5,
    )

    ax.plot(
        df_gr[x_gr], df_gr[y_gr],
        color=COL_RM,
        linewidth=1.8,
        linestyle="--",
        label="running mean (predicted, global)",
        zorder=5,
    )

    ax.plot(
        df_gc[x_gc], df_gc[tgt_gc],
        color=COL_TGT,
        linewidth=1.8,
        linestyle="-",
        alpha=0.85,
        label=tgt_prefix_label,
        zorder=6,
    )

    ax.plot(
        df_gr[x_gr], df_gr[tgt_gr],
        color=COL_TGT,
        linewidth=1.2,
        linestyle="--",
        alpha=0.55,
        label=tgt_rm_label,
        zorder=6,
    )

    ax.set_title(
        f"Real-needle TTS ({block_tag}) — {dataset.upper()} — global MOS: concat vs running mean\n{system}/{mode}",
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


def plot_block(csv_dir: Path, dataset: str, system: str, mode: str, block_tag: str) -> Path:
    df_bc = pd.read_csv(csv_dir / CSV_BLOCK_CONCAT)
    df_br = pd.read_csv(csv_dir / CSV_BLOCK_RM)

    df_bc, s_bc, e_bc, x_bc = _ensure_mid_start_end(df_bc, "block concat")
    df_br, s_br, e_br, x_br = _ensure_mid_start_end(df_br, "block rm")

    y_bc = pick_first_existing(df_bc, ["mos_pred_block_concat", "mos_pred_block", "mos_pred", "mos"], "block concat pred")
    tgt_bc = pick_first_existing(df_bc, ["mos_target_block_durw", "mos_target_block", "mos_target"], "block target (concat)")

    y_br = pick_first_existing(df_br, ["mos_pred_block_rm_durw", "mos_pred_block_rm", "mos_pred_block", "mos_pred", "mos"], "block rm pred")
    tgt_br = pick_first_existing(df_br, ["mos_target_block_durw", "mos_target_block", "mos_target"], "block target (rm)")

    df_bc = df_bc.sort_values(x_bc).reset_index(drop=True)
    df_br = df_br.sort_values(x_br).reset_index(drop=True)

    tgt_prefix_label = "target MOS (block/prefix)"
    tgt_rm_label = "target MOS (block/running)"
    tgt_prefix_label += " (dur-weighted)" if is_dur_weighted(tgt_bc) else " (unweighted)"
    tgt_rm_label     += " (dur-weighted)" if is_dur_weighted(tgt_br) else " (unweighted)"

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # towers (step outlines)
    xs_c, ys_c = _stairs_from_blocks(df_bc, s_bc, e_bc, y_bc)
    xs_r, ys_r = _stairs_from_blocks(df_br, s_br, e_br, y_br)
    ax.plot(xs_c, ys_c, color=COL_CONCAT, linewidth=STEP_W_CONCAT, alpha=STEP_ALPHA_CONCAT, zorder=2)
    ax.plot(xs_r, ys_r, color=COL_RM, linewidth=STEP_W_RM, alpha=STEP_ALPHA_RM, linestyle="--", zorder=2)

    # main curves
    ax.plot(
        df_bc[x_bc], df_bc[y_bc],
        color=COL_CONCAT,
        linewidth=3.0,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2.0,
        label="concat (predicted, block)",
        zorder=5,
    )
    ax.plot(
        df_br[x_br], df_br[y_br],
        color=COL_RM,
        linewidth=1.8,
        linestyle="--",
        label="running mean (predicted, block)",
        zorder=5,
    )

    ax.plot(
        df_bc[x_bc], df_bc[tgt_bc],
        color=COL_TGT,
        linewidth=1.8,
        linestyle="-",
        alpha=0.85,
        label=tgt_prefix_label,
        zorder=6,
    )
    ax.plot(
        df_br[x_br], df_br[tgt_br],
        color=COL_TGT,
        linewidth=1.2,
        linestyle="--",
        alpha=0.55,
        label=tgt_rm_label,
        zorder=6,
    )

    ax.set_title(
        f"Real-needle TTS ({block_tag}) — {dataset.upper()} — block MOS: concat vs running mean\n{system}/{mode}",
        fontsize=18,
        **fkw(),
    )
    ax.set_xlabel("block time (midpoint) [s]", fontsize=20, **fkw())
    ax.set_ylabel("MOS", fontsize=20, **fkw())
    pretty_axes(ax)

    leg = ax.legend(loc="upper right", fontsize=12, frameon=True)
    leg.get_frame().set_alpha(0.95)

    fig.tight_layout()
    out_path = csv_dir / "plot_block_concat_vs_running_mean.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def main():
    results_tts = find_results_tts_root()
    block_roots = find_block_roots(results_tts)

    ok, fail = 0, 0

    for block_root in block_roots:
        block_tag = block_root.name  # block_5 / block_20 / block
        for dataset, system, mode, d in iter_leaf_dirs(block_root):
            try:
                wrote = False

                if (d / CSV_GLOBAL_CONCAT).exists() and (d / CSV_GLOBAL_RM).exists():
                    out = plot_global(d, dataset, system, mode, block_tag)
                    print(f"[OK]  {block_tag}/{dataset}/{system}/{mode} -> {out.name}")
                    ok += 1
                    wrote = True

                if (d / CSV_BLOCK_CONCAT).exists() and (d / CSV_BLOCK_RM).exists():
                    out = plot_block(d, dataset, system, mode, block_tag)
                    print(f"[OK]  {block_tag}/{dataset}/{system}/{mode} -> {out.name}")
                    ok += 1
                    wrote = True

                if not wrote:
                    print(f"[SKIP] {block_tag}/{dataset}/{system}/{mode} (missing csvs)")
            except Exception as e:
                print(f"[FAIL] {block_tag}/{dataset}/{system}/{mode}: {e}")
                fail += 1

    print(f"\nDONE: ok={ok} fail={fail} roots={[str(p) for p in block_roots]}")


if __name__ == "__main__":
    main()