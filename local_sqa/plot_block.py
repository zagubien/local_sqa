from __future__ import annotations

import os
from pathlib import Path

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

SKIP_DIR_NAMES = {"packets"}
SKIP_PREFIXES = ("_",)

# block "towers": step edges (no fill), more visible but not ugly
STEP_ALPHA_CONCAT = 0.80
STEP_ALPHA_RM     = 0.75
STEP_W_CONCAT     = 2.4
STEP_W_RM         = 2.1

# optional faint block boundaries (usually keep off)
DRAW_BLOCK_BOUNDARIES = False
BOUNDARY_ALPHA = 0.06
BOUNDARY_W = 0.7


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


def find_results_block_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for cand in [
        script_dir / "results_block",
        script_dir.parent / "results_block",
        Path.cwd() / "results_block",
    ]:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError("could not find results_block next to this script / parent / cwd")


def _iter_run_roots(results_block: Path, dataset: str) -> list[Path]:
    """
    new layout:
      results_block/<dataset>/<train_id>/<run_dir>/
    where <run_dir> is e.g. bvcc_ckpt_best_SRCC_t180_s0_block20
    """
    ds_root = results_block / dataset
    if not ds_root.exists():
        return []

    run_roots: list[Path] = []
    for train_dir in sorted([p for p in ds_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for run_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            # be strict-ish: only take dirs that look like our runs
            if dataset == "bvcc" and not run_dir.name.startswith("bvcc_"):
                continue
            if dataset == "somos" and not run_dir.name.startswith("somos_"):
                continue
            run_roots.append(run_dir)
    return run_roots


def iter_variant_dirs(root: Path):
    """
    root = .../<run_dir>
    children are sys*/rev_sys*/rand_sys*
    each has variant subdirs (original/no_pause/long_pause/long_pause_noise)
    """
    for sys_dir in root.iterdir():
        if not sys_dir.is_dir():
            continue
        if sys_dir.name in SKIP_DIR_NAMES:
            continue
        if sys_dir.name.startswith(SKIP_PREFIXES):
            continue
        if _is_debug_path(sys_dir):
            continue

        if not (sys_dir.name.startswith("sys") or sys_dir.name.startswith("rev_sys") or sys_dir.name.startswith("rand_sys")):
            continue

        for var_dir in sys_dir.iterdir():
            if not var_dir.is_dir():
                continue
            if var_dir.name in SKIP_DIR_NAMES:
                continue
            if var_dir.name.startswith(SKIP_PREFIXES):
                continue
            if _is_debug_path(var_dir):
                continue

            has_global = (var_dir / CSV_GLOBAL_CONCAT).exists() and (var_dir / CSV_GLOBAL_RM).exists()
            has_block  = (var_dir / CSV_BLOCK_CONCAT).exists() and (var_dir / CSV_BLOCK_RM).exists()
            if has_global or has_block:
                yield var_dir


def plot_global_one(csv_dir: Path, root: Path, dataset_label: str) -> Path:
    c_global_concat = csv_dir / CSV_GLOBAL_CONCAT
    c_global_rm     = csv_dir / CSV_GLOBAL_RM

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
        f"{dataset_label} — global MOS: concat vs running mean\n{rel}",
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


def _ensure_mid_start_end(df: pd.DataFrame, what: str) -> tuple[pd.DataFrame, str, str, str]:
    if "block_start_s" in df.columns and "block_end_s" in df.columns:
        s_col, e_col = "block_start_s", "block_end_s"
    else:
        s_col = pick_first_existing(df, ["block_start_s", "start_s", "start"], f"{what} start")
        e_col = pick_first_existing(df, ["block_end_s", "end_s", "end"], f"{what} end")

    if "block_mid_s" not in df.columns:
        df = df.copy()
        df["block_mid_s"] = 0.5 * (df[s_col] + df[e_col])
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


def plot_block_one(csv_dir: Path, root: Path, dataset_label: str) -> Path:
    c_block_concat = csv_dir / CSV_BLOCK_CONCAT
    c_block_rm     = csv_dir / CSV_BLOCK_RM

    df_bc = pd.read_csv(c_block_concat)
    df_br = pd.read_csv(c_block_rm)

    df_bc, s_bc, e_bc, x_bc = _ensure_mid_start_end(df_bc, "block concat")
    df_br, s_br, e_br, x_br = _ensure_mid_start_end(df_br, "block rm")

    y_bc = pick_first_existing(
        df_bc,
        ["mos_pred_block_concat", "mos_pred_block", "mos_pred", "mos"],
        "block concat predicted MOS",
    )
    tgt_bc = pick_first_existing(
        df_bc,
        ["mos_target_block_durw", "mos_target_block_mean", "mos_target_block", "mos_target"],
        "block concat target MOS",
    )

    y_br = pick_first_existing(
        df_br,
        ["mos_pred_block_rm_durw", "mos_pred_block_rm", "mos_pred_block", "mos_pred", "mos"],
        "block RM predicted MOS",
    )
    tgt_br = pick_first_existing(
        df_br,
        ["mos_target_block_durw", "mos_target_block_mean", "mos_target_block", "mos_target"],
        "block RM target MOS",
    )

    df_bc = df_bc.sort_values(x_bc).reset_index(drop=True)
    df_br = df_br.sort_values(x_br).reset_index(drop=True)

    tgt_prefix_label = "target MOS (block/prefix)"
    tgt_rm_label = "target MOS (block/running)"
    tgt_prefix_label += " (dur-weighted)" if is_dur_weighted(tgt_bc) else " (unweighted)"
    tgt_rm_label     += " (dur-weighted)" if is_dur_weighted(tgt_br) else " (unweighted)"

    try:
        rel = str(csv_dir.relative_to(root))
    except Exception:
        rel = str(csv_dir)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # block "towers" (more visible), but NO fill
    xs_c, ys_c = _stairs_from_blocks(df_bc, s_bc, e_bc, y_bc)
    xs_r, ys_r = _stairs_from_blocks(df_br, s_br, e_br, y_br)

    ax.plot(xs_c, ys_c, color=COL_CONCAT, linewidth=STEP_W_CONCAT, alpha=STEP_ALPHA_CONCAT, zorder=2)
    ax.plot(xs_r, ys_r, color=COL_RM, linewidth=STEP_W_RM, alpha=STEP_ALPHA_RM, linestyle="--", zorder=2)

    if DRAW_BLOCK_BOUNDARIES:
        boundaries = sorted(set([float(v) for v in df_bc[s_bc].tolist()] + [float(v) for v in df_bc[e_bc].tolist()]))
        for b in boundaries:
            ax.axvline(b, color="#000000", linewidth=BOUNDARY_W, alpha=BOUNDARY_ALPHA, zorder=0)

    # main curves (unchanged)
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
        f"{dataset_label} — block MOS: concat vs running mean\n{rel}",
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


def _process_dataset(results_block: Path, dataset: str, dataset_label: str) -> tuple[int, int]:
    roots = _iter_run_roots(results_block, dataset)
    if not roots:
        print(f"[WARN] no roots found for dataset={dataset} under {results_block}")
        return 0, 0

    ok, fail = 0, 0
    for root in roots:
        dirs = sorted(set(iter_variant_dirs(root)), key=lambda p: str(p))
        if not dirs:
            print(f"[WARN] no result dirs found under {root}")
            continue

        for d in dirs:
            try:
                wrote_any = False

                if (d / CSV_GLOBAL_CONCAT).exists() and (d / CSV_GLOBAL_RM).exists():
                    out = plot_global_one(d, root, dataset_label)
                    print(f"[OK]  {dataset}/{root.parent.name}/{d.relative_to(root)} -> {out.name}")
                    ok += 1
                    wrote_any = True

                if (d / CSV_BLOCK_CONCAT).exists() and (d / CSV_BLOCK_RM).exists():
                    out = plot_block_one(d, root, dataset_label)
                    print(f"[OK]  {dataset}/{root.parent.name}/{d.relative_to(root)} -> {out.name}")
                    ok += 1
                    wrote_any = True

                if not wrote_any:
                    print(f"[SKIP] {dataset}/{root.parent.name}/{d.relative_to(root)} (missing csvs)")
            except Exception as e:
                try:
                    rel = d.relative_to(root)
                except Exception:
                    rel = d
                print(f"[FAIL] {dataset}/{root.parent.name}/{rel}: {e}")
                fail += 1

    return ok, fail


def main():
    results_block = find_results_block_root()

    ok1, fail1 = _process_dataset(results_block, "bvcc", "BVCC")
    ok2, fail2 = _process_dataset(results_block, "somos", "SOMOS")

    print(f"\nDONE: ok={ok1+ok2} fail={fail1+fail2}")


if __name__ == "__main__":
    main()