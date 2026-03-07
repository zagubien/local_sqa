from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# set this if autodetect fails
FORCE_EXPERIMENT_ROOT: Optional[str] = None

# overwrite pngs
OVERWRITE = True

# expected files
GLOBAL_CONCAT = "global_mos_concat.csv"
GLOBAL_RM = "global_mos_running_mean.csv"
BLOCK_CONCAT = "block_mos_concat.csv"
BLOCK_RM = "block_mos_running_mean.csv"

# output plots
OUT_GLOBAL_PNG = "plot_global_concat_vs_running_mean.png"
OUT_BLOCK_PNG = "plot_block_concat_vs_block_rm.png"

# variants
VARIANTS = ["original", "no_pause", "long_pause", "long_pause_noise"]

# UPB-ish colors (pred)
C_CONCAT = "#009FE3"   # himmelblau
C_RM = "#00305D"       # ultrablau

# targets (like old plot: black + grey dashed)
C_TGT = "#111111"
C_TGT2 = "#7A7A7A"

# styles similar to your "old" plot example
LINE_W = 2.6
TARGET_W = 2.2
RM_DASH = (0, (4, 2))
TGT_DASH = (0, (3, 2))
MARKER_SIZE = 7
MARKER_EDGE_W = 2.2

# block diagram styles
BAR_ALPHA = 0.30
BAR_EDGE_W = 2.0

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})


def _try_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        x = x.strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def find_experiment_root(script_dir: Path) -> Path:
    if FORCE_EXPERIMENT_ROOT:
        p = Path(FORCE_EXPERIMENT_ROOT).expanduser()
        if p.exists():
            return p.resolve()

    # prefer results_block
    for cand in [
        script_dir / "results_block",
        script_dir.parent / "results_block",
        Path.cwd() / "results_block",
    ]:
        if cand.exists():
            return cand.resolve()

    # fallback to results
    for cand in [
        script_dir / "results",
        script_dir.parent / "results",
        Path.cwd() / "results",
    ]:
        if cand.exists():
            return cand.resolve()

    return (script_dir / "results_block").resolve()


def find_bvcc_run_dirs(root: Path) -> List[Path]:
    cands = sorted([p for p in root.glob("bvcc_*") if p.is_dir()])
    if cands:
        return cands
    return sorted([p for p in root.rglob("bvcc_*") if p.is_dir()])


def is_system_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    n = p.name
    return n.startswith("sys") or n.startswith("rev_sys") or n.startswith("rand_sys")


def _safe_savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _apply_axes_style(xlabel: str) -> None:
    plt.xlabel(xlabel)
    plt.ylabel("MOS")
    plt.grid(True, alpha=0.18)
    plt.legend(loc="upper right", frameon=True, framealpha=0.95)


# -------- global csv loading --------

def load_global_concat(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # seconds, mos_pred_concat, mos_target_concat_durw
    rows = read_csv_rows(path)
    t, pred, tgt = [], [], []
    for r in rows:
        a = _try_float(r.get("seconds", ""))
        b = _try_float(r.get("mos_pred_concat", ""))
        c = _try_float(r.get("mos_target_concat_durw", ""))
        if a is None or b is None or c is None:
            continue
        t.append(a)
        pred.append(b)
        tgt.append(c)
    return np.array(t, float), np.array(pred, float), np.array(tgt, float)


def load_global_rm(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # elapsed_s, mos_pred_rm_durw, mos_target_rm_durw
    rows = read_csv_rows(path)
    t, pred, tgt = [], [], []
    for r in rows:
        a = _try_float(r.get("elapsed_s", ""))
        b = _try_float(r.get("mos_pred_rm_durw", ""))
        c = _try_float(r.get("mos_target_rm_durw", ""))
        if a is None or b is None or c is None:
            continue
        t.append(a)
        pred.append(b)
        tgt.append(c)
    return np.array(t, float), np.array(pred, float), np.array(tgt, float)


# -------- block csv loading --------

def load_block_concat(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # block_start_s, block_end_s, mos_pred_block_concat, mos_target_block_durw
    rows = read_csv_rows(path)
    bs, be, pred, tgt = [], [], [], []
    for r in rows:
        a = _try_float(r.get("block_start_s", ""))
        b = _try_float(r.get("block_end_s", ""))
        mp = _try_float(r.get("mos_pred_block_concat", ""))
        mt = _try_float(r.get("mos_target_block_durw", ""))
        if a is None or b is None or mp is None or mt is None:
            continue
        bs.append(a)
        be.append(b)
        pred.append(mp)
        tgt.append(mt)
    bs = np.array(bs, float)
    be = np.array(be, float)
    pred = np.array(pred, float)
    tgt = np.array(tgt, float)
    mid = 0.5 * (bs + be)
    return bs, be, mid, pred, tgt


def load_block_rm(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # block_start_s, block_end_s, mos_pred_block_rm_durw, mos_target_block_durw
    rows = read_csv_rows(path)
    bs, be, pred, tgt = [], [], [], []
    for r in rows:
        a = _try_float(r.get("block_start_s", ""))
        b = _try_float(r.get("block_end_s", ""))
        mp = _try_float(r.get("mos_pred_block_rm_durw", ""))
        mt = _try_float(r.get("mos_target_block_durw", ""))
        if a is None or b is None or mp is None or mt is None:
            continue
        bs.append(a)
        be.append(b)
        pred.append(mp)
        tgt.append(mt)
    bs = np.array(bs, float)
    be = np.array(be, float)
    pred = np.array(pred, float)
    tgt = np.array(tgt, float)
    mid = 0.5 * (bs + be)
    return bs, be, mid, pred, tgt


# -------- plotting --------

def plot_global(variant_dir: Path, title_prefix: str) -> bool:
    p_concat = variant_dir / GLOBAL_CONCAT
    p_rm = variant_dir / GLOBAL_RM
    out_png = variant_dir / OUT_GLOBAL_PNG

    if not p_concat.exists() or not p_rm.exists():
        return False
    if out_png.exists() and not OVERWRITE:
        return True

    t_c, pred_c, tgt_prefix = load_global_concat(p_concat)
    t_r, pred_r, tgt_running = load_global_rm(p_rm)

    if t_c.size < 2 or t_r.size < 2:
        return False

    plt.figure(figsize=(12, 6))

    plt.plot(
        t_c, pred_c,
        color=C_CONCAT,
        linewidth=LINE_W,
        marker="o",
        markersize=MARKER_SIZE,
        markerfacecolor="white",
        markeredgewidth=MARKER_EDGE_W,
        label="concat (predicted, global)",
    )

    plt.plot(
        t_r, pred_r,
        color=C_RM,
        linewidth=LINE_W,
        linestyle=RM_DASH,
        label="running mean (predicted, global)",
    )

    plt.plot(
        t_c, tgt_prefix,
        color=C_TGT,
        linewidth=TARGET_W,
        label="target MOS (prefix) (dur-weighted)",
    )

    plt.plot(
        t_r, tgt_running,
        color=C_TGT2,
        linewidth=TARGET_W,
        linestyle=TGT_DASH,
        label="target MOS (running) (dur-weighted)",
    )

    plt.title(f"{title_prefix}\nGlobal MOS: concat vs running mean")
    _apply_axes_style("audio length [s]")
    _safe_savefig(out_png)
    return True


def plot_blocks_as_block_diagram(variant_dir: Path, title_prefix: str) -> bool:
    p_bc = variant_dir / BLOCK_CONCAT
    p_br = variant_dir / BLOCK_RM
    out_png = variant_dir / OUT_BLOCK_PNG

    if not p_bc.exists() or not p_br.exists():
        return False
    if out_png.exists() and not OVERWRITE:
        return True

    bs_c, be_c, mid_c, pred_c, tgt_prefix = load_block_concat(p_bc)
    bs_r, be_r, mid_r, pred_r, tgt_running = load_block_rm(p_br)

    if pred_c.size < 1 or pred_r.size < 1:
        return False

    plt.figure(figsize=(12, 6))

    # concat blocks as filled bars spanning [start,end)
    widths_c = be_c - bs_c
    plt.bar(
        bs_c,
        pred_c,
        width=widths_c,
        align="edge",
        color=C_CONCAT,
        alpha=BAR_ALPHA,
        edgecolor=C_CONCAT,
        linewidth=BAR_EDGE_W,
        label="concat (predicted, block)",
    )

    # running-mean blocks as outlined bars (no fill), slightly shifted to avoid perfect overlap
    # shift by a tiny fraction of block width (visual only)
    shift = 0.06
    widths_r = be_r - bs_r
    plt.bar(
        bs_r + shift * widths_r,
        pred_r,
        width=(1.0 - 2.0 * shift) * widths_r,
        align="edge",
        color="none",
        edgecolor=C_RM,
        linewidth=BAR_EDGE_W,
        linestyle="-",
        label="running mean (predicted, block)",
    )

    # targets as step curves (blockwise constant), like a "block diagram" line
    plt.step(
        bs_c,
        tgt_prefix,
        where="post",
        color=C_TGT,
        linewidth=TARGET_W,
        label="target MOS (block/prefix) (dur-weighted)",
    )
    plt.step(
        bs_r,
        tgt_running,
        where="post",
        color=C_TGT2,
        linewidth=TARGET_W,
        linestyle=TGT_DASH,
        label="target MOS (block/running) (dur-weighted)",
    )

    plt.title(f"{title_prefix}\nBlock MOS (20s): concat vs running mean")
    _apply_axes_style("time [s]")
    _safe_savefig(out_png)
    return True


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = find_experiment_root(script_dir)

    run_dirs = find_bvcc_run_dirs(root)
    if not run_dirs:
        print(f"[error] no bvcc_* run dir found under: {root}")
        return

    made_global = 0
    made_block = 0
    skipped = 0

    for run_dir in run_dirs:
        sys_dirs = sorted([p for p in run_dir.iterdir() if is_system_dir(p)])
        if not sys_dirs:
            continue

        for sys_dir in sys_dirs:
            for variant in VARIANTS:
                vdir = sys_dir / variant
                if not vdir.exists():
                    continue

                title_prefix = f"{run_dir.name} — {sys_dir.name} / {variant}"

                ok1 = plot_global(vdir, title_prefix)
                ok2 = plot_blocks_as_block_diagram(vdir, title_prefix)

                if ok1:
                    made_global += 1
                if ok2:
                    made_block += 1
                if not ok1 and not ok2:
                    skipped += 1

    print(f"[ok] root: {root}")
    print(f"[ok] global plots written: {made_global}")
    print(f"[ok] block plots written:  {made_block}")
    if skipped:
        print(f"[warn] skipped (missing csvs): {skipped}")


if __name__ == "__main__":
    main()