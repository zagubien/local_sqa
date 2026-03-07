from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

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

# expected train folders: train18, train19, ...
TRAIN_IDS = {18, 19, 23, 24, 26, 27}

# stronger "towers" (step edges) without fill
STEP_ALPHA_CONCAT = 0.55
STEP_ALPHA_RM     = 0.50
STEP_W_CONCAT     = 1.8
STEP_W_RM         = 1.6

DRAW_BLOCK_BOUNDARIES = False
BOUNDARY_ALPHA = 0.06
BOUNDARY_W = 0.7


def fkw():
    return {"fontproperties": PROP} if PROP else {}


def pick_first_existing(df: pd.DataFrame, candidates: List[str], what: str) -> str:
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
    return ("_debug" in p.parts) or any(part.endswith("_debug") for part in p.parts)


def _valid_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if p.name in SKIP_DIR_NAMES:
        return False
    if p.name.startswith(SKIP_PREFIXES):
        return False
    if _is_debug_path(p):
        return False
    return True


def find_dialogue_root() -> Path:
    """
    find results_block_dialogue in:
      - script dir
      - parent
      - cwd
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "results_block_dialogue",
        script_dir.parent / "results_block_dialogue",
        Path.cwd() / "results_block_dialogue",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    raise FileNotFoundError("could not find results_block_dialogue next to this script / parent / cwd")


def find_block_dataset_roots(dialogue_root: Path) -> List[Path]:
    roots: List[Path] = []
    for p in dialogue_root.iterdir():
        if not _valid_dir(p):
            continue
        if not re.fullmatch(r"dialogue_block\d+", p.name):
            continue
        for ds in ["bvcc", "somos"]:
            ds_dir = p / ds
            if ds_dir.exists() and ds_dir.is_dir():
                roots.append(ds_dir.resolve())
    return sorted(roots, key=lambda x: str(x))


def _parse_train_dirname(name: str) -> int | None:
    m = re.fullmatch(r"train(\d+)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def iter_variant_dirs(ds_root: Path) -> Iterable[Path]:
    """
    ds_root example:
      results_block_dialogue/dialogue_block1/bvcc

    expected:
      train18/
        dlg_* /
          original|no_pause|long_pause|long_pause_noise/
            *.csv
    """
    for train_dir in ds_root.iterdir():
        if not _valid_dir(train_dir):
            continue

        tr = _parse_train_dirname(train_dir.name)
        if tr is None or tr not in TRAIN_IDS:
            continue

        for pair_dir in train_dir.iterdir():
            if not _valid_dir(pair_dir):
                continue
            if not (pair_dir.name.startswith("dlg_") or pair_dir.name.startswith("rand_dlg_") or pair_dir.name.startswith("rev_dlg_")):
                continue

            for var_dir in pair_dir.iterdir():
                if not _valid_dir(var_dir):
                    continue

                has_global = (var_dir / CSV_GLOBAL_CONCAT).exists() and (var_dir / CSV_GLOBAL_RM).exists()
                has_block  = (var_dir / CSV_BLOCK_CONCAT).exists() and (var_dir / CSV_BLOCK_RM).exists()
                if has_global or has_block:
                    yield var_dir


def plot_global_one(csv_dir: Path, ds_root: Path) -> Path:
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

    rel = str(csv_dir.relative_to(ds_root))
    dataset_name = ds_root.parts[-1].upper()  # bvcc / somos
    block_name = ds_root.parent.name          # dialogue_blockX

    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.plot(
        df_gc[x_gc], df_gc[y_gc],
        color=COL_CONCAT, linewidth=3.0,
        marker="o", markersize=6,
        markerfacecolor="white", markeredgewidth=2.0,
        label="concat (predicted, global)",
    )
    ax.plot(
        df_gr[x_gr], df_gr[y_gr],
        color=COL_RM, linewidth=1.8, linestyle="--",
        label="running mean (predicted, global)",
    )
    ax.plot(
        df_gc[x_gc], df_gc[tgt_gc],
        color=COL_TGT, linewidth=1.8, linestyle="-", alpha=0.85,
        label=tgt_prefix_label,
    )
    ax.plot(
        df_gr[x_gr], df_gr[tgt_gr],
        color=COL_TGT, linewidth=1.2, linestyle="--", alpha=0.55,
        label=tgt_rm_label,
    )

    ax.set_title(f"{dataset_name} {block_name} — global MOS\n{rel}", fontsize=18, **fkw())
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


def _ensure_mid_start_end(df: pd.DataFrame, what: str) -> Tuple[pd.DataFrame, str, str, str]:
    if "block_start_s" in df.columns and "block_end_s" in df.columns:
        s_col, e_col = "block_start_s", "block_end_s"
    else:
        s_col = pick_first_existing(df, ["block_start_s", "start_s", "start"], f"{what} start")
        e_col = pick_first_existing(df, ["block_end_s", "end_s", "end"], f"{what} end")

    if "block_mid_s" not in df.columns:
        df = df.copy()
        df["block_mid_s"] = 0.5 * (df[s_col].astype(float) + df[e_col].astype(float))
    return df, s_col, e_col, "block_mid_s"


def _stairs_from_blocks(df: pd.DataFrame, start_col: str, end_col: str, y_col: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for _, r in df.iterrows():
        s = float(r[start_col])
        e = float(r[end_col])
        y = float(r[y_col])
        xs.extend([s, e])
        ys.extend([y, y])
    return xs, ys


def plot_block_one(csv_dir: Path, ds_root: Path) -> Path:
    df_bc = pd.read_csv(csv_dir / CSV_BLOCK_CONCAT)
    df_br = pd.read_csv(csv_dir / CSV_BLOCK_RM)

    df_bc, s_bc, e_bc, x_bc = _ensure_mid_start_end(df_bc, "block concat")
    df_br, s_br, e_br, x_br = _ensure_mid_start_end(df_br, "block rm")

    y_bc = pick_first_existing(df_bc, ["mos_pred_block_concat", "mos_pred_block", "mos_pred", "mos"], "block concat predicted MOS")
    tgt_bc = pick_first_existing(df_bc, ["mos_target_block_durw", "mos_target_block_mean", "mos_target_block", "mos_target"], "block concat target MOS")

    y_br = pick_first_existing(df_br, ["mos_pred_block_rm_durw", "mos_pred_block_rm", "mos_pred_block", "mos_pred", "mos"], "block RM predicted MOS")
    tgt_br = pick_first_existing(df_br, ["mos_target_block_durw", "mos_target_block_mean", "mos_target_block", "mos_target"], "block RM target MOS")

    df_bc = df_bc.sort_values(x_bc).reset_index(drop=True)
    df_br = df_br.sort_values(x_br).reset_index(drop=True)

    tgt_prefix_label = "target MOS (block/prefix)"
    tgt_rm_label = "target MOS (block/running)"
    tgt_prefix_label += " (dur-weighted)" if is_dur_weighted(tgt_bc) else " (unweighted)"
    tgt_rm_label     += " (dur-weighted)" if is_dur_weighted(tgt_br) else " (unweighted)"

    rel = str(csv_dir.relative_to(ds_root))
    dataset_name = ds_root.parts[-1].upper()
    block_name = ds_root.parent.name

    fig, ax = plt.subplots(figsize=(13, 6.5))

    xs_c, ys_c = _stairs_from_blocks(df_bc, s_bc, e_bc, y_bc)
    xs_r, ys_r = _stairs_from_blocks(df_br, s_br, e_br, y_br)

    ax.plot(xs_c, ys_c, color=COL_CONCAT, linewidth=STEP_W_CONCAT, alpha=STEP_ALPHA_CONCAT, zorder=2)
    ax.plot(xs_r, ys_r, color=COL_RM, linewidth=STEP_W_RM, alpha=STEP_ALPHA_RM, linestyle="--", zorder=2)

    if DRAW_BLOCK_BOUNDARIES:
        boundaries = sorted(set([float(v) for v in df_bc[s_bc].tolist()] + [float(v) for v in df_bc[e_bc].tolist()]))
        for b in boundaries:
            ax.axvline(b, color="#000000", linewidth=BOUNDARY_W, alpha=BOUNDARY_ALPHA, zorder=0)

    ax.plot(
        df_bc[x_bc], df_bc[y_bc],
        color=COL_CONCAT, linewidth=3.0,
        marker="o", markersize=6,
        markerfacecolor="white", markeredgewidth=2.0,
        label="concat (predicted, block)", zorder=5,
    )
    ax.plot(
        df_br[x_br], df_br[y_br],
        color=COL_RM, linewidth=1.8, linestyle="--",
        label="running mean (predicted, block)", zorder=5,
    )
    ax.plot(
        df_bc[x_bc], df_bc[tgt_bc],
        color=COL_TGT, linewidth=1.8, linestyle="-", alpha=0.85,
        label=tgt_prefix_label, zorder=6,
    )
    ax.plot(
        df_br[x_br], df_br[tgt_br],
        color=COL_TGT, linewidth=1.2, linestyle="--", alpha=0.55,
        label=tgt_rm_label, zorder=6,
    )

    ax.set_title(f"{dataset_name} {block_name} — block MOS\n{rel}", fontsize=18, **fkw())
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


def main() -> None:
    dialogue_root = find_dialogue_root()
    roots = find_block_dataset_roots(dialogue_root)
    if not roots:
        raise FileNotFoundError(f"no dialogue_block*/(bvcc|somos) found under {dialogue_root}")

    ok, fail = 0, 0
    for ds_root in roots:
        dirs = sorted(set(iter_variant_dirs(ds_root)), key=lambda p: str(p))
        if not dirs:
            print(f"[WARN] no result dirs under {ds_root}")
            continue

        for d in dirs:
            try:
                wrote_any = False

                if (d / CSV_GLOBAL_CONCAT).exists() and (d / CSV_GLOBAL_RM).exists():
                    out = plot_global_one(d, ds_root)
                    print(f"[OK]  {d.relative_to(ds_root)} -> {out.name}")
                    ok += 1
                    wrote_any = True

                if (d / CSV_BLOCK_CONCAT).exists() and (d / CSV_BLOCK_RM).exists():
                    out = plot_block_one(d, ds_root)
                    print(f"[OK]  {d.relative_to(ds_root)} -> {out.name}")
                    ok += 1
                    wrote_any = True

                if not wrote_any:
                    print(f"[SKIP] {d.relative_to(ds_root)} (missing csvs)")
            except Exception as e:
                try:
                    rel = d.relative_to(ds_root)
                except Exception:
                    rel = d
                print(f"[FAIL] {rel}: {e}")
                fail += 1

    print(f"\nDONE: ok={ok} fail={fail} roots={[str(r) for r in roots]}")


if __name__ == "__main__":
    main()