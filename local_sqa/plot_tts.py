from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CSV_CONCAT = "global_mos_concat.csv"
CSV_RM = "global_mos_running_mean.csv"

SKIP_DIRS = {"packets"}
SKIP_PREFIXES = ("_",)


def pick_first_existing(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing column for {what}. have={list(df.columns)} candidates={candidates}")


def is_dur_weighted(col_name: str) -> bool:
    s = col_name.lower()
    return ("durw" in s) or ("dur_w" in s) or ("dur-weight" in s) or ("durweight" in s)


def pretty_axes(ax):
    ax.grid(True, linewidth=0.6, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def find_repo_root(script_path: Path) -> Path:
    return script_path.resolve().parent.parent


def iter_csv_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if d.name in SKIP_DIRS:
            continue
        if d.name.startswith(SKIP_PREFIXES):
            continue
        c1 = d / CSV_CONCAT
        c2 = d / CSV_RM
        if c1.exists() and c2.exists():
            out.append(d)
    return sorted(set(out), key=lambda p: str(p))


def parse_dataset_and_system(csv_dir: Path, tts_base: Path) -> Optional[Tuple[str, str, str]]:
    try:
        rel = csv_dir.relative_to(tts_base)
    except Exception:
        rel = csv_dir

    parts = list(rel.parts)
    for i, p in enumerate(parts):
        if p in ("bvcc", "somos"):
            if i + 1 >= len(parts):
                return None
            dataset = p
            system = parts[i + 1]
            tag = "/".join(parts[: i + 3]) if i + 2 < len(parts) else "/".join(parts)
            return dataset, system, tag
    return None


def find_baseline_original_dir(results_dir: Path, dataset: str, system: str) -> Optional[Path]:
    if dataset == "bvcc":
        base = results_dir / "BVCC_26"
        cand = base / system / "original"
        return cand if cand.exists() else None

    if dataset == "somos":
        base = results_dir / "SOMOS_26"
        cands = [
            base / system / "original",
            base / f"sv{system}" / "original",
            base / f"sys{system}" / "original",
            base / f"sv_{system}" / "original",
        ]
        for c in cands:
            if c.exists():
                return c
        return None

    return None


def load_global_pair(csv_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_c = pd.read_csv(csv_dir / CSV_CONCAT)
    df_r = pd.read_csv(csv_dir / CSV_RM)
    return df_c, df_r


def pick_cols_concat(df: pd.DataFrame, prefix: str) -> Tuple[str, str, str]:
    x = pick_first_existing(df, ["seconds", "seconds_orig", "elapsed_s", "time_s"], f"{prefix} concat x-axis")
    y = pick_first_existing(df, ["mos_pred_concat", "mos_pred", "mos"], f"{prefix} concat predicted MOS")
    tgt = pick_first_existing(
        df,
        ["mos_target_concat_durw", "mos_target_concat_mean", "mos_target_concat_avg", "mos_target_concat", "mos_target"],
        f"{prefix} concat target MOS",
    )
    return x, y, tgt


def pick_cols_rm(df: pd.DataFrame, prefix: str) -> Tuple[str, str, str]:
    x = pick_first_existing(df, ["elapsed_s", "seconds", "seconds_orig", "time_s"], f"{prefix} rm x-axis")
    y = pick_first_existing(
        df,
        ["mos_pred_rm_durw", "mos_pred_rm_mean", "mos_pred_rm", "mos_running_mean", "mos_pred", "mos"],
        f"{prefix} rm predicted MOS",
    )
    tgt = pick_first_existing(
        df,
        ["mos_target_rm_durw", "mos_target_rm_mean", "mos_target_rm", "mos_target"],
        f"{prefix} rm target MOS",
    )
    return x, y, tgt


def maybe_find_needle_x(df_concat: pd.DataFrame, xcol: str) -> Optional[float]:
    # supports both flags you used so far
    for flag in ["is_real_needle_segment", "is_tts_replaced_segment", "is_needle_segment"]:
        if flag in df_concat.columns:
            m = df_concat[flag].astype(float) == 1.0
            if m.any():
                try:
                    return float(df_concat.loc[m, xcol].iloc[0])
                except Exception:
                    return None
    return None


def plot_one(tts_dir: Path, baseline_dir: Optional[Path], title_tag: str, out_name: str = "plot_tts_vs_original.png") -> Path:
    df_tc, df_tr = load_global_pair(tts_dir)
    df_tc = df_tc.sort_values(by=list(df_tc.columns)[0]).reset_index(drop=True)
    df_tr = df_tr.sort_values(by=list(df_tr.columns)[0]).reset_index(drop=True)

    x_tc, y_tc, tgt_tc = pick_cols_concat(df_tc, "tts")
    x_tr, y_tr, tgt_tr = pick_cols_rm(df_tr, "tts")

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # tts predicted
    ax.plot(df_tc[x_tc], df_tc[y_tc], linewidth=2.6, marker="o", markersize=5, label="TTS concat (pred)")
    ax.plot(df_tr[x_tr], df_tr[y_tr], linewidth=1.8, linestyle="--", label="TTS running mean (pred)")

    # tts target
    tgt_label_c = "target (concat)"
    tgt_label_r = "target (rm)"
    tgt_label_c += " dur-w" if is_dur_weighted(tgt_tc) else ""
    tgt_label_r += " dur-w" if is_dur_weighted(tgt_tr) else ""
    ax.plot(df_tc[x_tc], df_tc[tgt_tc], linewidth=1.4, alpha=0.70, label=f"TTS {tgt_label_c}")
    ax.plot(df_tr[x_tr], df_tr[tgt_tr], linewidth=1.2, linestyle="--", alpha=0.55, label=f"TTS {tgt_label_r}")

    # baseline overlay
    if baseline_dir is not None:
        try:
            df_oc, df_or = load_global_pair(baseline_dir)
            x_oc, y_oc, _tgt_oc = pick_cols_concat(df_oc, "orig")
            x_or, y_or, _tgt_or = pick_cols_rm(df_or, "orig")

            df_oc = df_oc.sort_values(x_oc).reset_index(drop=True)
            df_or = df_or.sort_values(x_or).reset_index(drop=True)

            ax.plot(df_oc[x_oc], df_oc[y_oc], linewidth=2.0, alpha=0.55, label="ORIG concat (pred)")
            ax.plot(df_or[x_or], df_or[y_or], linewidth=1.5, linestyle="--", alpha=0.55, label="ORIG running mean (pred)")
        except Exception as e:
            ax.text(0.02, 0.02, f"baseline load failed: {e}", transform=ax.transAxes, fontsize=10, va="bottom")

    # needle marker (if present)
    nx = maybe_find_needle_x(df_tc, x_tc)
    if nx is not None:
        ax.axvline(nx, linewidth=1.6, linestyle=":", alpha=0.9, label="needle position")

    ax.set_title(f"TTS vs ORIGINAL — {title_tag}", fontsize=16)
    ax.set_xlabel("audio length [s]", fontsize=14)
    ax.set_ylabel("MOS", fontsize=14)
    pretty_axes(ax)

    leg = ax.legend(loc="upper right", fontsize=11, frameon=True)
    leg.get_frame().set_alpha(0.95)

    fig.tight_layout()
    out_path = tts_dir / out_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["bvcc", "somos", "both"], default="both")
    ap.add_argument("--tts-subdirs", nargs="*", default=["results_tts/reverse_tts", "results_tts"])
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    pkg_root = script_path.parent
    results_dir = pkg_root / "results"

    print("repo_root:", repo_root)
    print("results_dir:", results_dir)

    ok, fail, skipped = 0, 0, 0

    for sub in args.tts_subdirs:
        tts_base = pkg_root / sub
        if not tts_base.exists():
            continue

        csv_dirs = iter_csv_dirs(tts_base)
        if not csv_dirs:
            continue

        for d in csv_dirs:
            parsed = parse_dataset_and_system(d, tts_base)
            if parsed is None:
                continue
            dataset, system, tag = parsed

            if args.only != "both" and dataset != args.only:
                continue

            baseline_dir = find_baseline_original_dir(results_dir, dataset, system)
            if baseline_dir is None:
                skipped += 1

            try:
                out = plot_one(d, baseline_dir, f"{sub}/{dataset}/{system}")
                print(f"[OK] {sub}:{dataset}:{system} -> {out}")
                ok += 1
            except Exception as e:
                print(f"[FAIL] {sub}:{dataset}:{system}: {e}")
                fail += 1

    print(f"\nDONE: ok={ok} fail={fail} skipped_baseline={skipped}")


if __name__ == "__main__":
    main()
