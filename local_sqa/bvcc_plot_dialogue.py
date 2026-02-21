from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORCE_RESULTS_ROOT: Optional[str] = None

OUT_NAME = "plot_rm_vs_targets.png"

NUKE_BAD_PLOTS_GLOBALLY = True
BAD_PLOT_GLOB = "plot_raw_vs_filtered*.png"

# if true: in each pause_* delete all plot*.png except OUT_NAME
KEEP_ONLY_OUT_PLOT = True

DIALOGUE_PREFIX = "BVCC_dialogue_"
SCHEDULE_DIRS = {"five_sentences", "one_sentence", "three_sentences"}


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


def find_results_root(script_dir: Path) -> Path:
    if FORCE_RESULTS_ROOT:
        p = Path(FORCE_RESULTS_ROOT).expanduser()
        if p.exists() and p.is_dir():
            return p.resolve()

    for cand in [script_dir / "results", script_dir.parent / "results", Path.cwd() / "results"]:
        if cand.exists() and cand.is_dir():
            return cand.resolve()

    raise FileNotFoundError("could not find results/ folder")


def _train_id_from_name(name: str) -> Optional[int]:
    if not name.startswith(DIALOGUE_PREFIX):
        return None
    tail = name[len(DIALOGUE_PREFIX):]
    return int(tail) if tail.isdigit() else None


def find_dialogue_roots(results_root: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        tid = _train_id_from_name(d.name)
        if tid is None:
            continue
        out.append((tid, d.resolve()))
    out.sort(key=lambda x: x[0])
    return out


def parse_packet_csv(packet_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = packet_path.read_text(encoding="utf-8").splitlines()
    header = "idx,src_system,fn,mos_target,path,dur_s"

    i0 = None
    for i, r in enumerate(rows):
        if r.strip() == header:
            i0 = i + 1
            break
    if i0 is None:
        raise RuntimeError(f"invalid packet.csv (header not found): {packet_path}")

    idxs, mos, dur = [], [], []
    for r in rows[i0:]:
        r = r.strip()
        if not r:
            continue
        parts = r.split(",", 5)
        if len(parts) != 6:
            continue
        idx_s, _src, _fn, mos_s, _p, dur_s = parts
        idxs.append(int(idx_s))
        mos.append(float(mos_s))
        dur.append(float(dur_s))

    if len(idxs) < 2:
        raise RuntimeError(f"too few packet rows: {packet_path}")

    return np.asarray(idxs, dtype=int), np.asarray(mos, dtype=float), np.asarray(dur, dtype=float)


def durw_prefix_mean(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(durations, dtype=np.float64)
    d = np.maximum(d, 1e-12)
    num = np.cumsum(v * d)
    den = np.cumsum(d)
    return (num / den).astype(np.float64)


def load_running_mean(rm_csv: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv_rows(rm_csv)
    if not rows:
        raise RuntimeError(f"empty csv: {rm_csv}")

    x, y_pred_rm, y_tgt_rm = [], [], []
    for r in rows:
        xs = _try_float(r.get("elapsed_orig_s", ""))
        pr = _try_float(r.get("mos_pred_rm_durw", ""))
        tg = _try_float(r.get("mos_target_rm_durw", ""))
        if xs is None or pr is None or tg is None:
            continue
        x.append(xs)
        y_pred_rm.append(pr)
        y_tgt_rm.append(tg)

    if len(x) < 2:
        raise RuntimeError(f"too few usable rows: {rm_csv}")

    return np.asarray(x, dtype=float), np.asarray(y_pred_rm, dtype=float), np.asarray(y_tgt_rm, dtype=float)


def load_concat(concat_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(concat_csv)
    if not rows:
        raise RuntimeError(f"empty csv: {concat_csv}")

    x, y = [], []
    for r in rows:
        xs = _try_float(r.get("seconds_orig", ""))
        pr = _try_float(r.get("mos_pred_concat", ""))
        if xs is None or pr is None:
            continue
        x.append(xs)
        y.append(pr)

    if len(x) < 2:
        raise RuntimeError(f"too few usable rows: {concat_csv}")

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def get_packet_path(pause_dir: Path) -> Optional[Path]:
    c1 = pause_dir / "packet.csv"
    if c1.exists():
        return c1
    c2 = pause_dir.parent / "packet.csv"
    if c2.exists():
        return c2
    return None


def nuke_bad_plots(results_root: Path) -> int:
    n = 0
    for p in results_root.rglob(BAD_PLOT_GLOB):
        try:
            p.unlink()
            n += 1
        except Exception:
            pass
    return n


def cleanup_pause_plots(pause_dir: Path) -> int:
    n = 0
    if not KEEP_ONLY_OUT_PLOT:
        return 0
    for p in pause_dir.glob("plot*.png"):
        if p.name == OUT_NAME:
            continue
        try:
            p.unlink()
            n += 1
        except Exception:
            pass
    return n


def plot_one(
    out_png: Path,
    title: str,
    x_tgt: np.ndarray,
    y_tgt_concatgrid: np.ndarray,
    x_rm: np.ndarray,
    y_rm_raw: np.ndarray,
    y_tgt_rm: np.ndarray,
    x_cat: Optional[np.ndarray],
    y_cat_raw: Optional[np.ndarray],
) -> None:
    plt.figure(figsize=(20, 4))

    if x_cat is not None and y_cat_raw is not None:
        plt.plot(x_cat, y_cat_raw, color="0.75", linewidth=1.5, label="concat (raw pred)")

    plt.plot(x_tgt, y_tgt_concatgrid, color="C0", linewidth=2.5, label="target MOS (concat grid)")
    plt.plot(x_rm, y_rm_raw, color="C1", linestyle="--", linewidth=1.5, alpha=0.65, label="rm (raw pred)")
    plt.plot(x_rm, y_tgt_rm, color="C3", linestyle=":", linewidth=2.2, label="target MOS (rm grid)")

    plt.xlabel("audio length [s] (orig)")
    plt.ylabel("MOS")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def iter_pause_dirs(dialogue_root: Path) -> List[Path]:
    pause_dirs = []
    for sch in SCHEDULE_DIRS:
        base = dialogue_root / sch
        if not base.exists():
            continue
        for p in base.rglob("pause_*"):
            if p.is_dir():
                pause_dirs.append(p)
    pause_dirs = sorted(set(pause_dirs), key=lambda x: str(x))
    return pause_dirs


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    results_root = find_results_root(script_dir)

    dialogue_roots = find_dialogue_roots(results_root)
    if not dialogue_roots:
        print(f"[error] no {DIALOGUE_PREFIX}<N> dirs under: {results_root}")
        return

    nuked = 0
    if NUKE_BAD_PLOTS_GLOBALLY:
        nuked = nuke_bad_plots(results_root)

    total_pause = 0
    plotted = 0
    deleted = 0
    skipped = 0

    for tid, root in dialogue_roots:
        pause_dirs = iter_pause_dirs(root)
        total_pause += len(pause_dirs)

        for pause_dir in pause_dirs:
            rm_csv = pause_dir / "global_mos_running_mean.csv"
            concat_csv = pause_dir / "global_mos_concat.csv"

            if not rm_csv.exists():
                skipped += 1
                continue

            deleted += cleanup_pause_plots(pause_dir)

            try:
                x_rm, y_rm_raw, y_tgt_rm = load_running_mean(rm_csv)
            except Exception:
                skipped += 1
                continue

            n_rm = int(x_rm.size)
            pkt = get_packet_path(pause_dir)
            if pkt is not None:
                try:
                    _idxs, mos_t, dur_s = parse_packet_csv(pkt)
                    mos_t = mos_t[:n_rm]
                    dur_s = dur_s[:n_rm]
                    x_tgt = np.cumsum(dur_s).astype(np.float64)
                    y_tgt_concatgrid = durw_prefix_mean(mos_t, dur_s)
                except Exception:
                    x_tgt = x_rm
                    y_tgt_concatgrid = y_tgt_rm
            else:
                x_tgt = x_rm
                y_tgt_concatgrid = y_tgt_rm

            x_cat = None
            y_cat = None
            if concat_csv.exists():
                try:
                    x_cat, y_cat = load_concat(concat_csv)
                except Exception:
                    x_cat, y_cat = None, None

            rel = pause_dir.relative_to(root)
            title = f"{root.name} | {rel.as_posix()}"

            out_png = pause_dir / OUT_NAME
            plot_one(
                out_png=out_png,
                title=title,
                x_tgt=x_tgt,
                y_tgt_concatgrid=y_tgt_concatgrid,
                x_rm=x_rm,
                y_rm_raw=y_rm_raw,
                y_tgt_rm=y_tgt_rm,
                x_cat=x_cat,
                y_cat_raw=y_cat,
            )
            plotted += 1

    print(f"[ok] results_root: {results_root}")
    print(f"[ok] dialogue_roots: {len(dialogue_roots)}")
    if NUKE_BAD_PLOTS_GLOBALLY:
        print(f"[ok] nuked bad plots (global): {nuked}")
    if KEEP_ONLY_OUT_PLOT:
        print(f"[ok] deleted other plot*.png in pause dirs: {deleted}")
    print(f"[ok] pause dirs found: {total_pause}")
    print(f"[ok] plots written: {plotted}")
    if skipped:
        print(f"[warn] skipped (missing/bad rm csv): {skipped}")


if __name__ == "__main__":
    main()
