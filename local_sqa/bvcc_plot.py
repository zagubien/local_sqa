from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

FORCE_EXPERIMENT_ROOT: Optional[str] = None

PREFERRED_TARGET_NAMES = [
    "target_mos", "mos_target", "gt_mos", "true_mos", "mos_true", "target", "gt", "true", "mos",
]
PREFERRED_PRED_NAMES = [
    "pred_mos", "mos_pred", "pred", "prediction", "estimate", "est_mos", "y_pred",
]
PREFERRED_ID_NAMES = [
    "utt", "utterance", "utt_id", "id", "key", "item", "file", "filename", "wav", "wav_path", "path",
]


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


def numeric_columns(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    cols = list(rows[0].keys())
    good = []
    for c in cols:
        vals = []
        for row in rows:
            v = _try_float(row.get(c, ""))
            if v is not None:
                vals.append(v)
        if len(vals) >= max(3, int(0.5 * len(rows))):
            good.append(c)
    return good


def pick_named_column(cols: List[str], preferred: List[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for name in preferred:
        if name in cols_l:
            return cols_l[name]
    return None


def rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)

    xs = x[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> Tuple[int, float, float, float, float]:
    m = np.isfinite(y) & np.isfinite(yhat)
    y = y[m]
    yhat = yhat[m]
    n = int(y.size)
    if n < 2:
        return n, float("nan"), float("nan"), float("nan"), float("nan")

    pcc = float(np.corrcoef(y, yhat)[0, 1])

    ry = rankdata_avg_ties(y)
    ryh = rankdata_avg_ties(yhat)
    srcc = float(np.corrcoef(ry, ryh)[0, 1])

    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(float(np.mean(err * err))))

    return n, pcc, srcc, mae, rmse


def scatter_plot(y: np.ndarray, yhat: np.ndarray, title: str, out_png: Path) -> None:
    n, pcc, srcc, mae, rmse = compute_metrics(y, yhat)

    plt.figure(figsize=(6, 6))
    plt.scatter(y, yhat, alpha=0.6)

    lo = float(np.nanmin([np.nanmin(y), np.nanmin(yhat)]))
    hi = float(np.nanmax([np.nanmax(y), np.nanmax(yhat)]))
    if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
        plt.plot([lo, hi], [lo, hi])

    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.title(title)
    plt.text(
        0.02, 0.98,
        f"n={n}\nPCC={pcc:.3f}\nSRCC={srcc:.3f}\nMAE={mae:.3f}\nRMSE={rmse:.3f}",
        transform=plt.gca().transAxes,
        va="top",
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def find_experiment_root(script_dir: Path) -> Path:
    if FORCE_EXPERIMENT_ROOT:
        p = Path(FORCE_EXPERIMENT_ROOT).expanduser()
        if p.exists():
            return p.resolve()

    results_root = None
    for cand in [script_dir / "results", script_dir.parent / "results", Path.cwd() / "results"]:
        if cand.exists():
            results_root = cand.resolve()
            break
    if results_root is None:
        return (script_dir / "results" / "two_speaker_26").resolve()

    if (results_root / "two_speaker_26").exists():
        return (results_root / "two_speaker_26").resolve()

    # fallback: pick any two_speaker_* folder
    twos = sorted(results_root.glob("two_speaker_*"))
    if twos:
        return twos[-1].resolve()

    return results_root.resolve()


def extract_y_yhat_from_global(global_csv: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    rows = read_csv_rows(global_csv)
    if not rows:
        return None, None, "empty csv"

    cols = list(rows[0].keys())
    num_cols = numeric_columns(rows)

    tcol = pick_named_column(cols, PREFERRED_TARGET_NAMES)
    pcol = pick_named_column(cols, PREFERRED_PRED_NAMES)

    # if both exist but are not numeric -> ignore
    if tcol and tcol not in num_cols:
        tcol = None
    if pcol and pcol not in num_cols:
        pcol = None

    # easiest case: both inside same csv
    if tcol and pcol and tcol != pcol:
        y = np.array([_try_float(r.get(tcol, "")) for r in rows], dtype=float)
        yhat = np.array([_try_float(r.get(pcol, "")) for r in rows], dtype=float)
        return y, yhat, f"cols: {tcol}, {pcol}"

    # fallback: two numeric cols -> assume target,pred
    if len(num_cols) >= 2:
        y = np.array([_try_float(r.get(num_cols[0], "")) for r in rows], dtype=float)
        yhat = np.array([_try_float(r.get(num_cols[1], "")) for r in rows], dtype=float)
        return y, yhat, f"cols: {num_cols[0]}, {num_cols[1]}"

    # last try: join with packet.csv (one numeric column likely pred)
    if len(num_cols) == 1:
        pred_col = num_cols[0]
        pause_dir = global_csv.parent
        sys_dir = pause_dir.parent
        packet = sys_dir / "packet.csv"
        if not packet.exists():
            return None, None, "only one numeric col and no packet.csv"

        pr = read_csv_rows(packet)
        if not pr:
            return None, None, "packet.csv empty"

        pcols = list(pr[0].keys())
        pid = pick_named_column(cols, PREFERRED_ID_NAMES)
        qid = pick_named_column(pcols, PREFERRED_ID_NAMES)

        tcol_p = pick_named_column(pcols, PREFERRED_TARGET_NAMES)
        if tcol_p is None:
            pnum = numeric_columns(pr)
            if pnum:
                tcol_p = pnum[0]

        if pid is None or qid is None or tcol_p is None:
            return None, None, "packet join failed (missing id/target cols)"

        target_map: Dict[str, float] = {}
        for row in pr:
            k = (row.get(qid, "") or "").strip()
            v = _try_float(row.get(tcol_p, ""))
            if k and v is not None:
                target_map[k] = float(v)

        y_list = []
        yhat_list = []
        hit = 0
        for row in rows:
            k = (row.get(pid, "") or "").strip()
            pv = _try_float(row.get(pred_col, ""))
            tv = target_map.get(k, None)
            if k and pv is not None and tv is not None:
                y_list.append(tv)
                yhat_list.append(pv)
                hit += 1

        if hit < 3:
            return None, None, "packet join got too few matches"

        return np.array(y_list, dtype=float), np.array(yhat_list, dtype=float), f"joined via {pid}={qid}, target={tcol_p}, pred={pred_col}"

    return None, None, "could not find usable columns"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = find_experiment_root(script_dir)

    pause_dirs = sorted([p for p in root.rglob("pause_*") if p.is_dir()])
    if not pause_dirs:
        print(f"[error] no pause_* folders found under: {root}")
        return

    made = 0
    skipped = 0

    for pause_dir in pause_dirs:
        concat_csv = pause_dir / "global_mos_concat.csv"
        rm_csv = pause_dir / "global_mos_running_mean.csv"

        # only do folders that actually have the expected files
        if not concat_csv.exists() and not rm_csv.exists():
            continue

        # build a short label from path parts
        rel = pause_dir.relative_to(root)
        parts = rel.parts
        segment = parts[0] if len(parts) > 0 else "segment"
        sys_pair = parts[1] if len(parts) > 1 else "sys_pair"
        pause_name = pause_dir.name

        if concat_csv.exists():
            y, yhat, info = extract_y_yhat_from_global(concat_csv)
            if y is None or yhat is None:
                print(f"[skip] {segment}/{sys_pair}/{pause_name} concat: {info}")
                skipped += 1
            else:
                out_png = pause_dir / "plot_global_mos_concat.png"
                scatter_plot(y, yhat, f"{segment} {sys_pair} {pause_name} [concat]", out_png)
                made += 1

        if rm_csv.exists():
            y, yhat, info = extract_y_yhat_from_global(rm_csv)
            if y is None or yhat is None:
                print(f"[skip] {segment}/{sys_pair}/{pause_name} running_mean: {info}")
                skipped += 1
            else:
                out_png = pause_dir / "plot_global_mos_running_mean.png"
                scatter_plot(y, yhat, f"{segment} {sys_pair} {pause_name} [running_mean]", out_png)
                made += 1

    print(f"[ok] root: {root}")
    print(f"[ok] plots written: {made}")
    if skipped:
        print(f"[warn] skipped: {skipped} (check printed reasons)")


if __name__ == "__main__":
    main()
