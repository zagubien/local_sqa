from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


RESULTS_TTS_DIRNAME = "results_tts"

# which files to read in each forward/ folder
FILES = {
    "concat": ("block_mos_concat.csv", "mos_pred_block_concat"),
    "running_mean": ("block_mos_running_mean.csv", "mos_pred_block_rm_durw"),
}

# ignore folders like block_5_alt
IGNORE_SUFFIX = "_alt"

OUT_CSV_NAME = "needle_eer_results.csv"
OUT_TXT_NAME = "needle_eer_results.txt"


def _safe_interp(x: np.ndarray, y: np.ndarray):
    return interp1d(
        x,
        y,
        bounds_error=False,
        fill_value=(y[0], y[-1]) if len(y) else (np.nan, np.nan),
        assume_sorted=False,
    )


def get_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    # labels: 1 = good (no needle), 0 = bad (needle)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    f_tpr = _safe_interp(fpr, tpr)
    eer = brentq(lambda x: 1.0 - x - float(f_tpr(x)), 0.0, 1.0)

    f_th = _safe_interp(fpr, thresholds)
    eer_th = float(f_th(eer))
    return float(eer), float(eer_th)


def classify_stats(scores: np.ndarray, labels: np.ndarray, th: float) -> Dict[str, float]:
    # predict bad if score < th
    pred_good = (scores >= th).astype(np.int64)  # 1=good
    pred_bad = 1 - pred_good                      # 1=bad

    true_good = labels
    true_bad = 1 - labels

    n_good = float(np.sum(true_good == 1))
    n_bad = float(np.sum(true_bad == 1))

    miss = float(np.sum((true_bad == 1) & (pred_good == 1)))  # bad predicted good
    fa = float(np.sum((true_good == 1) & (pred_bad == 1)))    # good predicted bad

    miss_rate = miss / n_bad if n_bad > 0 else float("nan")
    fa_rate = fa / n_good if n_good > 0 else float("nan")
    acc = float(np.mean(pred_good == true_good)) if len(scores) else float("nan")

    return {
        "n_blocks": float(len(scores)),
        "n_good": n_good,
        "n_bad": n_bad,
        "acc_at_th": acc,
        "miss_rate_at_th": miss_rate,
        "fa_rate_at_th": fa_rate,
    }


def parse_block_seconds(block_dir_name: str) -> Optional[int]:
    # "block_20" -> 20
    if not block_dir_name.startswith("block_"):
        return None
    tail = block_dir_name[len("block_"):]
    if not tail.isdigit():
        return None
    return int(tail)


def find_results_root(script_path: Path) -> Path:
    # assume the script is inside repo and results_tts is somewhere above or next to it
    # first: sibling
    sib = script_path.parent / RESULTS_TTS_DIRNAME
    if sib.exists():
        return sib.resolve()

    # then: walk upwards
    cur = script_path.resolve().parent
    for _ in range(8):
        cand = cur / RESULTS_TTS_DIRNAME
        if cand.exists():
            return cand.resolve()
        cur = cur.parent

    raise RuntimeError(f"could not find '{RESULTS_TTS_DIRNAME}' near {script_path}")


def iter_forward_dirs(results_root: Path) -> List[Tuple[Path, int, str, str]]:
    # yields (forward_dir, block_seconds, dataset, system_id)
    out: List[Tuple[Path, int, str, str]] = []
    for block_dir in sorted(results_root.iterdir()):
        if not block_dir.is_dir():
            continue
        name = block_dir.name
        if not name.startswith("block_"):
            continue
        if name.endswith(IGNORE_SUFFIX):
            continue

        block_seconds = parse_block_seconds(name)
        if block_seconds is None:
            continue

        for dataset_dir in sorted(block_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name  # "bvcc" or "somos"

            for system_dir in sorted(dataset_dir.iterdir()):
                if not system_dir.is_dir():
                    continue
                system_id = system_dir.name

                forward_dir = system_dir / "forward"
                if forward_dir.is_dir():
                    out.append((forward_dir, block_seconds, dataset, system_id))
    return out


def read_scores_labels(csv_path: Path, score_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    if "has_needle" not in df.columns:
        raise RuntimeError(f"{csv_path} missing 'has_needle'")

    if score_col not in df.columns:
        raise RuntimeError(f"{csv_path} missing '{score_col}'")

    scores = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=np.float64)
    has_needle = pd.to_numeric(df["has_needle"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

    m = np.isfinite(scores)
    scores = scores[m]
    has_needle = has_needle[m]

    labels = 1 - np.clip(has_needle, 0, 1)  # 1=good, 0=bad
    return scores, labels.astype(np.int64)


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.6f}"


def main() -> None:
    script_path = Path(__file__).resolve()
    results_root = find_results_root(script_path)

    rows: List[Dict[str, object]] = []
    problems: List[str] = []

    forward_dirs = iter_forward_dirs(results_root)
    if not forward_dirs:
        raise RuntimeError(f"no forward dirs found under {results_root}")

    for forward_dir, block_seconds, dataset, system_id in forward_dirs:
        for method, (fname, score_col) in FILES.items():
            csv_path = forward_dir / fname
            if not csv_path.exists():
                problems.append(f"[missing] {csv_path}")
                continue

            try:
                scores, labels = read_scores_labels(csv_path, score_col=score_col)
            except Exception as e:
                problems.append(f"[bad csv] {csv_path} -> {e}")
                continue

            if len(scores) < 2 or len(np.unique(labels)) < 2:
                problems.append(f"[skip one-class] {csv_path} (n={len(scores)} classes={sorted(set(labels.tolist()))})")
                continue

            try:
                eer, eer_th = get_eer(scores, labels)
                stats = classify_stats(scores, labels, eer_th)
            except Exception as e:
                problems.append(f"[eer fail] {csv_path} -> {e}")
                continue

            rows.append({
                "block_seconds": block_seconds,
                "dataset": dataset,
                "system_id": system_id,
                "method": method,
                "csv": str(csv_path.relative_to(results_root)),
                "eer": eer,
                "eer_th": eer_th,
                **stats,
            })

    if not rows:
        raise RuntimeError("no usable rows computed (check problems in output txt)")

    df = pd.DataFrame(rows).sort_values(["block_seconds", "dataset", "system_id", "method"])
    out_csv = results_root / OUT_CSV_NAME
    df.to_csv(out_csv, index=False)

    # aggregate summary for txt
    def agg_mean(s: pd.Series) -> float:
        v = pd.to_numeric(s, errors="coerce")
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    group_cols = ["block_seconds", "dataset", "method"]
    df_sum = df.groupby(group_cols, as_index=False).agg({
        "eer": agg_mean,
        "acc_at_th": agg_mean,
        "miss_rate_at_th": agg_mean,
        "fa_rate_at_th": agg_mean,
        "n_blocks": "sum",
        "n_good": "sum",
        "n_bad": "sum",
    }).sort_values(group_cols)

    out_txt = results_root / OUT_TXT_NAME
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("needle in the haystack: EER evaluation\n")
        f.write(f"results_root: {results_root}\n")
        f.write(f"ignored: block_*{IGNORE_SUFFIX}\n\n")

        f.write("summary (mean over systems)\n")
        f.write("block_s  dataset  method         eer      acc     miss     fa     n_blocks  n_good  n_bad\n")
        f.write("-" * 90 + "\n")
        for _, r in df_sum.iterrows():
            f.write(
                f"{int(r['block_seconds']):7d}  "
                f"{str(r['dataset']):7s}  "
                f"{str(r['method']):12s}  "
                f"{fmt(float(r['eer'])):>7s}  "
                f"{fmt(float(r['acc_at_th'])):>7s}  "
                f"{fmt(float(r['miss_rate_at_th'])):>7s}  "
                f"{fmt(float(r['fa_rate_at_th'])):>7s}  "
                f"{int(r['n_blocks']):8d}  "
                f"{int(r['n_good']):6d}  "
                f"{int(r['n_bad']):5d}\n"
            )

        f.write("\nper system (raw)\n")
        f.write(df.to_string(index=False))
        f.write("\n\nproblems / skipped\n")
        if problems:
            for p in problems:
                f.write(p + "\n")
        else:
            f.write("(none)\n")

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_txt}")


if __name__ == "__main__":
    main()