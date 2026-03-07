from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr, kendalltau  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -------------------------
# config
# -------------------------

TRAIN_IDS = {18, 19, 23, 24, 26, 27}

VARIANTS = {"original", "no_pause", "long_pause", "long_pause_noise"}

CSV_GLOBAL_CONCAT = "global_mos_concat.csv"
CSV_GLOBAL_RM = "global_mos_running_mean.csv"
CSV_BLOCK_CONCAT = "block_mos_concat.csv"
CSV_BLOCK_RM = "block_mos_running_mean.csv"

OUT_TARGET = "dialogue_corrs_target.csv"
OUT_CR = "dialogue_corrs_concat_vs_rm.csv"
OUT_FAIL = "dialogue_corrs_failures.csv"
OUT_REPORT = "dialogue_correlations_report.txt"

TRAIN_LABEL = {
    19: "w2v2-large + Transformer",
    18: "w2v2-base + Transformer",
    23: "w2v2-large + BiLSTM",
    24: "w2v2-base + BiLSTM",
    27: "w2v2-large + Conv",
    26: "w2v2-base + Conv",
}

DIALOGUE_BLOCK_RE = re.compile(r"^dialogue_block(?P<blk>\d+)$")
TRAIN_RE = re.compile(r"^train(?P<t>\d+)$")


# -------------------------
# correlation helpers
# -------------------------

def _drop_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _drop_pair(a, b)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    if _HAS_SCIPY:
        return float(pearsonr(a, b)[0])
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _drop_pair(a, b)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    if _HAS_SCIPY:
        return float(spearmanr(a, b).correlation)
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _drop_pair(a, b)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    if _HAS_SCIPY:
        return float(kendalltau(a, b).correlation)
    try:
        return float(pd.Series(a).corr(pd.Series(b), method="kendall"))
    except Exception:
        return np.nan


def corr_triplet(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, int]:
    a2, b2 = _drop_pair(a, b)
    n = int(len(a2))
    return _pearson(a2, b2), _spearman(a2, b2), _kendall(a2, b2), n


def _fmt(x, nd=6) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "nan"
    return f"{float(x):.{nd}f}"


# -------------------------
# locating / traversal
# -------------------------

def find_results_block_dialogue_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "results_block_dialogue",
        script_dir.parent / "results_block_dialogue",
        Path.cwd() / "results_block_dialogue",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    raise FileNotFoundError("could not find results_block_dialogue (next to script / parent / cwd)")


def iter_cases(root: Path) -> Iterable[Dict]:
    """
    yields dict with:
      block_s, dataset, train, train_label, pair, mode, variant, case_dir (Path)
    """
    for blk_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        mblk = DIALOGUE_BLOCK_RE.match(blk_dir.name)
        if not mblk:
            continue
        block_s = int(mblk.group("blk"))

        for dataset_dir in sorted([p for p in blk_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            dataset = dataset_dir.name.lower()
            if dataset not in ("bvcc", "somos"):
                continue

            for train_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                mt = TRAIN_RE.match(train_dir.name)
                if not mt:
                    continue
                train_id = int(mt.group("t"))
                if train_id not in TRAIN_IDS:
                    continue

                for pair_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                    if not (pair_dir.name.startswith("dlg_") or pair_dir.name.startswith("rev_dlg_") or pair_dir.name.startswith("rand_dlg_")):
                        continue

                    mode = "forward"
                    base_pair = pair_dir.name
                    if base_pair.startswith("rev_dlg_"):
                        mode = "reverse"
                        base_pair = base_pair[len("rev_"):]  # keep dlg_...
                    elif base_pair.startswith("rand_dlg_"):
                        mode = "random"
                        base_pair = base_pair[len("rand_"):]  # keep dlg_...

                    for var_dir in sorted([p for p in pair_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                        if var_dir.name not in VARIANTS:
                            continue

                        yield {
                            "block_s": block_s,
                            "dataset": dataset,
                            "train": train_id,
                            "train_label": TRAIN_LABEL.get(train_id, f"train{train_id}"),
                            "pair": pair_dir.name,
                            "base_pair": base_pair,
                            "mode": mode,
                            "variant": var_dir.name,
                            "case_dir": var_dir.resolve(),
                            "rel_path": str(var_dir.relative_to(root)),
                        }


# -------------------------
# CSV reading and alignment
# -------------------------

def pick_first_existing(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing column for {what}. have={list(df.columns)} candidates={candidates}")


def _read_global_concat(p: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(p)
    x = pick_first_existing(df, ["seconds", "elapsed_s", "time_s"], "global concat x")
    y = pick_first_existing(df, ["mos_pred_concat", "mos_pred", "mos"], "global concat pred")
    t = pick_first_existing(df, ["mos_target_concat_durw", "mos_target_concat", "mos_target"], "global concat target")
    df = df.sort_values(x).reset_index(drop=True)
    return (
        pd.to_numeric(df[x], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64),
    )


def _read_global_rm(p: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(p)
    x = pick_first_existing(df, ["elapsed_s", "seconds", "time_s"], "global rm x")
    y = pick_first_existing(df, ["mos_pred_rm_durw", "mos_pred_rm", "mos_pred", "mos"], "global rm pred")
    t = pick_first_existing(df, ["mos_target_rm_durw", "mos_target_rm", "mos_target"], "global rm target")
    df = df.sort_values(x).reset_index(drop=True)
    return (
        pd.to_numeric(df[x], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64),
    )


def _ensure_block_mid(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "block_mid_s" in df.columns:
        return df, "block_mid_s"
    if "block_start_s" in df.columns and "block_end_s" in df.columns:
        df = df.copy()
        df["block_mid_s"] = 0.5 * (
            pd.to_numeric(df["block_start_s"], errors="coerce")
            + pd.to_numeric(df["block_end_s"], errors="coerce")
        )
        return df, "block_mid_s"
    x = pick_first_existing(df, ["block_mid", "mid_s", "mid"], "block x")
    return df, x


def _read_block_concat(p: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(p)
    df, x = _ensure_block_mid(df)
    y = pick_first_existing(df, ["mos_pred_block_concat", "mos_pred_block", "mos_pred", "mos"], "block concat pred")
    t = pick_first_existing(df, ["mos_target_block_durw", "mos_target_block", "mos_target"], "block concat target")
    df = df.sort_values(x).reset_index(drop=True)
    return (
        pd.to_numeric(df[x], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64),
    )


def _read_block_rm(p: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(p)
    df, x = _ensure_block_mid(df)
    y = pick_first_existing(df, ["mos_pred_block_rm_durw", "mos_pred_block_rm", "mos_pred_block", "mos_pred", "mos"], "block rm pred")
    t = pick_first_existing(df, ["mos_target_block_durw", "mos_target_block", "mos_target"], "block rm target")
    df = df.sort_values(x).reset_index(drop=True)
    return (
        pd.to_numeric(df[x], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64),
    )


def _align_asof(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    tol_s: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    a = pd.DataFrame({"t": ax, "a": ay}).dropna().sort_values("t")
    b = pd.DataFrame({"t": bx, "b": by}).dropna().sort_values("t")
    if a.empty or b.empty:
        return np.array([]), np.array([]), 0
    m = pd.merge_asof(a, b, on="t", direction="nearest", tolerance=tol_s).dropna()
    return m["a"].to_numpy(dtype=np.float64), m["b"].to_numpy(dtype=np.float64), int(len(m))


def compute_case(case_dir: Path) -> Tuple[Dict, Dict]:
    # returns (target_metrics, concat_vs_rm_metrics)

    target_metrics: Dict[str, float] = {}
    cr_metrics: Dict[str, float] = {}

    # --- global to target ---
    gc = case_dir / CSV_GLOBAL_CONCAT
    gr = case_dir / CSV_GLOBAL_RM
    if gc.exists():
        x, y, t = _read_global_concat(gc)
        p, s, k, n = corr_triplet(y, t)
        target_metrics.update({
            "global_concat_PCC": p, "global_concat_SRCC": s, "global_concat_KTAU": k, "global_concat_n": n
        })
    if gr.exists():
        x, y, t = _read_global_rm(gr)
        p, s, k, n = corr_triplet(y, t)
        target_metrics.update({
            "global_rm_PCC": p, "global_rm_SRCC": s, "global_rm_KTAU": k, "global_rm_n": n
        })

    # --- block to target ---
    bc = case_dir / CSV_BLOCK_CONCAT
    br = case_dir / CSV_BLOCK_RM
    if bc.exists():
        x, y, t = _read_block_concat(bc)
        p, s, k, n = corr_triplet(y, t)
        target_metrics.update({
            "block_concat_PCC": p, "block_concat_SRCC": s, "block_concat_KTAU": k, "block_concat_n": n
        })
    if br.exists():
        x, y, t = _read_block_rm(br)
        p, s, k, n = corr_triplet(y, t)
        target_metrics.update({
            "block_rm_PCC": p, "block_rm_SRCC": s, "block_rm_KTAU": k, "block_rm_n": n
        })

    # --- concat vs rm (pred) ---
    if gc.exists() and gr.exists():
        x1, y1, _t1 = _read_global_concat(gc)
        x2, y2, _t2 = _read_global_rm(gr)
        a, b, n = _align_asof(x1, y1, x2, y2, tol_s=0.2)
        p, s, k, n2 = corr_triplet(a, b)
        cr_metrics.update({
            "global_cr_PCC": p, "global_cr_SRCC": s, "global_cr_KTAU": k, "global_cr_n": n2
        })

    if bc.exists() and br.exists():
        x1, y1, _t1 = _read_block_concat(bc)
        x2, y2, _t2 = _read_block_rm(br)
        # block mids should match, but be safe: inner merge on t
        m = pd.merge(
            pd.DataFrame({"t": x1, "a": y1}),
            pd.DataFrame({"t": x2, "b": y2}),
            on="t",
            how="inner",
        ).dropna()
        a = m["a"].to_numpy(dtype=np.float64)
        b = m["b"].to_numpy(dtype=np.float64)
        p, s, k, n2 = corr_triplet(a, b)
        cr_metrics.update({
            "block_cr_PCC": p, "block_cr_SRCC": s, "block_cr_KTAU": k, "block_cr_n": n2
        })

    return target_metrics, cr_metrics


# -------------------------
# reporting
# -------------------------

def write_report(out_path: Path, df_t: pd.DataFrame, df_cr: pd.DataFrame, df_fail: pd.DataFrame) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("dialogue block correlations\n")
        f.write("metrics: PCC/SRCC/KTAU for pred vs target and concat-vs-rm (pred)\n\n")

        if df_t.empty:
            f.write("(no target rows)\n\n")
        else:
            for ds in ["bvcc", "somos"]:
                sub = df_t[df_t["dataset"] == ds].copy()
                if sub.empty:
                    continue
                f.write(f"=== {ds.upper()} ===\n")
                for blk in sorted(sub["block_s"].unique()):
                    s2 = sub[sub["block_s"] == blk].copy()
                    f.write(f"\n  block{int(blk)}\n")
                    for grid in ["global_concat_SRCC", "global_rm_SRCC", "block_concat_SRCC", "block_rm_SRCC"]:
                        if grid not in s2.columns:
                            continue
                        g = s2.groupby(["train", "train_label"])[grid].agg(["mean", "median", "count"]).reset_index()
                        g = g.sort_values("mean", ascending=False)
                        f.write(f"    {grid}\n")
                        for _, r in g.iterrows():
                            f.write(
                                f"      train{int(r['train'])} {r['train_label']}: "
                                f"mean={_fmt(r['mean'])} median={_fmt(r['median'])} n={int(r['count'])}\n"
                            )

                f.write("\n")

        if not df_cr.empty:
            f.write("=== concat vs rm (pred) SRCC summary ===\n")
            for ds in ["bvcc", "somos"]:
                sub = df_cr[df_cr["dataset"] == ds].copy()
                if sub.empty:
                    continue
                f.write(f"\n  {ds.upper()}\n")
                for blk in sorted(sub["block_s"].unique()):
                    s2 = sub[sub["block_s"] == blk].copy()
                    f.write(f"    block{int(blk)}\n")
                    for grid in ["global_cr_SRCC", "block_cr_SRCC"]:
                        if grid not in s2.columns:
                            continue
                        g = s2.groupby(["train", "train_label"])[grid].agg(["mean", "median", "count"]).reset_index()
                        g = g.sort_values("mean", ascending=False)
                        for _, r in g.iterrows():
                            f.write(
                                f"      {grid} train{int(r['train'])} {r['train_label']}: "
                                f"mean={_fmt(r['mean'])} median={_fmt(r['median'])} n={int(r['count'])}\n"
                            )

        if not df_fail.empty:
            f.write("\n=== failures ===\n")
            vc = df_fail["error_type"].value_counts()
            for k, v in vc.items():
                f.write(f"  {k}: {int(v)}\n")
            f.write("\nfirst 50 failures:\n")
            for _, r in df_fail.head(50).iterrows():
                f.write(f"  {r['rel_path']}: {r['error_type']}: {r['error']}\n")


# -------------------------
# main
# -------------------------

def main() -> None:
    root = find_results_block_dialogue_root()

    target_rows: List[Dict] = []
    cr_rows: List[Dict] = []
    fail_rows: List[Dict] = []

    for meta in iter_cases(root):
        case_dir: Path = meta["case_dir"]

        try:
            trow, crow = compute_case(case_dir)
            if trow:
                target_rows.append({**meta, **trow})
            if crow:
                cr_rows.append({**meta, **crow})
        except Exception as e:
            fail_rows.append({
                **meta,
                "error_type": type(e).__name__,
                "error": str(e),
            })

    df_t = pd.DataFrame(target_rows)
    df_cr = pd.DataFrame(cr_rows)
    df_fail = pd.DataFrame(fail_rows)

    out_t = root / OUT_TARGET
    out_cr = root / OUT_CR
    out_fail = root / OUT_FAIL
    out_rep = root / OUT_REPORT

    df_t.to_csv(out_t, index=False)
    df_cr.to_csv(out_cr, index=False)
    df_fail.to_csv(out_fail, index=False)

    write_report(out_rep, df_t, df_cr, df_fail)

    print(f"[OK] root: {root}")
    print(f"[OK] wrote: {out_t}")
    print(f"[OK] wrote: {out_cr}")
    print(f"[OK] wrote: {out_rep}")
    print(f"[OK] wrote: {out_fail}")
    if not df_fail.empty:
        print(f"[WARN] failures: {len(df_fail)}")


if __name__ == "__main__":
    main()