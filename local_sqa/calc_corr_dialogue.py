from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr, kendalltau  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


SMOOTH_WIN = 9 

CONCAT_FILE = "global_mos_concat.csv"
RM_FILE = "global_mos_running_mean.csv"

OUT_TARGET = "two_speaker_target_corrs.csv"
OUT_CR = "two_speaker_concat_vs_rm_corrs.csv"
OUT_REPORT = "two_speaker_correlations_report.txt"

OUT_TARGET_LONG = "two_speaker_target_long.csv"
OUT_PAIRWISE = "two_speaker_pairwise_deltas_to_target.csv"
OUT_WINCOUNTS = "two_speaker_win_counts_to_target.csv"
OUT_FAILURES = "two_speaker_failures.csv"

# mapping you gave (train-id -> label)
TRAIN_LABEL = {
    19: "w2v2-large + Transformer",
    18: "w2v2-base + Transformer",
    23: "w2v2-large + BiLSTM",
    24: "w2v2-base + BiLSTM",
    27: "w2v2-large + Conv",
    26: "w2v2-base + Conv",
}

_TWO_RE = re.compile(r"^two_speaker_(\d+)$")


# ---------------- basics ----------------
def find_results_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "results",
        script_dir.parent / "results",
        script_dir.parent.parent / "results",
        Path.cwd() / "results",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    raise FileNotFoundError(f"could not find results/ (tried: {[str(c) for c in candidates]})")


def _odd_window(n: int, win: int) -> int:
    if win <= 0:
        return 0
    w = int(win)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if (n % 2 == 1) else max(1, n - 1)
    if w < 3:
        return 0
    return w


def smooth_centered(y: np.ndarray, win: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    w = _odd_window(len(y), win)
    if w == 0:
        return y.copy()
    return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)


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


# ---------------- structure iter ----------------
def iter_two_speaker_roots(results_root: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        m = _TWO_RE.match(d.name)
        if m:
            out.append((int(m.group(1)), d))
    return sorted(out, key=lambda x: x[0])


def iter_cases(train_root: Path):
    # train_root/{five_sentences,three_sentences,one_sentence}/{pair}/{pause_*}/csvs
    for sent_dir in sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        # accept only these if present, else accept any *_sentence*
        if not (sent_dir.name in {"five_sentences", "three_sentences", "one_sentence"} or "sentence" in sent_dir.name):
            continue

        for pair_dir in sorted([p for p in sent_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            if not (pair_dir.name.startswith("sys") or pair_dir.name.startswith("rev_") or pair_dir.name.startswith("rand_")):
                continue

            for pause_dir in sorted([p for p in pair_dir.iterdir() if p.is_dir() and p.name.startswith("pause_")], key=lambda p: p.name):
                c = pause_dir / CONCAT_FILE
                r = pause_dir / RM_FILE
                if c.exists() and r.exists():
                    yield sent_dir.name, pair_dir.name, pause_dir


def parse_meta(train_id: int, train_root: Path, sentences: str, pair: str, pause_dir: Path) -> Dict:
    mode = "forward"
    base_pair = pair
    if pair.startswith("rev_"):
        mode = "reverse"
        base_pair = pair[len("rev_"):]
    elif pair.startswith("rand_"):
        mode = "random"
        base_pair = pair[len("rand_"):]

    return {
        "train": int(train_id),
        "train_label": TRAIN_LABEL.get(int(train_id), f"train{train_id}"),
        "two_root": train_root.name,
        "sentences": sentences,
        "pair": pair,
        "base_pair": base_pair,
        "mode": mode,
        "pause_kind": pause_dir.name,
        "rel_path": str(pause_dir.relative_to(train_root)),
        "abs_path": str(pause_dir),
    }


# ---------------- core computation ----------------
def _series_by_idx(df: pd.DataFrame, idx_col: str, val_col: str) -> Tuple[np.ndarray, np.ndarray]:
    d = df[[idx_col, val_col]].copy()
    d[idx_col] = pd.to_numeric(d[idx_col], errors="coerce")
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna()
    if d.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    g = d.groupby(idx_col, as_index=False)[val_col].mean()
    g = g.sort_values(idx_col).reset_index(drop=True)
    idx = g[idx_col].to_numpy(dtype=np.int64)
    val = g[val_col].to_numpy(dtype=np.float64)
    return idx, val


def compute_one(pause_dir: Path) -> Tuple[Dict, Dict]:
    df_c = pd.read_csv(pause_dir / CONCAT_FILE)
    df_r = pd.read_csv(pause_dir / RM_FILE)

    # expected fixed columns (your data is consistent)
    need_c = {"idx", "mos_pred_concat", "mos_target_concat_durw"}
    need_r = {"idx", "mos_pred_rm_durw", "mos_target_rm_durw"}

    if not need_c.issubset(set(df_c.columns)):
        raise RuntimeError(f"concat csv missing cols: {sorted(list(need_c - set(df_c.columns)))}")
    if not need_r.issubset(set(df_r.columns)):
        raise RuntimeError(f"rm csv missing cols: {sorted(list(need_r - set(df_r.columns)))}")

    ic, c_pred = _series_by_idx(df_c, "idx", "mos_pred_concat")
    _,  c_tgt  = _series_by_idx(df_c, "idx", "mos_target_concat_durw")

    ir, r_pred = _series_by_idx(df_r, "idx", "mos_pred_rm_durw")
    _,  r_tgt  = _series_by_idx(df_r, "idx", "mos_target_rm_durw")

    if len(ic) < 2 or len(ir) < 2:
        raise RuntimeError("too few rows after cleaning")

    # align pred/tgt within each grid by idx intersection
    def _align(a_idx, a, b_idx, b):
        m = pd.merge(
            pd.DataFrame({"idx": a_idx, "a": a}),
            pd.DataFrame({"idx": b_idx, "b": b}),
            on="idx",
            how="inner",
        )
        return m["a"].to_numpy(dtype=np.float64), m["b"].to_numpy(dtype=np.float64), int(len(m))

    c_pred2, c_tgt2, n_c = _align(ic, c_pred, ic, c_tgt)
    r_pred2, r_tgt2, n_r = _align(ir, r_pred, ir, r_tgt)

    c_filt = smooth_centered(c_pred2, SMOOTH_WIN)
    r_filt = smooth_centered(r_pred2, SMOOTH_WIN)

    c_p_raw, c_s_raw, c_k_raw, c_n_raw = corr_triplet(c_pred2, c_tgt2)
    c_p_fil, c_s_fil, c_k_fil, c_n_fil = corr_triplet(c_filt, c_tgt2)

    r_p_raw, r_s_raw, r_k_raw, r_n_raw = corr_triplet(r_pred2, r_tgt2)
    r_p_fil, r_s_fil, r_k_fil, r_n_fil = corr_triplet(r_filt, r_tgt2)

    target_row = {
        "concat_PCC_raw": c_p_raw, "concat_SRCC_raw": c_s_raw, "concat_KTAU_raw": c_k_raw, "concat_n_raw": c_n_raw,
        "concat_PCC_filt": c_p_fil, "concat_SRCC_filt": c_s_fil, "concat_KTAU_filt": c_k_fil, "concat_n_filt": c_n_fil,
        "concat_dPCC": (c_p_fil - c_p_raw) if np.isfinite(c_p_fil) and np.isfinite(c_p_raw) else np.nan,
        "concat_dSRCC": (c_s_fil - c_s_raw) if np.isfinite(c_s_fil) and np.isfinite(c_s_raw) else np.nan,
        "concat_dKTAU": (c_k_fil - c_k_raw) if np.isfinite(c_k_fil) and np.isfinite(c_k_raw) else np.nan,

        "rm_PCC_raw": r_p_raw, "rm_SRCC_raw": r_s_raw, "rm_KTAU_raw": r_k_raw, "rm_n_raw": r_n_raw,
        "rm_PCC_filt": r_p_fil, "rm_SRCC_filt": r_s_fil, "rm_KTAU_filt": r_k_fil, "rm_n_filt": r_n_fil,
        "rm_dPCC": (r_p_fil - r_p_raw) if np.isfinite(r_p_fil) and np.isfinite(r_p_raw) else np.nan,
        "rm_dSRCC": (r_s_fil - r_s_raw) if np.isfinite(r_s_fil) and np.isfinite(r_s_raw) else np.nan,
        "rm_dKTAU": (r_k_fil - r_k_raw) if np.isfinite(r_k_fil) and np.isfinite(r_k_raw) else np.nan,

        "smooth_win": int(SMOOTH_WIN),
        "n_concat_aligned": int(n_c),
        "n_rm_aligned": int(n_r),
    }

    # concat vs rm (align by idx intersection)
    a_raw, b_raw, n_ab = _align(ic, c_pred, ir, r_pred)
    a_fil = smooth_centered(a_raw, SMOOTH_WIN)
    b_fil = smooth_centered(b_raw, SMOOTH_WIN)

    cr_p_raw, cr_s_raw, cr_k_raw, cr_n_raw = corr_triplet(a_raw, b_raw)
    cr_p_fil, cr_s_fil, cr_k_fil, cr_n_fil = corr_triplet(a_fil, b_fil)

    cr_row = {
        "cr_PCC_raw": cr_p_raw, "cr_SRCC_raw": cr_s_raw, "cr_KTAU_raw": cr_k_raw, "cr_n_raw": cr_n_raw,
        "cr_PCC_filt": cr_p_fil, "cr_SRCC_filt": cr_s_fil, "cr_KTAU_filt": cr_k_fil, "cr_n_filt": cr_n_fil,
        "cr_dPCC": (cr_p_fil - cr_p_raw) if np.isfinite(cr_p_fil) and np.isfinite(cr_p_raw) else np.nan,
        "cr_dSRCC": (cr_s_fil - cr_s_raw) if np.isfinite(cr_s_fil) and np.isfinite(cr_s_raw) else np.nan,
        "cr_dKTAU": (cr_k_fil - cr_k_raw) if np.isfinite(cr_k_fil) and np.isfinite(cr_k_raw) else np.nan,
        "smooth_win": int(SMOOTH_WIN),
        "n_idx_intersection": int(n_ab),
    }

    return target_row, cr_row


# ---------------- long + comparisons ----------------
def target_long(df_t: pd.DataFrame) -> pd.DataFrame:
    if df_t.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df_t.iterrows():
        base = {
            "train": int(r["train"]),
            "train_label": r["train_label"],
            "sentences": r["sentences"],
            "base_pair": r["base_pair"],
            "mode": r["mode"],
            "pause_kind": r["pause_kind"],
        }
        rows.append({
            **base,
            "grid": "concat",
            "PCC_raw": r["concat_PCC_raw"], "SRCC_raw": r["concat_SRCC_raw"], "KTAU_raw": r["concat_KTAU_raw"], "n_raw": r["concat_n_raw"],
            "PCC_filt": r["concat_PCC_filt"], "SRCC_filt": r["concat_SRCC_filt"], "KTAU_filt": r["concat_KTAU_filt"], "n_filt": r["concat_n_filt"],
            "dPCC": r["concat_dPCC"], "dSRCC": r["concat_dSRCC"], "dKTAU": r["concat_dKTAU"],
        })
        rows.append({
            **base,
            "grid": "rm",
            "PCC_raw": r["rm_PCC_raw"], "SRCC_raw": r["rm_SRCC_raw"], "KTAU_raw": r["rm_KTAU_raw"], "n_raw": r["rm_n_raw"],
            "PCC_filt": r["rm_PCC_filt"], "SRCC_filt": r["rm_SRCC_filt"], "KTAU_filt": r["rm_KTAU_filt"], "n_filt": r["rm_n_filt"],
            "dPCC": r["rm_dPCC"], "dSRCC": r["rm_dSRCC"], "dKTAU": r["rm_dKTAU"],
        })
    return pd.DataFrame(rows)


def win_counts(df_long: pd.DataFrame, metric: str, case_cols: List[str]) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()

    def _best(g: pd.DataFrame) -> pd.Series:
        g2 = g.dropna(subset=[metric]).copy()
        if g2.empty:
            return pd.Series({"best_train": np.nan})
        idx = g2[metric].astype(float).idxmax()
        return pd.Series({"best_train": int(g2.loc[idx, "train"])})

    best = df_long.groupby(case_cols, dropna=False).apply(_best).reset_index()
    best = best.dropna(subset=["best_train"])
    best["best_train"] = best["best_train"].astype(int)
    cnt = best.groupby(case_cols[:-1] + ["best_train"]).size().reset_index(name="count")  # case cols include grid at end
    cnt["best_label"] = cnt["best_train"].map(lambda t: TRAIN_LABEL.get(int(t), str(t)))
    return cnt.sort_values(["count"], ascending=False).reset_index(drop=True)


def pairwise_deltas(df_long: pd.DataFrame, metric: str, case_cols: List[str]) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()

    pv = df_long.pivot_table(index=case_cols, columns="train", values=metric, aggfunc="first")
    trains = sorted([c for c in pv.columns if pd.api.types.is_number(c)])

    out_rows = []
    for i in range(len(trains)):
        for j in range(i + 1, len(trains)):
            a = trains[i]
            b = trains[j]
            xa = pv[a].to_numpy()
            xb = pv[b].to_numpy()
            m = np.isfinite(xa) & np.isfinite(xb)
            if int(np.sum(m)) == 0:
                continue
            d = xb[m] - xa[m]
            out_rows.append({
                "metric": metric,
                "train_a": int(a),
                "train_b": int(b),
                "label_a": TRAIN_LABEL.get(int(a), str(a)),
                "label_b": TRAIN_LABEL.get(int(b), str(b)),
                "n": int(np.sum(m)),
                "mean_delta": float(np.nanmean(d)),
                "median_delta": float(np.nanmedian(d)),
                "wins_b": int(np.sum(d > 1e-9)),
                "wins_a": int(np.sum(d < -1e-9)),
                "ties": int(np.sum(np.abs(d) <= 1e-9)),
            })
    return pd.DataFrame(out_rows).sort_values(["mean_delta"], ascending=False).reset_index(drop=True)


def write_section(f, title: str, df_long: pd.DataFrame, df_cr: pd.DataFrame):
    f.write(f"====================\n{title}\n====================\n\n")

    if df_long.empty:
        f.write("(no data)\n\n")
        return

    # 1) mean/median SRCC_filt to target (by train, grid)
    f.write("1) mean/median SRCC_filt to target (by train, grid)\n")
    for grid in ["concat", "rm"]:
        sub = df_long[df_long["grid"] == grid].copy()
        g = sub.groupby(["train", "train_label"])["SRCC_filt"].agg(["mean", "median", "count"]).reset_index()
        g = g.sort_values(["mean"], ascending=False)
        f.write(f"\n  grid={grid}\n")
        for _, r in g.iterrows():
            f.write(
                f"    train{int(r['train'])} {r['train_label']}: mean={_fmt(r['mean'])} median={_fmt(r['median'])} n={int(r['count'])}\n"
            )
    f.write("\n")

    # 2) win counts (best SRCC_filt) per train (by grid)
    f.write("2) win counts (best SRCC_filt) per train (by grid)\n")
    case_cols = ["sentences", "base_pair", "mode", "pause_kind", "grid"]
    wc = win_counts(df_long, "SRCC_filt", case_cols)
    for grid in ["concat", "rm"]:
        f.write(f"\n  grid={grid}\n")
        w2 = wc[wc["grid"] == grid].copy() if "grid" in wc.columns else pd.DataFrame()
        if w2.empty:
            f.write("    (no win-count data)\n")
        else:
            # group by train across cases
            gg = w2.groupby(["best_train", "best_label"])["count"].sum().reset_index().sort_values("count", ascending=False)
            for _, r in gg.iterrows():
                f.write(f"    {r['best_label']} (train{int(r['best_train'])}): {int(r['count'])}\n")
    f.write("\n")

    # 3) pairwise mean deltas (SRCC_filt to target)
    f.write("3) pairwise mean deltas (SRCC_filt to target)  (train_b - train_a)\n")
    pw = pairwise_deltas(df_long, "SRCC_filt", ["sentences", "base_pair", "mode", "pause_kind", "grid"])
    if pw.empty:
        f.write("(no pairwise deltas)\n\n")
    else:
        for _, r in pw.iterrows():
            f.write(
                f"  {r['label_b']} (train{int(r['train_b'])}) - {r['label_a']} (train{int(r['train_a'])}): "
                f"mean={_fmt(r['mean_delta'])} median={_fmt(r['median_delta'])} n={int(r['n'])} "
                f"wins_b={int(r['wins_b'])} wins_a={int(r['wins_a'])} ties={int(r['ties'])}\n"
            )
        f.write("\n")

    # 4) concat-vs-rm SRCC_filt (by train)
    f.write("4) concat-vs-rm SRCC_filt (by train)\n")
    if df_cr.empty:
        f.write("(no concat-vs-rm rows)\n\n")
    else:
        g2 = df_cr.groupby(["train", "train_label"])["cr_SRCC_filt"].agg(["mean", "median", "count"]).reset_index()
        g2 = g2.sort_values(["mean"], ascending=False)
        for _, r in g2.iterrows():
            f.write(
                f"  train{int(r['train'])} {r['train_label']}: mean={_fmt(r['mean'])} median={_fmt(r['median'])} n={int(r['count'])}\n"
            )
        f.write("\n")


def main() -> None:
    results_root = find_results_root()
    two_roots = iter_two_speaker_roots(results_root)
    if not two_roots:
        raise FileNotFoundError(f"no two_speaker_* folders under: {results_root}")

    target_rows: List[Dict] = []
    cr_rows: List[Dict] = []
    fail_rows: List[Dict] = []

    for train_id, train_root in two_roots:
        for sentences, pair, pause_dir in iter_cases(train_root):
            meta = parse_meta(train_id, train_root, sentences, pair, pause_dir)
            try:
                trow, crow = compute_one(pause_dir)
                target_rows.append({**meta, **trow})
                cr_rows.append({**meta, **crow})
            except Exception as e:
                fail_rows.append({**meta, "error_type": type(e).__name__, "error": str(e)})

    df_t = pd.DataFrame(target_rows)
    df_cr = pd.DataFrame(cr_rows)
    df_fail = pd.DataFrame(fail_rows)

    # main outputs (like "üblich")
    (results_root / OUT_TARGET).write_text(df_t.to_csv(index=False) if not df_t.empty else "", encoding="utf-8")
    (results_root / OUT_CR).write_text(df_cr.to_csv(index=False) if not df_cr.empty else "", encoding="utf-8")
    (results_root / OUT_FAILURES).write_text(df_fail.to_csv(index=False) if not df_fail.empty else "", encoding="utf-8")

    # long + comparison tables
    df_long = target_long(df_t)
    (results_root / OUT_TARGET_LONG).write_text(df_long.to_csv(index=False) if not df_long.empty else "", encoding="utf-8")

    pw = pairwise_deltas(df_long, "SRCC_filt", ["sentences", "base_pair", "mode", "pause_kind", "grid"]) if not df_long.empty else pd.DataFrame()
    (results_root / OUT_PAIRWISE).write_text(pw.to_csv(index=False) if not pw.empty else "", encoding="utf-8")

    wc = win_counts(df_long, "SRCC_filt", ["sentences", "base_pair", "mode", "pause_kind", "grid"]) if not df_long.empty else pd.DataFrame()
    (results_root / OUT_WINCOUNTS).write_text(wc.to_csv(index=False) if not wc.empty else "", encoding="utf-8")

    # report txt (like your BVCC/SOMOS one, but sections by sentence-set + overall)
    report_path = results_root / OUT_REPORT
    with report_path.open("w", encoding="utf-8") as f:
        f.write("two_speaker correlations (all trainings)\n")
        f.write(f"smooth_win={SMOOTH_WIN} (0 disables)\n\n")

        if df_long.empty:
            f.write("(no data)\n")
        else:
            # overall
            write_section(f, "two_speaker (all)", df_long, df_cr)

            # per sentence-set
            for s in ["one_sentence", "three_sentences", "five_sentences"]:
                sub_long = df_long[df_long["sentences"] == s].copy()
                sub_cr = df_cr[df_cr["sentences"] == s].copy()
                if not sub_long.empty or not sub_cr.empty:
                    write_section(f, s, sub_long, sub_cr)

        if not df_fail.empty:
            f.write("FAILURES\n")
            vc = df_fail["error_type"].value_counts()
            for k, v in vc.items():
                f.write(f"  {k}: {int(v)}\n")
            f.write("\nfirst 50 failures:\n")
            for _, r in df_fail.head(50).iterrows():
                f.write(f"  {r['two_root']}/{r['rel_path']}: {r['error_type']}: {r['error']}\n")

    print(f"[done] wrote: {results_root / OUT_TARGET}")
    print(f"[done] wrote: {results_root / OUT_CR}")
    print(f"[done] wrote: {results_root / OUT_REPORT}")
    print(f"[done] wrote: {results_root / OUT_TARGET_LONG}")
    print(f"[done] wrote: {results_root / OUT_PAIRWISE}")
    print(f"[done] wrote: {results_root / OUT_WINCOUNTS}")
    print(f"[done] wrote: {results_root / OUT_FAILURES}")
    if not df_fail.empty:
        print(f"[warn] failures: {len(df_fail)}")


if __name__ == "__main__":
    main()
