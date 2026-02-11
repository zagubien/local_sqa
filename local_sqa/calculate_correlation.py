import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
    _HAS_STATS = True
except Exception:
    _HAS_STATS = False


DPI = 300

CONCAT_FILE = "global_mos_concat.csv"
RM_FILE     = "global_mos_running_mean.csv"

OUT_CONCAT  = "filter_global_mos_concat.csv"
OUT_RM      = "filter_global_mos_running_mean.csv"
OUT_PLOT    = "plot_filter.png"

OUT_TXT_TARGET = "correlations_to_target_raw_vs_filtered.txt"
OUT_TXT_CR     = "correlations_concat_vs_rm_raw_vs_filtered.txt"

OUT_TXT_CROSS  = "training_comparison.txt"

VARIANT_DIRS = ["original", "no_pause", "long_pause", "long_pause_noise"]

SKIP_NAMES = {
    "__pycache__", ".git", ".idea", ".vscode",
    "_debug", "packets", "utils", "without_dw"
}

FONT_PATH = "/Users/igorzagubien/Library/Fonts/Karla-VariableFont_wght.ttf"
PROP = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

COL_CONCAT = "#EF3A84"
COL_RM     = "#0025AA"
COL_TGT    = "#111111"
COL_RAW    = "#999999"

# mapping you gave (train-id -> label)
TRAIN_LABEL = {
    19: "w2v2-large + Transformer",
    18: "w2v2-base + Transformer",
    23: "w2v2-large + BiLSTM",
    24: "w2v2-base + BiLSTM",
    27: "w2v2-large + Conv",
    26: "w2v2-base + Conv",
}


def fkw():
    return {"fontproperties": PROP} if PROP else {}


def pick_first_existing(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing column for {what}. have={list(df.columns)} candidates={candidates}")


def pretty_axes(ax):
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.tick_params(labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def find_results_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "results",
        script_dir.parent / "results",
        script_dir.parent.parent / "results",
        script_dir.parent.parent.parent / "results",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(f"could not find results/ folder. tried: {[str(c) for c in candidates]}")


def _dataset_base_and_train(name: str):
    if "_" not in name:
        return None
    base, tail = name.rsplit("_", 1)
    if tail.isdigit():
        return base, int(tail)
    return None


def model_info_from_dataset_root(dataset_root: Path) -> str:
    bt = _dataset_base_and_train(dataset_root.name)
    if not bt:
        return dataset_root.name
    _, tr = bt
    return TRAIN_LABEL.get(tr, dataset_root.name)


def dataset_base_from_root(dataset_root: Path) -> str:
    bt = _dataset_base_and_train(dataset_root.name)
    if not bt:
        return dataset_root.name
    base, _ = bt
    return base


def train_id_from_root(dataset_root: Path) -> int | None:
    bt = _dataset_base_and_train(dataset_root.name)
    if not bt:
        return None
    _, tr = bt
    return tr


def is_root_duplicate_dir(d: Path, dataset_root: Path) -> bool:
    if d.parent != dataset_root:
        return False
    for v in VARIANT_DIRS:
        if (d / v).is_dir():
            return True
    return False


def iter_result_dirs(dataset_root: Path):
    for d in dataset_root.rglob("*"):
        if not d.is_dir():
            continue
        if any(p in SKIP_NAMES for p in d.parts):
            continue
        if is_root_duplicate_dir(d, dataset_root):
            continue

        c1 = d / CONCAT_FILE
        c2 = d / RM_FILE
        if c1.exists() and c2.exists():
            yield d


def _odd_window(n: int, win: int) -> int:
    if n <= 1:
        return 1
    w = int(win)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if (n % 2 == 1) else n - 1
    if w < 3:
        w = 3 if n >= 3 else n
    return w


def smooth_series(y: np.ndarray, win: int = 11, poly: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3:
        return y.copy()

    w = _odd_window(n, win)

    if _HAS_SAVGOL and w >= 3:
        p = int(poly)
        if p >= w:
            p = max(1, w - 2)
        return savgol_filter(y, window_length=w, polyorder=p, mode="interp")

    s = pd.Series(y)
    return s.rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)


def load_and_filter(csv_dir: Path, win: int = 11, poly: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_c = pd.read_csv(csv_dir / CONCAT_FILE)
    df_r = pd.read_csv(csv_dir / RM_FILE)

    x_c = pick_first_existing(df_c, ["seconds", "elapsed_s", "time_s"], "concat x-axis")
    y_c = pick_first_existing(df_c, ["mos_pred_concat", "mos_pred", "mos"], "concat predicted MOS")
    t_c = pick_first_existing(
        df_c,
        ["mos_target_concat_durw", "mos_target_concat_mean", "mos_target_concat_avg", "mos_target_concat", "mos_target"],
        "concat target MOS",
    )

    x_r = pick_first_existing(df_r, ["elapsed_s", "seconds", "time_s"], "running-mean x-axis")
    y_r = pick_first_existing(
        df_r,
        ["mos_pred_rm_durw", "mos_pred_rm_mean", "mos_pred_rm", "mos_running_mean", "mos_pred", "mos"],
        "running-mean predicted MOS",
    )
    t_r = pick_first_existing(
        df_r,
        ["mos_target_rm_durw", "mos_target_rm_mean", "mos_target_rm", "mos_target"],
        "running-mean target MOS",
    )

    df_c = df_c.sort_values(x_c).reset_index(drop=True)
    df_r = df_r.sort_values(x_r).reset_index(drop=True)

    ycf = smooth_series(df_c[y_c].to_numpy(), win=win, poly=poly)
    yrf = smooth_series(df_r[y_r].to_numpy(), win=win, poly=poly)

    out_c = df_c.copy()
    out_r = df_r.copy()

    out_c[f"{y_c}_raw"] = out_c[y_c]
    out_c[y_c] = ycf

    out_r[f"{y_r}_raw"] = out_r[y_r]
    out_r[y_r] = yrf

    meta = {
        "concat": {"x": x_c, "pred": y_c, "tgt": t_c},
        "rm":     {"x": x_r, "pred": y_r, "tgt": t_r},
    }
    return out_c, out_r, meta


def plot_filter(csv_dir: Path, dataset_root: Path, df_c: pd.DataFrame, df_r: pd.DataFrame, meta: dict) -> Path:
    x_c = meta["concat"]["x"]
    y_c = meta["concat"]["pred"]
    t_c = meta["concat"]["tgt"]
    y_c_raw = f"{y_c}_raw"

    x_r = meta["rm"]["x"]
    y_r = meta["rm"]["pred"]
    t_r = meta["rm"]["tgt"]
    y_r_raw = f"{y_r}_raw"

    model_info = model_info_from_dataset_root(dataset_root)
    base = dataset_base_from_root(dataset_root)
    train = train_id_from_root(dataset_root)

    try:
        rel = str(csv_dir.relative_to(dataset_root))
    except Exception:
        rel = str(csv_dir)

    fig, ax = plt.subplots(1, 1, figsize=(13, 6.8))

    if y_c_raw in df_c.columns:
        ax.plot(df_c[x_c], df_c[y_c_raw], color=COL_RAW, linewidth=1.2, alpha=0.55, label="concat (raw pred)")
    ax.plot(df_c[x_c], df_c[y_c], color=COL_CONCAT, linewidth=2.6, alpha=0.95, label="concat (filtered pred)")
    ax.plot(df_c[x_c], df_c[t_c], color=COL_TGT, linewidth=1.8, alpha=0.80, label="target MOS (concat grid)")

    if y_r_raw in df_r.columns:
        ax.plot(df_r[x_r], df_r[y_r_raw], color=COL_RAW, linewidth=1.2, alpha=0.35, linestyle="--", label="rm (raw pred)")
    ax.plot(df_r[x_r], df_r[y_r], color=COL_RM, linewidth=2.2, alpha=0.95, linestyle="--", label="rm (filtered pred)")
    ax.plot(df_r[x_r], df_r[t_r], color=COL_TGT, linewidth=1.2, alpha=0.45, linestyle=":", label="target MOS (rm grid)")

    ax.set_title(f"{base} train{train} ({model_info}) — raw vs filtered\n{rel}", fontsize=16, **fkw())
    ax.set_xlabel("audio length [s]", fontsize=14, **fkw())
    ax.set_ylabel("MOS", fontsize=14, **fkw())
    pretty_axes(ax)
    ax.legend(loc="best", fontsize=10, frameon=True)

    fig.tight_layout()
    out_path = csv_dir / OUT_PLOT
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def _drop_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _pearson_clean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    if _HAS_STATS:
        return float(pearsonr(a, b)[0])
    return float(np.corrcoef(a, b)[0, 1])


def _spearman_clean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    if _HAS_STATS:
        return float(spearmanr(a, b).correlation)
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def _kendall_clean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    if _HAS_STATS:
        return float(kendalltau(a, b).correlation)
    try:
        return float(pd.Series(a).corr(pd.Series(b), method="kendall"))
    except Exception:
        return np.nan


def corr_triplet(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    a, b = _drop_pair(a, b)
    n = int(len(a))
    return _pearson_clean(a, b), _spearman_clean(a, b), _kendall_clean(a, b), n


def _align_merge_round(df_a: pd.DataFrame, xa: str, ya: str, df_b: pd.DataFrame, xb: str, yb: str, dec: int):
    a = df_a[[xa, ya]].dropna().copy()
    b = df_b[[xb, yb]].dropna().copy()
    a["t"] = a[xa].round(dec)
    b["t"] = b[xb].round(dec)
    m = pd.merge(a[["t", ya]], b[["t", yb]], on="t", how="inner")
    return m[ya].to_numpy(dtype=np.float64), m[yb].to_numpy(dtype=np.float64), f"round{dec}"


def _align_merge_asof(df_a: pd.DataFrame, xa: str, ya: str, df_b: pd.DataFrame, xb: str, yb: str, tol: float):
    a = df_a[[xa, ya]].dropna().sort_values(xa).copy()
    b = df_b[[xb, yb]].dropna().sort_values(xb).copy()
    a = a.rename(columns={xa: "t"})
    b = b.rename(columns={xb: "t"})
    m = pd.merge_asof(a, b, on="t", direction="nearest", tolerance=tol, suffixes=("_a", "_b"))
    m = m.dropna()
    return m[ya].to_numpy(dtype=np.float64), m[yb].to_numpy(dtype=np.float64), f"asof_tol{tol:g}"


def align_two_by_time(df_a: pd.DataFrame, xa: str, ya: str, df_b: pd.DataFrame, xb: str, yb: str):
    for dec in (3, 2, 1, 0):
        a, b, how = _align_merge_round(df_a, xa, ya, df_b, xb, yb, dec)
        if len(a) >= 3:
            return a, b, f"{how} (n={len(a)})"
    for tol in (0.05, 0.1, 0.2, 0.5):
        a, b, how = _align_merge_asof(df_a, xa, ya, df_b, xb, yb, tol)
        if len(a) >= 3:
            return a, b, f"{how} (n={len(a)})"
    a, b, how = _align_merge_round(df_a, xa, ya, df_b, xb, yb, 0)
    return a, b, f"{how} (n={len(a)})"


def align_concat_rm(df_c: pd.DataFrame, df_r: pd.DataFrame, meta: dict, col_c: str, col_r: str):
    x_c = meta["concat"]["x"]
    x_r = meta["rm"]["x"]
    return align_two_by_time(df_c, x_c, col_c, df_r, x_r, col_r)


def parse_path_meta(dataset_root: Path, d: Path) -> dict:
    rel = d.relative_to(dataset_root)
    parts = list(rel.parts)

    system_id = parts[0] if parts else d.name
    variant = "root"
    if len(parts) >= 2 and parts[1] in VARIANT_DIRS:
        variant = parts[1]
    elif d.name in VARIANT_DIRS:
        variant = d.name

    mode = "forward"
    base_system_id = system_id
    if system_id.startswith("rev_"):
        mode = "reverse"
        base_system_id = system_id[len("rev_"):]
    elif system_id.startswith("rand_"):
        mode = "random"
        base_system_id = system_id[len("rand_"):]

    return {
        "system_id": system_id,
        "base_system_id": base_system_id,
        "mode": mode,
        "variant": variant,
        "rel_path": str(rel),
    }


def _fmt(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def _fmt_delta(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "nan"
    return f"{float(x):+.{nd}f}"


def _line(cols, widths):
    out = []
    for c, w in zip(cols, widths):
        out.append(str(c)[:w].ljust(w))
    return "  ".join(out).rstrip()


def write_report_target(path: Path, dataset_root: Path, df: pd.DataFrame, fail_lines: list[str], win: int, poly: int):
    with open(path, "w", encoding="utf-8") as f:
        base = dataset_base_from_root(dataset_root)
        tr = train_id_from_root(dataset_root)
        f.write(f"{dataset_root.name} | {base} train{tr} ({model_info_from_dataset_root(dataset_root)})\n")
        f.write(f"correlations to target: raw vs filtered | win={win} poly={poly}\n\n")

        if df.empty:
            f.write("(no rows)\n")
        else:
            df2 = df.copy()
            df2["variant"] = pd.Categorical(df2["variant"], categories=VARIANT_DIRS, ordered=True)
            df2 = df2.sort_values(["base_system_id", "mode", "variant", "rel_path"])

            widths = [14, 8, 14, 6, 9, 9, 9, 9, 9, 9, 6]
            header = _line(
                ["system", "mode", "variant", "grid", "PCCraw", "PCCf", "dPCC", "SRCCraw", "SRCCf", "dSRCC", "n"],
                widths
            )

            for (sid, mode), g in df2.groupby(["base_system_id", "mode"], dropna=False):
                f.write(f"{sid} | {mode}\n")
                f.write(header + "\n")

                for _, r in g.iterrows():
                    f.write(_line([
                        sid, mode, r["variant"], "c",
                        _fmt(r["concat_PCC_raw"]), _fmt(r["concat_PCC_filt"]), _fmt_delta(r["concat_dPCC"]),
                        _fmt(r["concat_SRCC_raw"]), _fmt(r["concat_SRCC_filt"]), _fmt_delta(r["concat_dSRCC"]),
                        int(r["concat_n_filt"]) if np.isfinite(r["concat_n_filt"]) else int(r["concat_n_raw"]),
                    ], widths) + "\n")

                    f.write(_line([
                        sid, mode, r["variant"], "rm",
                        _fmt(r["rm_PCC_raw"]), _fmt(r["rm_PCC_filt"]), _fmt_delta(r["rm_dPCC"]),
                        _fmt(r["rm_SRCC_raw"]), _fmt(r["rm_SRCC_filt"]), _fmt_delta(r["rm_dSRCC"]),
                        int(r["rm_n_filt"]) if np.isfinite(r["rm_n_filt"]) else int(r["rm_n_raw"]),
                    ], widths) + "\n")

                f.write("\n")

        if fail_lines:
            f.write("\nFAILURES\n")
            for line in fail_lines:
                f.write(line.rstrip() + "\n")


def write_report_cr(path: Path, dataset_root: Path, df: pd.DataFrame, fail_lines: list[str], win: int, poly: int):
    with open(path, "w", encoding="utf-8") as f:
        base = dataset_base_from_root(dataset_root)
        tr = train_id_from_root(dataset_root)
        f.write(f"{dataset_root.name} | {base} train{tr} ({model_info_from_dataset_root(dataset_root)})\n")
        f.write(f"concat vs rm correlations: raw vs filtered | win={win} poly={poly}\n\n")

        if df.empty:
            f.write("(no rows)\n")
        else:
            df2 = df.copy()
            df2["variant"] = pd.Categorical(df2["variant"], categories=VARIANT_DIRS, ordered=True)
            df2 = df2.sort_values(["base_system_id", "mode", "variant", "rel_path"])

            widths = [14, 8, 14, 9, 9, 9, 9, 6, 18]
            header = _line(["system", "mode", "variant", "PCCraw", "PCCf", "dPCC", "SRCCraw", "n", "align"], widths)

            for (sid, mode), g in df2.groupby(["base_system_id", "mode"], dropna=False):
                f.write(f"{sid} | {mode}\n")
                f.write(header + "\n")

                for _, r in g.iterrows():
                    n_use = int(r["cr_n_filt"]) if np.isfinite(r["cr_n_filt"]) else int(r["cr_n_raw"])
                    align = str(r.get("align_filt", r.get("align_raw", "")))
                    f.write(_line([
                        sid, mode, r["variant"],
                        _fmt(r["cr_PCC_raw"]), _fmt(r["cr_PCC_filt"]), _fmt_delta(r["cr_dPCC"]),
                        _fmt(r["cr_SRCC_raw"]),
                        n_use,
                        align,
                    ], widths) + "\n")

                f.write("\n")

        if fail_lines:
            f.write("\nFAILURES\n")
            for line in fail_lines:
                f.write(line.rstrip() + "\n")


def process_dataset(dataset_root: Path, win: int = 11, poly: int = 2):
    if not dataset_root.exists():
        print(f"[SKIP] missing dataset dir: {dataset_root}")
        return pd.DataFrame(), pd.DataFrame(), []

    dirs = sorted(set(iter_result_dirs(dataset_root)), key=lambda p: str(p))
    if not dirs:
        print(f"[SKIP] no result dirs found under: {dataset_root}")
        return pd.DataFrame(), pd.DataFrame(), []

    base = dataset_base_from_root(dataset_root)
    train = train_id_from_root(dataset_root)

    target_rows = []
    cr_rows = []
    fail_lines = []

    ok, fail = 0, 0
    for d in dirs:
        rel = d.relative_to(dataset_root)
        try:
            out_c, out_r, meta = load_and_filter(d, win=win, poly=poly)

            out_c.to_csv(d / OUT_CONCAT, index=False)
            out_r.to_csv(d / OUT_RM, index=False)
            plot_filter(d, dataset_root, out_c, out_r, meta)

            m = parse_path_meta(dataset_root, d)
            m["dataset_root"] = dataset_root.name
            m["dataset_base"] = base
            m["train"] = train
            m["train_label"] = TRAIN_LABEL.get(train, str(train))

            yc = meta["concat"]["pred"]
            tc = meta["concat"]["tgt"]
            yr = meta["rm"]["pred"]
            tr = meta["rm"]["tgt"]

            yc_raw = f"{yc}_raw"
            yr_raw = f"{yr}_raw"

            c_raw_p, c_raw_s, c_raw_k, c_raw_n = corr_triplet(out_c[yc_raw].to_numpy(), out_c[tc].to_numpy())
            c_fil_p, c_fil_s, c_fil_k, c_fil_n = corr_triplet(out_c[yc].to_numpy(), out_c[tc].to_numpy())

            r_raw_p, r_raw_s, r_raw_k, r_raw_n = corr_triplet(out_r[yr_raw].to_numpy(), out_r[tr].to_numpy())
            r_fil_p, r_fil_s, r_fil_k, r_fil_n = corr_triplet(out_r[yr].to_numpy(), out_r[tr].to_numpy())

            target_rows.append({
                **m,
                "concat_PCC_raw": c_raw_p, "concat_SRCC_raw": c_raw_s, "concat_KTAU_raw": c_raw_k, "concat_n_raw": c_raw_n,
                "concat_PCC_filt": c_fil_p, "concat_SRCC_filt": c_fil_s, "concat_KTAU_filt": c_fil_k, "concat_n_filt": c_fil_n,
                "concat_dPCC": (c_fil_p - c_raw_p) if np.isfinite(c_fil_p) and np.isfinite(c_raw_p) else np.nan,
                "concat_dSRCC": (c_fil_s - c_raw_s) if np.isfinite(c_fil_s) and np.isfinite(c_raw_s) else np.nan,
                "concat_dKTAU": (c_fil_k - c_raw_k) if np.isfinite(c_fil_k) and np.isfinite(c_raw_k) else np.nan,

                "rm_PCC_raw": r_raw_p, "rm_SRCC_raw": r_raw_s, "rm_KTAU_raw": r_raw_k, "rm_n_raw": r_raw_n,
                "rm_PCC_filt": r_fil_p, "rm_SRCC_filt": r_fil_s, "rm_KTAU_filt": r_fil_k, "rm_n_filt": r_fil_n,
                "rm_dPCC": (r_fil_p - r_raw_p) if np.isfinite(r_fil_p) and np.isfinite(r_raw_p) else np.nan,
                "rm_dSRCC": (r_fil_s - r_raw_s) if np.isfinite(r_fil_s) and np.isfinite(r_raw_s) else np.nan,
                "rm_dKTAU": (r_fil_k - r_raw_k) if np.isfinite(r_fil_k) and np.isfinite(r_raw_k) else np.nan,
            })

            a_raw, b_raw, how_raw = align_concat_rm(out_c, out_r, meta, yc_raw, yr_raw)
            a_fil, b_fil, how_fil = align_concat_rm(out_c, out_r, meta, yc, yr)

            cr_raw_p, cr_raw_s, cr_raw_k, cr_raw_n = corr_triplet(a_raw, b_raw)
            cr_fil_p, cr_fil_s, cr_fil_k, cr_fil_n = corr_triplet(a_fil, b_fil)

            cr_rows.append({
                **m,
                "cr_PCC_raw": cr_raw_p, "cr_SRCC_raw": cr_raw_s, "cr_KTAU_raw": cr_raw_k, "cr_n_raw": cr_raw_n, "align_raw": how_raw,
                "cr_PCC_filt": cr_fil_p, "cr_SRCC_filt": cr_fil_s, "cr_KTAU_filt": cr_fil_k, "cr_n_filt": cr_fil_n, "align_filt": how_fil,
                "cr_dPCC": (cr_fil_p - cr_raw_p) if np.isfinite(cr_fil_p) and np.isfinite(cr_raw_p) else np.nan,
                "cr_dSRCC": (cr_fil_s - cr_raw_s) if np.isfinite(cr_fil_s) and np.isfinite(cr_raw_s) else np.nan,
                "cr_dKTAU": (cr_fil_k - cr_raw_k) if np.isfinite(cr_fil_k) and np.isfinite(cr_raw_k) else np.nan,
            })

            print(f"[OK]  {dataset_root.name}/{rel}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {dataset_root.name}/{rel}: {e}")
            fail_lines.append(f"{dataset_root.name}/{rel}: {e}")
            fail += 1

    df_t = pd.DataFrame(target_rows)
    df_cr = pd.DataFrame(cr_rows)

    write_report_target(dataset_root / OUT_TXT_TARGET, dataset_root, df_t, fail_lines, win=win, poly=poly)
    write_report_cr(dataset_root / OUT_TXT_CR, dataset_root, df_cr, fail_lines, win=win, poly=poly)

    print(f"DONE {dataset_root.name}: ok={ok} fail={fail}\n")
    return df_t, df_cr, fail_lines


def target_long(df_t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_t.iterrows():
        base = {
            "dataset_base": r["dataset_base"],
            "train": r["train"],
            "train_label": r["train_label"],
            "base_system_id": r["base_system_id"],
            "mode": r["mode"],
            "variant": r["variant"],
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


def cr_long(df_cr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_cr.iterrows():
        rows.append({
            "dataset_base": r["dataset_base"],
            "train": r["train"],
            "train_label": r["train_label"],
            "base_system_id": r["base_system_id"],
            "mode": r["mode"],
            "variant": r["variant"],
            "PCC_raw": r["cr_PCC_raw"], "SRCC_raw": r["cr_SRCC_raw"], "KTAU_raw": r["cr_KTAU_raw"], "n_raw": r["cr_n_raw"],
            "PCC_filt": r["cr_PCC_filt"], "SRCC_filt": r["cr_SRCC_filt"], "KTAU_filt": r["cr_KTAU_filt"], "n_filt": r["cr_n_filt"],
            "dPCC": r["cr_dPCC"], "dSRCC": r["cr_dSRCC"], "dKTAU": r["cr_dKTAU"],
            "align_raw": r.get("align_raw", ""),
            "align_filt": r.get("align_filt", ""),
        })
    return pd.DataFrame(rows)


def _pairwise_deltas(df: pd.DataFrame, metric: str, group_cols: list[str]) -> pd.DataFrame:
    # df must have: train, group_cols, metric
    out_rows = []
    if df.empty:
        return pd.DataFrame(out_rows)

    # pivot per case
    case_cols = [c for c in group_cols if c not in ["train", "train_label"]]
    pv = df.pivot_table(index=case_cols, columns="train", values=metric, aggfunc="first")
    trains = [c for c in pv.columns if pd.api.types.is_number(c)]

    trains = sorted(trains)
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
    return pd.DataFrame(out_rows)


def _win_counts(df: pd.DataFrame, metric: str, case_cols: list[str]) -> pd.DataFrame:
    # for each case: which train has max(metric)
    if df.empty:
        return pd.DataFrame()

    def _best_train(g):
        g2 = g.dropna(subset=[metric]).copy()
        if g2.empty:
            return pd.Series({"best_train": np.nan})
        idx = g2[metric].astype(float).idxmax()
        return pd.Series({"best_train": int(g2.loc[idx, "train"])})

    best = df.groupby(case_cols, dropna=False).apply(_best_train).reset_index()
    best = best.dropna(subset=["best_train"])
    best["best_train"] = best["best_train"].astype(int)
    cnt = best.groupby(case_cols[0:1] + ["best_train"]).size().reset_index(name="count")  # keep dataset_base in front
    cnt["best_label"] = cnt["best_train"].map(lambda t: TRAIN_LABEL.get(int(t), str(t)))
    return cnt.sort_values([case_cols[0], "count"], ascending=[True, False])


def _mode_effects(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # compare reverse/random vs forward per (dataset_base, train, system, variant, grid)
    if df.empty:
        return pd.DataFrame()

    key = ["dataset_base", "train", "base_system_id", "variant", "grid"]
    pv = df.pivot_table(index=key, columns="mode", values=metric, aggfunc="first")
    rows = []
    for idx, r in pv.iterrows():
        fwd = r.get("forward", np.nan)
        rev = r.get("reverse", np.nan)
        rnd = r.get("random", np.nan)
        rows.append({
            "dataset_base": idx[0],
            "train": int(idx[1]),
            "train_label": TRAIN_LABEL.get(int(idx[1]), str(idx[1])),
            "base_system_id": idx[2],
            "variant": idx[3],
            "grid": idx[4],
            f"{metric}_forward": fwd,
            f"{metric}_reverse": rev,
            f"{metric}_random": rnd,
            f"d_reverse_minus_forward": (rev - fwd) if np.isfinite(rev) and np.isfinite(fwd) else np.nan,
            f"d_random_minus_forward": (rnd - fwd) if np.isfinite(rnd) and np.isfinite(fwd) else np.nan,
        })
    return pd.DataFrame(rows)


def _variant_effects(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # compare variants vs original per (dataset_base, train, system, mode, grid)
    if df.empty:
        return pd.DataFrame()

    key = ["dataset_base", "train", "base_system_id", "mode", "grid"]
    pv = df.pivot_table(index=key, columns="variant", values=metric, aggfunc="first")
    rows = []
    for idx, r in pv.iterrows():
        base = r.get("original", np.nan)
        for v in VARIANT_DIRS:
            if v == "original":
                continue
            vv = r.get(v, np.nan)
            rows.append({
                "dataset_base": idx[0],
                "train": int(idx[1]),
                "train_label": TRAIN_LABEL.get(int(idx[1]), str(idx[1])),
                "base_system_id": idx[2],
                "mode": idx[3],
                "grid": idx[4],
                "variant": v,
                f"{metric}_original": base,
                f"{metric}_{v}": vv,
                "delta": (vv - base) if np.isfinite(vv) and np.isfinite(base) else np.nan,
            })
    return pd.DataFrame(rows)


def _concat_vs_rm_diff(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # per case: metric(concat) - metric(rm)
    if df.empty:
        return pd.DataFrame()

    key = ["dataset_base", "train", "base_system_id", "mode", "variant"]
    pv = df.pivot_table(index=key, columns="grid", values=metric, aggfunc="first")
    rows = []
    for idx, r in pv.iterrows():
        c = r.get("concat", np.nan)
        rm = r.get("rm", np.nan)
        rows.append({
            "dataset_base": idx[0],
            "train": int(idx[1]),
            "train_label": TRAIN_LABEL.get(int(idx[1]), str(idx[1])),
            "base_system_id": idx[2],
            "mode": idx[3],
            "variant": idx[4],
            f"{metric}_concat": c,
            f"{metric}_rm": rm,
            "delta_concat_minus_rm": (c - rm) if np.isfinite(c) and np.isfinite(rm) else np.nan,
        })
    return pd.DataFrame(rows)


def _load_pred_series(csv_dir: Path, which: str, win: int, poly: int) -> pd.DataFrame:
    if which == "concat":
        df = pd.read_csv(csv_dir / CONCAT_FILE)
        x = pick_first_existing(df, ["seconds", "elapsed_s", "time_s"], "concat x-axis")
        y = pick_first_existing(df, ["mos_pred_concat", "mos_pred", "mos"], "concat predicted MOS")
    elif which == "rm":
        df = pd.read_csv(csv_dir / RM_FILE)
        x = pick_first_existing(df, ["elapsed_s", "seconds", "time_s"], "rm x-axis")
        y = pick_first_existing(df, ["mos_pred_rm_durw", "mos_pred_rm_mean", "mos_pred_rm", "mos_running_mean", "mos_pred", "mos"], "rm predicted MOS")
    else:
        raise ValueError(which)

    df = df.sort_values(x).reset_index(drop=True)
    raw = df[y].to_numpy(dtype=np.float64)
    fil = smooth_series(raw, win=win, poly=poly)
    return pd.DataFrame({"t": df[x].to_numpy(dtype=np.float64), "raw": raw, "filt": fil})


def _build_case_map(dataset_root: Path):
    mp = {}
    for d in iter_result_dirs(dataset_root):
        m = parse_path_meta(dataset_root, d)
        key = (m["base_system_id"], m["mode"], m["variant"])
        if key not in mp:
            mp[key] = d
        else:
            if len(str(d)) < len(str(mp[key])):
                mp[key] = d
    return mp


def curve_similarity_concat_pairs(results_root: Path, base: str, trains: list[int], win: int, poly: int) -> pd.DataFrame:
    # per pair of trains, per common case: correlation of concat pred curves (raw+filt)
    roots = {tr: results_root / f"{base}_{tr}" for tr in trains}
    maps = {tr: _build_case_map(roots[tr]) for tr in trains if roots[tr].exists()}

    out_rows = []
    pairs = []
    tlist = sorted([tr for tr in trains if tr in maps])
    for i in range(len(tlist)):
        for j in range(i + 1, len(tlist)):
            pairs.append((tlist[i], tlist[j]))

    for a, b in pairs:
        common = sorted(set(maps[a].keys()) & set(maps[b].keys()))
        for (sid, mode, variant) in common:
            da = maps[a][(sid, mode, variant)]
            db = maps[b][(sid, mode, variant)]
            try:
                sa = _load_pred_series(da, "concat", win=win, poly=poly)
                sb = _load_pred_series(db, "concat", win=win, poly=poly)

                ar, br, how_raw = align_two_by_time(sa, "t", "raw", sb, "t", "raw")
                af, bf, how_filt = align_two_by_time(sa, "t", "filt", sb, "t", "filt")

                p_raw, s_raw, k_raw, n_raw = corr_triplet(ar, br)
                p_fil, s_fil, k_fil, n_fil = corr_triplet(af, bf)

                out_rows.append({
                    "dataset_base": base,
                    "train_a": a, "label_a": TRAIN_LABEL.get(a, str(a)),
                    "train_b": b, "label_b": TRAIN_LABEL.get(b, str(b)),
                    "base_system_id": sid,
                    "mode": mode,
                    "variant": variant,
                    "PCC_raw": p_raw, "SRCC_raw": s_raw, "n_raw": n_raw, "align_raw": how_raw,
                    "PCC_filt": p_fil, "SRCC_filt": s_fil, "n_filt": n_fil, "align_filt": how_filt,
                })
            except Exception:
                continue

    return pd.DataFrame(out_rows)


def write_training_comparison_txt(path: Path, df_target_long: pd.DataFrame, df_cr_long: pd.DataFrame, ds_base: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n====================\n{ds_base}\n====================\n")

        sub = df_target_long[df_target_long["dataset_base"] == ds_base].copy()
        if sub.empty:
            f.write("(no target_long rows)\n")
            return

        # overall means
        f.write("\n1) mean/median SRCC_filt to target (by train, grid)\n")
        g = sub.groupby(["train", "grid"], dropna=False)["SRCC_filt"].agg(["mean", "median", "count"]).reset_index()
        g["label"] = g["train"].map(lambda t: TRAIN_LABEL.get(int(t), str(t)))
        g = g.sort_values(["grid", "mean"], ascending=[True, False])
        for grid in ["concat", "rm"]:
            gg = g[g["grid"] == grid]
            f.write(f"\n  grid={grid}\n")
            for _, r in gg.iterrows():
                f.write(f"    train{int(r['train'])} {r['label']}: mean={_fmt(r['mean'])} median={_fmt(r['median'])} n={int(r['count'])}\n")

        # win counts (best SRCC_filt)
        f.write("\n2) win counts (best SRCC_filt) per train (by grid)\n")
        case_cols = ["dataset_base", "base_system_id", "mode", "variant", "grid"]
        wc = _win_counts(sub, "SRCC_filt", case_cols)
        if wc.empty:
            f.write("(no win-count data)\n")
        else:
            for grid in ["concat", "rm"]:
                f.write(f"\n  grid={grid}\n")
                w2 = sub[sub["grid"] == grid].copy()
                wc2 = _win_counts(w2, "SRCC_filt", case_cols)
                if wc2.empty:
                    f.write("    (none)\n")
                    continue
                for _, r in wc2.iterrows():
                    f.write(f"    {r['best_label']} (train{int(r['best_train'])}): {int(r['count'])}\n")

        # pairwise deltas summary
        f.write("\n3) pairwise mean deltas (SRCC_filt to target)  (train_b - train_a)\n")
        pw = _pairwise_deltas(sub, "SRCC_filt", ["dataset_base", "base_system_id", "mode", "variant", "grid", "train"])
        if pw.empty:
            f.write("(no pairwise deltas)\n")
        else:
            pw = pw.sort_values(["mean_delta"], ascending=False)
            for _, r in pw.iterrows():
                f.write(
                    f"  {r['label_b']} (train{int(r['train_b'])}) - {r['label_a']} (train{int(r['train_a'])}): "
                    f"mean={_fmt(r['mean_delta'])} median={_fmt(r['median_delta'])} n={int(r['n'])} "
                    f"wins_b={int(r['wins_b'])} wins_a={int(r['wins_a'])} ties={int(r['ties'])}\n"
                )

        # concat vs rm correlation (internal consistency)
        f.write("\n4) concat-vs-rm SRCC_filt (by train)\n")
        subcr = df_cr_long[df_cr_long["dataset_base"] == ds_base].copy()
        if subcr.empty:
            f.write("(no concat-vs-rm rows)\n")
        else:
            g2 = subcr.groupby(["train"], dropna=False)["SRCC_filt"].agg(["mean", "median", "count"]).reset_index()
            g2["label"] = g2["train"].map(lambda t: TRAIN_LABEL.get(int(t), str(t)))
            g2 = g2.sort_values(["mean"], ascending=False)
            for _, r in g2.iterrows():
                f.write(f"  train{int(r['train'])} {r['label']}: mean={_fmt(r['mean'])} median={_fmt(r['median'])} n={int(r['count'])}\n")


def main():
    results_root = find_results_root()
    WIN = 11
    POLY = 2

    out = {}
    dataset_roots = []
    for d in sorted(results_root.iterdir()):
        if d.is_dir() and (d.name.startswith("BVCC_") or d.name.startswith("SOMOS_")):
            dataset_roots.append(d)

    if not dataset_roots:
        raise FileNotFoundError(f"no dataset dirs found under {results_root} (expected BVCC_* / SOMOS_*)")

    all_t = []
    all_cr = []

    for ds_root in dataset_roots:
        df_t, df_cr, fails = process_dataset(ds_root, win=WIN, poly=POLY)
        out[ds_root.name] = {"t": df_t, "cr": df_cr, "fails": fails}
        if not df_t.empty:
            all_t.append(df_t)
        if not df_cr.empty:
            all_cr.append(df_cr)

    all_t_df = pd.concat(all_t, ignore_index=True) if all_t else pd.DataFrame()
    all_cr_df = pd.concat(all_cr, ignore_index=True) if all_cr else pd.DataFrame()

    # long tables
    tgt_long = target_long(all_t_df) if not all_t_df.empty else pd.DataFrame()
    crl = cr_long(all_cr_df) if not all_cr_df.empty else pd.DataFrame()

    # write global CSVs
    if not tgt_long.empty:
        (results_root / "summary_target_long.csv").write_text(tgt_long.to_csv(index=False), encoding="utf-8")
    else:
        (results_root / "summary_target_long.csv").write_text("", encoding="utf-8")

    if not crl.empty:
        (results_root / "summary_concat_vs_rm_long.csv").write_text(crl.to_csv(index=False), encoding="utf-8")
    else:
        (results_root / "summary_concat_vs_rm_long.csv").write_text("", encoding="utf-8")

    # pairwise deltas to target (SRCC_filt + PCC_filt)
    pair_rows = []
    for base in sorted(tgt_long["dataset_base"].unique()) if not tgt_long.empty else []:
        sub = tgt_long[tgt_long["dataset_base"] == base].copy()
        for metric in ["SRCC_filt", "PCC_filt", "SRCC_raw", "PCC_raw"]:
            pw = _pairwise_deltas(sub, metric, ["dataset_base", "base_system_id", "mode", "variant", "grid", "train"])
            if not pw.empty:
                pw["dataset_base"] = base
                pair_rows.append(pw)
    pair_df = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    (results_root / "summary_pairwise_deltas_to_target.csv").write_text(pair_df.to_csv(index=False), encoding="utf-8")

    # win counts (best SRCC_filt)
    if not tgt_long.empty:
        wc = _win_counts(tgt_long, "SRCC_filt", ["dataset_base", "base_system_id", "mode", "variant", "grid"])
    else:
        wc = pd.DataFrame()
    (results_root / "summary_win_counts_to_target.csv").write_text(wc.to_csv(index=False), encoding="utf-8")

    # mode effects (reverse/random vs forward)
    me = _mode_effects(tgt_long, "SRCC_filt") if not tgt_long.empty else pd.DataFrame()
    (results_root / "summary_mode_effects.csv").write_text(me.to_csv(index=False), encoding="utf-8")

    # variant effects (each variant vs original)
    ve = _variant_effects(tgt_long, "SRCC_filt") if not tgt_long.empty else pd.DataFrame()
    (results_root / "summary_variant_effects.csv").write_text(ve.to_csv(index=False), encoding="utf-8")

    # concat vs rm diff for SRCC_filt
    crdiff = _concat_vs_rm_diff(tgt_long, "SRCC_filt") if not tgt_long.empty else pd.DataFrame()
    (results_root / "summary_concat_vs_rm_diff.csv").write_text(crdiff.to_csv(index=False), encoding="utf-8")

    # curve similarity across trainings (concat pred curves)
    curve_rows = []
    for base in ["BVCC", "SOMOS"]:
        # detect all trains present for this base
        trains = []
        for d in dataset_roots:
            bt = _dataset_base_and_train(d.name)
            if not bt:
                continue
            b, tr = bt
            if b == base:
                trains.append(tr)
        trains = sorted(set(trains))
        if len(trains) < 2:
            continue
        df_curve = curve_similarity_concat_pairs(results_root, base, trains, win=WIN, poly=POLY)
        if not df_curve.empty:
            curve_rows.append(df_curve)

    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    (results_root / "summary_curve_similarity_concat_pairs.csv").write_text(curve_df.to_csv(index=False), encoding="utf-8")

    # text report
    cross_path = results_root / OUT_TXT_CROSS
    cross_path.write_text("", encoding="utf-8")
    for base in ["BVCC", "SOMOS"]:
        write_training_comparison_txt(cross_path, tgt_long, crl, base)

    print(f"[DONE] wrote: {cross_path}")
    print(f"[DONE] wrote CSV summaries into: {results_root}")


if __name__ == "__main__":
    main()
