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


DPI = 300

CONCAT_FILE = "global_mos_concat.csv"
RM_FILE     = "global_mos_running_mean.csv"

OUT_CONCAT  = "filter_global_mos_concat.csv"
OUT_RM      = "filter_global_mos_running_mean.csv"
OUT_PLOT    = "plot_filter.png"

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

MODEL_INFO_MAP = {
    "BVCC_24": "base + BiLSTM (24)",
    "BVCC_27": "large + Conv (27)",
    "SOMOS_24": "base + BiLSTM (24)",
    "SOMOS_27": "large + Conv (27)",
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


def model_info_from_dataset_root(dataset_root: Path) -> str:
    if dataset_root.name in MODEL_INFO_MAP:
        return MODEL_INFO_MAP[dataset_root.name]
    # fallback: show folder name instead of "unknown"
    if re.fullmatch(r"(BVCC|SOMOS)_\d+", dataset_root.name):
        return dataset_root.name
    return "unknown model"


def dataset_name_from_root(dataset_root: Path) -> str:
    n = dataset_root.name
    if n.startswith("BVCC"):
        return "BVCC"
    if n.startswith("SOMOS"):
        return "SOMOS"
    return n


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
    c_concat = csv_dir / CONCAT_FILE
    c_rm     = csv_dir / RM_FILE

    df_c = pd.read_csv(c_concat)
    df_r = pd.read_csv(c_rm)

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
    dataset_name = dataset_name_from_root(dataset_root)

    try:
        rel = str(csv_dir.relative_to(dataset_root))
    except Exception:
        rel = str(csv_dir)

    fig, ax = plt.subplots(1, 1, figsize=(13, 6.0))

    if y_c_raw in df_c.columns:
        ax.plot(df_c[x_c], df_c[y_c_raw], color=COL_RAW, linewidth=1.2, alpha=0.6, label="concat (raw pred)")
    ax.plot(df_c[x_c], df_c[y_c], color=COL_CONCAT, linewidth=2.6, label="concat (filtered pred)")

    if y_r_raw in df_r.columns:
        ax.plot(df_r[x_r], df_r[y_r_raw], color=COL_RAW, linewidth=1.2, alpha=0.35, linestyle="--", label="running mean (raw pred)")
    ax.plot(df_r[x_r], df_r[y_r], color=COL_RM, linewidth=2.3, linestyle="--", label="running mean (filtered pred)")

    if t_c in df_c.columns:
        ax.plot(df_c[x_c], df_c[t_c], color=COL_TGT, linewidth=1.8, alpha=0.85, label="target MOS (concat)")
    if t_r in df_r.columns:
        ax.plot(df_r[x_r], df_r[t_r], color=COL_TGT, linewidth=1.6, linestyle=":", alpha=0.65, label="target MOS (running mean)")

    ax.set_title(f"{dataset_name} ({model_info}) — filtered MOS\n{rel}", fontsize=16, **fkw())
    ax.set_xlabel("audio length [s]", fontsize=14, **fkw())
    ax.set_ylabel("MOS", fontsize=14, **fkw())
    pretty_axes(ax)
    ax.legend(loc="best", fontsize=10, frameon=True)

    fig.tight_layout()
    out_path = csv_dir / OUT_PLOT
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def process_dataset(dataset_root: Path, win: int = 11, poly: int = 2) -> None:
    if not dataset_root.exists():
        print(f"[SKIP] missing dataset dir: {dataset_root}")
        return

    dirs = sorted(set(iter_result_dirs(dataset_root)), key=lambda p: str(p))
    if not dirs:
        print(f"[SKIP] no result dirs found under: {dataset_root}")
        return

    ok, fail = 0, 0
    for d in dirs:
        try:
            out_c, out_r, meta = load_and_filter(d, win=win, poly=poly)

            out_c_path = d / OUT_CONCAT
            out_r_path = d / OUT_RM
            out_c.to_csv(out_c_path, index=False)
            out_r.to_csv(out_r_path, index=False)

            p = plot_filter(d, dataset_root, out_c, out_r, meta)

            rel = d.relative_to(dataset_root)
            print(f"[OK]  {dataset_root.name}/{rel} -> {OUT_CONCAT}, {OUT_RM}, {p.name}")
            ok += 1
        except Exception as e:
            try:
                rel = d.relative_to(dataset_root)
            except Exception:
                rel = d
            print(f"[FAIL] {dataset_root.name}/{rel}: {e}")
            fail += 1

    print(f"\nDONE ({dataset_root.name}): ok={ok} fail={fail} win={win} poly={poly}\n")


def main():
    results_root = find_results_root()

    WIN  = 11
    POLY = 2

    dataset_roots = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("BVCC_") or d.name.startswith("SOMOS_"):
            dataset_roots.append(d)

    if not dataset_roots:
        raise FileNotFoundError(f"no dataset dirs found under {results_root} (expected BVCC_* / SOMOS_*)")

    for ds_root in dataset_roots:
        process_dataset(ds_root, win=WIN, poly=POLY)


if __name__ == "__main__":
    main()
