import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False


CONCAT_FILE = "global_mos_concat.csv"
RM_FILE = "global_mos_running_mean.csv"

PER_CASE_TXT = "end_mos_deviations.txt"

OUT_CASES_CSV = "end_mos_deviations_per_case.csv"
OUT_SUMMARY_TXT = "end_mos_deviation_summary.txt"

VARIANT_DIRS = ["original", "no_pause", "long_pause", "long_pause_noise"]

SKIP_NAMES = {
    "__pycache__", ".git", ".idea", ".vscode",
    "_debug", "packets", "utils", "without_dw",
}

TRAIN_INFO = {
    19: "w2v2-large + Transformer",
    18: "w2v2-base + Transformer",
    23: "w2v2-large + BiLSTM",
    24: "w2v2-base + BiLSTM",
    27: "w2v2-large + Conv",
    26: "w2v2-base + Conv",
}

# smoothing params
WIN = 11
POLY = 2


def pick_first_existing(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing column for {what}. have={list(df.columns)} candidates={candidates}")


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


def parse_dataset_and_train(dataset_root_name: str):
    m = re.match(r"^(BVCC|SOMOS)_(\d+)$", dataset_root_name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def iter_result_dirs(dataset_root: Path):
    for d in dataset_root.rglob("*"):
        if not d.is_dir():
            continue
        if any(p in SKIP_NAMES for p in d.parts):
            continue
        if (d / CONCAT_FILE).exists() and (d / RM_FILE).exists():
            yield d


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
        "base_system_id": base_system_id,
        "mode": mode,
        "system_id": system_id,
        "variant": variant,
        "rel_path": str(rel),
    }


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
    if w < 3:
        return y.copy()

    if _HAS_SAVGOL:
        p = int(poly)
        if p >= w:
            p = max(1, w - 2)
        return savgol_filter(y, window_length=w, polyorder=p, mode="interp")

    s = pd.Series(y)
    return s.rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)


def _last_values_with_filter(
    df: pd.DataFrame,
    x_candidates: list[str],
    pred_candidates: list[str],
    tgt_col: str,
    what: str,
    win: int,
    poly: int,
):
    x = pick_first_existing(df, x_candidates, f"{what} x-axis")
    pred = pick_first_existing(df, pred_candidates, f"{what} pred")

    use = df[[x, pred, tgt_col]].dropna().copy()
    if use.empty:
        raise ValueError(f"{what}: empty after dropna()")

    use = use.sort_values(x).reset_index(drop=True)

    t_end = float(use[x].iloc[-1])
    pred_raw = use[pred].to_numpy(dtype=np.float64)
    pred_filt = smooth_series(pred_raw, win=win, poly=poly)

    pred_end_raw = float(pred_raw[-1])
    pred_end_filt = float(pred_filt[-1])
    tgt_end = float(use[tgt_col].to_numpy(dtype=np.float64)[-1])

    return {
        "t_end": t_end,
        "pred_col": pred,
        "tgt_col": tgt_col,
        "pred_end_raw": pred_end_raw,
        "pred_end_filt": pred_end_filt,
        "tgt_end": tgt_end,
    }


def load_end_values(csv_dir: Path, win: int, poly: int):
    df_c = pd.read_csv(csv_dir / CONCAT_FILE)
    df_r = pd.read_csv(csv_dir / RM_FILE)

    tc = pick_first_existing(
        df_c,
        ["mos_target_concat_durw", "mos_target_concat_mean", "mos_target_concat_avg", "mos_target_concat", "mos_target"],
        "concat target",
    )
    tr = pick_first_existing(
        df_r,
        ["mos_target_rm_durw", "mos_target_rm_mean", "mos_target_rm", "mos_target"],
        "rm target",
    )

    concat = _last_values_with_filter(
        df_c,
        ["seconds", "elapsed_s", "time_s"],
        ["mos_pred_concat", "mos_pred", "mos"],
        tc,
        "concat",
        win=win,
        poly=poly,
    )

    rm = _last_values_with_filter(
        df_r,
        ["elapsed_s", "seconds", "time_s"],
        ["mos_pred_rm_durw", "mos_pred_rm_mean", "mos_pred_rm", "mos_running_mean", "mos_pred", "mos"],
        tr,
        "rm",
        win=win,
        poly=poly,
    )

    return {"concat": concat, "rm": rm}


def _fmt(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def _fmt_signed(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "nan"
    return f"{float(x):+.{nd}f}"


def write_per_case_txt(
    out_path: Path,
    dataset: str,
    train: int,
    train_label: str,
    dataset_root: Path,
    csv_dir: Path,
    meta: dict,
    ev: dict,
    win: int,
    poly: int,
):
    rel = str(csv_dir.relative_to(dataset_root))

    c = ev["concat"]
    r = ev["rm"]

    d_ct_raw = c["pred_end_raw"] - c["tgt_end"]
    d_rt_raw = r["pred_end_raw"] - r["tgt_end"]
    d_cr_raw = c["pred_end_raw"] - r["pred_end_raw"]

    d_ct_f = c["pred_end_filt"] - c["tgt_end"]
    d_rt_f = r["pred_end_filt"] - r["tgt_end"]
    d_cr_f = c["pred_end_filt"] - r["pred_end_filt"]

    lines = [
        f"dataset: {dataset}",
        f"train: {train} ({train_label})",
        f"case: {rel}",
        f"base_system_id: {meta['base_system_id']}",
        f"mode: {meta['mode']}",
        f"variant: {meta['variant']}",
        "",
        f"smoothing: win={win} poly={poly} (savgol={'yes' if _HAS_SAVGOL else 'no'})",
        "",
        f"end_seconds_concat: {_fmt(c['t_end'], nd=3)}",
        f"end_seconds_rm:     {_fmt(r['t_end'], nd=3)}",
        "",
        "RAW (end values)",
        f"  concat_pred_end: {_fmt(c['pred_end_raw'])}   (col={c['pred_col']})",
        f"  concat_tgt_end:  {_fmt(c['tgt_end'])}    (col={c['tgt_col']})",
        f"  rm_pred_end:     {_fmt(r['pred_end_raw'])}   (col={r['pred_col']})",
        f"  rm_tgt_end:      {_fmt(r['tgt_end'])}    (col={r['tgt_col']})",
        "",
        "RAW (deltas)",
        f"  delta_concat_minus_target: {_fmt_signed(d_ct_raw)}",
        f"  delta_rm_minus_target:     {_fmt_signed(d_rt_raw)}",
        f"  delta_concat_minus_rm:     {_fmt_signed(d_cr_raw)}",
        "",
        "FILTERED (end values)",
        f"  concat_pred_end_filt: {_fmt(c['pred_end_filt'])}",
        f"  rm_pred_end_filt:     {_fmt(r['pred_end_filt'])}",
        "",
        "FILTERED (deltas)",
        f"  delta_concat_minus_target_filt: {_fmt_signed(d_ct_f)}",
        f"  delta_rm_minus_target_filt:     {_fmt_signed(d_rt_f)}",
        f"  delta_concat_minus_rm_filt:     {_fmt_signed(d_cr_f)}",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _summ(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mae": float(np.mean(np.abs(arr))),
        "pos": int(np.sum(arr > 0)),
        "neg": int(np.sum(arr < 0)),
        "zero": int(np.sum(arr == 0)),
    }


def write_summary(path: Path, df_cases: pd.DataFrame, fails: list[str], win: int, poly: int):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"end MOS deviations summary | smoothing win={win} poly={poly} (savgol={'yes' if _HAS_SAVGOL else 'no'})\n\n")

        if df_cases.empty:
            f.write("no cases found\n")
        else:
            for ds in ["BVCC", "SOMOS"]:
                dsd = df_cases[df_cases["dataset"] == ds].copy()
                if dsd.empty:
                    continue

                f.write(f"{ds}\n\n")

                trains = sorted(dsd["train"].dropna().unique().tolist())
                for tr in trains:
                    g = dsd[dsd["train"] == tr]
                    label = TRAIN_INFO.get(int(tr), f"train{int(tr)}")
                    f.write(f"train{int(tr)} {label}\n")

                    for key_raw, key_f, title in [
                        ("d_concat_target_raw", "d_concat_target_filt", "concat - target"),
                        ("d_rm_target_raw", "d_rm_target_filt", "rm - target"),
                        ("d_concat_rm_raw", "d_concat_rm_filt", "concat - rm"),
                    ]:
                        s_raw = _summ(g[key_raw].to_numpy(dtype=np.float64))
                        s_f = _summ(g[key_f].to_numpy(dtype=np.float64))

                        if s_raw is None and s_f is None:
                            f.write(f"  {title}: no data\n")
                            continue

                        if s_raw is not None:
                            f.write(
                                "  "
                                f"{title} RAW: "
                                f"mean={_fmt_signed(s_raw['mean'])} "
                                f"mae={_fmt(s_raw['mae'])} "
                                f"min={_fmt_signed(s_raw['min'])} "
                                f"max={_fmt_signed(s_raw['max'])} "
                                f"median={_fmt_signed(s_raw['median'])} "
                                f"n={s_raw['n']} "
                                f"(pos={s_raw['pos']} neg={s_raw['neg']} zero={s_raw['zero']})\n"
                            )
                        if s_f is not None:
                            f.write(
                                "  "
                                f"{title} FILT: "
                                f"mean={_fmt_signed(s_f['mean'])} "
                                f"mae={_fmt(s_f['mae'])} "
                                f"min={_fmt_signed(s_f['min'])} "
                                f"max={_fmt_signed(s_f['max'])} "
                                f"median={_fmt_signed(s_f['median'])} "
                                f"n={s_f['n']} "
                                f"(pos={s_f['pos']} neg={s_f['neg']} zero={s_f['zero']})\n"
                            )

                        f.write("\n")

                    f.write("\n")

        if fails:
            f.write("failures\n")
            for line in fails:
                f.write(line.rstrip() + "\n")


def main():
    results_root = find_results_root()

    dataset_roots = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        ds, tr = parse_dataset_and_train(d.name)
        if ds in {"BVCC", "SOMOS"} and tr is not None:
            dataset_roots.append(d)

    if not dataset_roots:
        raise FileNotFoundError(f"no dataset dirs found under {results_root} (expected BVCC_* / SOMOS_*)")

    rows = []
    fails = []

    ok = 0
    fail = 0

    for ds_root in dataset_roots:
        ds, tr = parse_dataset_and_train(ds_root.name)
        train_label = TRAIN_INFO.get(int(tr), f"train{int(tr)}")

        dirs = sorted(set(iter_result_dirs(ds_root)), key=lambda p: str(p))
        if not dirs:
            fails.append(f"{ds_root.name}: no result dirs with both csvs")
            continue

        for d in dirs:
            rel = d.relative_to(ds_root)
            try:
                meta = parse_path_meta(ds_root, d)
                ev = load_end_values(d, win=WIN, poly=POLY)

                c = ev["concat"]
                r = ev["rm"]

                d_ct_raw = c["pred_end_raw"] - c["tgt_end"]
                d_rt_raw = r["pred_end_raw"] - r["tgt_end"]
                d_cr_raw = c["pred_end_raw"] - r["pred_end_raw"]

                d_ct_f = c["pred_end_filt"] - c["tgt_end"]
                d_rt_f = r["pred_end_filt"] - r["tgt_end"]
                d_cr_f = c["pred_end_filt"] - r["pred_end_filt"]

                write_per_case_txt(
                    d / PER_CASE_TXT,
                    ds,
                    int(tr),
                    train_label,
                    ds_root,
                    d,
                    meta,
                    ev,
                    win=WIN,
                    poly=POLY,
                )

                rows.append({
                    "dataset": ds,
                    "train": int(tr),
                    "train_label": train_label,
                    **meta,

                    "concat_sec": c["t_end"],
                    "rm_sec": r["t_end"],

                    "concat_pred_end_raw": c["pred_end_raw"],
                    "concat_pred_end_filt": c["pred_end_filt"],
                    "concat_tgt_end": c["tgt_end"],

                    "rm_pred_end_raw": r["pred_end_raw"],
                    "rm_pred_end_filt": r["pred_end_filt"],
                    "rm_tgt_end": r["tgt_end"],

                    "d_concat_target_raw": d_ct_raw,
                    "d_rm_target_raw": d_rt_raw,
                    "d_concat_rm_raw": d_cr_raw,

                    "d_concat_target_filt": d_ct_f,
                    "d_rm_target_filt": d_rt_f,
                    "d_concat_rm_filt": d_cr_f,
                })

                ok += 1
            except Exception as e:
                fails.append(f"{ds_root.name}/{rel}: {e}")
                fail += 1

    df = pd.DataFrame(rows)

    out_csv = results_root / OUT_CASES_CSV
    df.to_csv(out_csv, index=False)

    out_txt = results_root / OUT_SUMMARY_TXT
    write_summary(out_txt, df, fails, win=WIN, poly=POLY)

    print(f"[done] ok={ok} fail={fail}")
    print(f"[wrote] {out_csv}")
    print(f"[wrote] {out_txt}")
    print(f"[wrote per-case] {PER_CASE_TXT} in each leaf dir with csvs")


if __name__ == "__main__":
    main()
