from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -------------------------
# config
# -------------------------

RESULTS_BLOCK_DIRNAME = "results_block"

# variants
VARIANT_DIRS = ["original", "no_pause", "long_pause", "long_pause_noise"]

GLOBAL_CONCAT_FILE = "global_mos_concat.csv"
GLOBAL_RM_FILE = "global_mos_running_mean.csv"
BLOCK_CONCAT_FILE = "block_mos_concat.csv"
BLOCK_RM_FILE = "block_mos_running_mean.csv"

OUT_TXT = "correlations_results_block_report.txt"

TRAIN_LABEL = {
    19: "w2v2-large + Transformer",
    18: "w2v2-base + Transformer",
    23: "w2v2-large + BiLSTM",
    24: "w2v2-base + BiLSTM",
    27: "w2v2-large + Conv",
    26: "w2v2-base + Conv",
}


# -------------------------
# helpers
# -------------------------

def find_results_block_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for c in [
        script_dir / RESULTS_BLOCK_DIRNAME,
        script_dir.parent / RESULTS_BLOCK_DIRNAME,
        Path.cwd() / RESULTS_BLOCK_DIRNAME,
    ]:
        if c.exists() and c.is_dir():
            return c.resolve()
    raise FileNotFoundError(f"could not find {RESULTS_BLOCK_DIRNAME}/")


def pick_first_existing(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing column for {what}. have={list(df.columns)} candidates={candidates}")


def srcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    if _HAS_SCIPY:
        return float(spearmanr(a, b).correlation)
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def mean_median(vals: List[float]) -> Tuple[float, float, int]:
    v = [float(x) for x in vals if np.isfinite(x)]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.median(v)), int(len(v))


def fmt(x: float, nd: int = 6) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def base_dataset_name(ds_dir_name: str) -> str:
    return ds_dir_name.split("_", 1)[0].lower()


def block_seconds_from_ds_dir(ds_dir_name: str) -> Optional[int]:
    m = re.fullmatch(r"(bvcc|somos)_(\d+)", ds_dir_name.lower())
    if not m:
        return None
    return int(m.group(2))


def find_dataset_dirs(results_block: Path) -> List[Path]:
    # supports: bvcc, bvcc_1, bvcc_20, somos, somos_1, somos_20, ...
    pat = re.compile(r"^(bvcc|somos)(?:_\d+)?$", re.IGNORECASE)
    out = [p for p in results_block.iterdir() if p.is_dir() and pat.match(p.name)]
    return sorted(out, key=lambda p: p.name.lower())


# -------------------------
# path scanning
# -------------------------

def iter_train_dirs(ds_root: Path) -> List[Tuple[int, Path]]:
    out = []
    for d in sorted(ds_root.iterdir()):
        if d.is_dir() and d.name.isdigit():
            out.append((int(d.name), d))
    return out


def iter_variant_dirs(train_dir: Path) -> List[Path]:
    # expected: results_block/<dataset_dir>/<train>/<run>/<sys>/<variant>/*.csv
    out: List[Path] = []
    for run_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        for vdir in run_dir.rglob("*"):
            if not vdir.is_dir():
                continue
            if vdir.name not in VARIANT_DIRS:
                continue
            if any((vdir / fn).exists() for fn in [GLOBAL_CONCAT_FILE, GLOBAL_RM_FILE, BLOCK_CONCAT_FILE, BLOCK_RM_FILE]):
                out.append(vdir)
    return sorted(set(out), key=lambda p: str(p))


def parse_case(vdir: Path) -> Tuple[str, str]:
    sys_dir = vdir.parent.name  # sysXXXX / rev_sysXXXX / rand_sysXXXX / dlg_* ...
    mode = "forward"
    if sys_dir.startswith("rev_") or sys_dir.startswith("rev"):
        mode = "reverse"
    if sys_dir.startswith("rand_") or sys_dir.startswith("rand"):
        mode = "random"
    return sys_dir, mode


def ensure_block_mid(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "block_mid_s" in df.columns:
        return df, "block_mid_s"
    if "block_mid" in df.columns:
        return df, "block_mid"
    if "block_start_s" in df.columns and "block_end_s" in df.columns:
        df = df.copy()
        df["block_mid_s"] = 0.5 * (df["block_start_s"].astype(float) + df["block_end_s"].astype(float))
        return df, "block_mid_s"
    raise KeyError("no usable block x axis (need block_mid_s or block_start_s+block_end_s)")


# -------------------------
# correlations (same as before)
# -------------------------

def corr_global_concat_to_target(vdir: Path) -> float:
    df = pd.read_csv(vdir / GLOBAL_CONCAT_FILE)
    x = pick_first_existing(df, ["seconds", "elapsed_s", "time_s"], "global concat x")
    y = pick_first_existing(df, ["mos_pred_concat", "mos_pred", "mos"], "global concat pred")
    t = pick_first_existing(df, ["mos_target_concat_durw", "mos_target_concat", "mos_target"], "global concat tgt")
    df = df.sort_values(x).reset_index(drop=True)
    return srcc(df[y].to_numpy(), df[t].to_numpy())


def corr_global_rm_to_target(vdir: Path) -> float:
    df = pd.read_csv(vdir / GLOBAL_RM_FILE)
    x = pick_first_existing(df, ["elapsed_s", "seconds", "time_s"], "global rm x")
    y = pick_first_existing(df, ["mos_pred_rm_durw", "mos_pred_rm", "mos_pred", "mos"], "global rm pred")
    t = pick_first_existing(df, ["mos_target_rm_durw", "mos_target_rm", "mos_target"], "global rm tgt")
    df = df.sort_values(x).reset_index(drop=True)
    return srcc(df[y].to_numpy(), df[t].to_numpy())


def corr_block_concat_to_target(vdir: Path) -> float:
    df = pd.read_csv(vdir / BLOCK_CONCAT_FILE)
    df, x = ensure_block_mid(df)
    y = pick_first_existing(df, ["mos_pred_block_concat", "mos_pred", "mos"], "block concat pred")
    t = pick_first_existing(df, ["mos_target_block_durw", "mos_target"], "block concat tgt")
    df = df.sort_values(x).reset_index(drop=True)
    return srcc(df[y].to_numpy(), df[t].to_numpy())


def corr_block_rm_to_target(vdir: Path) -> float:
    df = pd.read_csv(vdir / BLOCK_RM_FILE)
    df, x = ensure_block_mid(df)
    y = pick_first_existing(df, ["mos_pred_block_rm_durw", "mos_pred_block_rm", "mos_pred", "mos"], "block rm pred")
    t = pick_first_existing(df, ["mos_target_block_durw", "mos_target"], "block rm tgt")
    df = df.sort_values(x).reset_index(drop=True)
    return srcc(df[y].to_numpy(), df[t].to_numpy())


def corr_global_concat_vs_rm(vdir: Path) -> float:
    dfc = pd.read_csv(vdir / GLOBAL_CONCAT_FILE)
    dfr = pd.read_csv(vdir / GLOBAL_RM_FILE)

    xc = pick_first_existing(dfc, ["seconds", "elapsed_s", "time_s"], "global concat x")
    yc = pick_first_existing(dfc, ["mos_pred_concat", "mos_pred", "mos"], "global concat pred")
    xr = pick_first_existing(dfr, ["elapsed_s", "seconds", "time_s"], "global rm x")
    yr = pick_first_existing(dfr, ["mos_pred_rm_durw", "mos_pred_rm", "mos_pred", "mos"], "global rm pred")

    dfc = dfc.sort_values(xc).reset_index(drop=True)
    dfr = dfr.sort_values(xr).reset_index(drop=True)

    a = dfc[[xc, yc]].dropna().rename(columns={xc: "t"})
    b = dfr[[xr, yr]].dropna().rename(columns={xr: "t"})
    a = a.sort_values("t")
    b = b.sort_values("t")
    m = pd.merge_asof(a, b, on="t", direction="nearest", tolerance=0.2, suffixes=("_c", "_r")).dropna()
    return srcc(m[yc].to_numpy(), m[yr].to_numpy())


def corr_block_concat_vs_rm(vdir: Path) -> float:
    dfc = pd.read_csv(vdir / BLOCK_CONCAT_FILE)
    dfr = pd.read_csv(vdir / BLOCK_RM_FILE)

    dfc, xc = ensure_block_mid(dfc)
    dfr, xr = ensure_block_mid(dfr)

    yc = pick_first_existing(dfc, ["mos_pred_block_concat", "mos_pred", "mos"], "block concat pred")
    yr = pick_first_existing(dfr, ["mos_pred_block_rm_durw", "mos_pred_block_rm", "mos_pred", "mos"], "block rm pred")

    a = dfc[[xc, yc]].dropna().rename(columns={xc: "t"})
    b = dfr[[xr, yr]].dropna().rename(columns={xr: "t"})
    m = pd.merge(a, b, on="t", how="inner").dropna()
    return srcc(m[yc].to_numpy(), m[yr].to_numpy())


GRIDS_TO_TARGET = [
    ("concat", corr_global_concat_to_target, GLOBAL_CONCAT_FILE),
    ("rm", corr_global_rm_to_target, GLOBAL_RM_FILE),
    ("block_concat", corr_block_concat_to_target, BLOCK_CONCAT_FILE),
    ("block_rm", corr_block_rm_to_target, BLOCK_RM_FILE),
]

GRIDS_CONSISTENCY = [
    ("concat_vs_rm", corr_global_concat_vs_rm, (GLOBAL_CONCAT_FILE, GLOBAL_RM_FILE)),
    ("block_concat_vs_rm", corr_block_concat_vs_rm, (BLOCK_CONCAT_FILE, BLOCK_RM_FILE)),
]


def compute_cases(ds_root: Path, ds_dir_name: str) -> List[Dict]:
    # ds_root is results_block/<ds_dir_name> (e.g. bvcc_1)
    cases: List[Dict] = []
    base = base_dataset_name(ds_dir_name)
    blk = block_seconds_from_ds_dir(ds_dir_name)

    for train_id, train_dir in iter_train_dirs(ds_root):
        label = TRAIN_LABEL.get(train_id, f"train{train_id}")
        for vdir in iter_variant_dirs(train_dir):
            sys_id, mode = parse_case(vdir)
            variant = vdir.name

            rec = {
                "dataset_base": base,
                "dataset_dir": ds_dir_name,
                "block_s": blk,
                "train": train_id,
                "label": label,
                "sys": sys_id,
                "mode": mode,
                "variant": variant,
                "vdir": vdir,
                "to_target": {},
                "consistency": {},
            }

            for grid_name, fn, required in GRIDS_TO_TARGET:
                if isinstance(required, str) and not (vdir / required).exists():
                    continue
                try:
                    rec["to_target"][grid_name] = float(fn(vdir))
                except Exception:
                    rec["to_target"][grid_name] = float("nan")

            for grid_name, fn, reqs in GRIDS_CONSISTENCY:
                if not all((vdir / r).exists() for r in reqs):
                    continue
                try:
                    rec["consistency"][grid_name] = float(fn(vdir))
                except Exception:
                    rec["consistency"][grid_name] = float("nan")

            cases.append(rec)

    return cases


# -------------------------
# reporting (existing + block-size compare)
# -------------------------

def report_dataset_like_before(f, title: str, cases: List[Dict]) -> None:
    f.write("=" * 20 + "\n")
    f.write(title + "\n")
    f.write("=" * 20 + "\n\n")

    trains = sorted(set(int(c["train"]) for c in cases))
    if not trains:
        f.write("no cases found.\n\n")
        return

    f.write("1) mean/median SRCC_filt to target (by train, grid)\n\n")
    for grid_name, _, _ in GRIDS_TO_TARGET:
        f.write(f"  grid={grid_name}\n")
        lines = []
        for tr in trains:
            vals = [c["to_target"].get(grid_name, np.nan) for c in cases if int(c["train"]) == tr]
            m, med, n = mean_median(vals)
            lines.append((m, tr, med, n))
        lines.sort(key=lambda x: (-(x[0] if np.isfinite(x[0]) else -1e9), x[1]))
        for m, tr, med, n in lines:
            f.write(f"    train{tr} {TRAIN_LABEL.get(tr, f'train{tr}')}: mean={fmt(m)} median={fmt(med)} n={n}\n")
        f.write("\n")

    f.write("2) win counts (best SRCC_filt) per train (by grid)\n\n")
    for grid_name, _, _ in GRIDS_TO_TARGET:
        win: Dict[int, int] = {tr: 0 for tr in trains}
        keys = sorted(set((c["sys"], c["mode"], c["variant"]) for c in cases))
        for k in keys:
            pool = [c for c in cases if (c["sys"], c["mode"], c["variant"]) == k]
            best_tr = None
            best_val = -1e18
            for c in pool:
                v = c["to_target"].get(grid_name, np.nan)
                if np.isfinite(v) and v > best_val:
                    best_val = float(v)
                    best_tr = int(c["train"])
            if best_tr is not None:
                win[best_tr] += 1

        f.write(f"  grid={grid_name}\n")
        for tr, cnt in sorted(win.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"    {TRAIN_LABEL.get(tr, f'train{tr}')} (train{tr}): {cnt}\n")
        f.write("\n")

    f.write("3) pairwise mean deltas (SRCC_filt to target) (train_b - train_a)\n")
    for grid_name, _, _ in GRIDS_TO_TARGET:
        f.write(f"\n  grid={grid_name}\n")

        inst: Dict[Tuple[str, str, str], Dict[int, float]] = {}
        for c in cases:
            key = (c["sys"], c["mode"], c["variant"])
            inst.setdefault(key, {})
            inst[key][int(c["train"])] = c["to_target"].get(grid_name, np.nan)

        pairs = []
        for a, b in itertools.combinations(trains, 2):
            deltas = []
            wins_b = 0
            wins_a = 0
            ties = 0
            for key, mp in inst.items():
                va = mp.get(a, np.nan)
                vb = mp.get(b, np.nan)
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                d = float(vb - va)
                deltas.append(d)
                if vb > va:
                    wins_b += 1
                elif va > vb:
                    wins_a += 1
                else:
                    ties += 1
            mean_d, med_d, n = mean_median(deltas)
            if n > 0:
                pairs.append((mean_d, a, b, med_d, n, wins_b, wins_a, ties))

        pairs.sort(key=lambda x: -(x[0] if np.isfinite(x[0]) else -1e9))
        for mean_d, a, b, med_d, n, wins_b, wins_a, ties in pairs:
            f.write(
                f"  {TRAIN_LABEL.get(b)} (train{b}) - {TRAIN_LABEL.get(a)} (train{a}): "
                f"mean={fmt(mean_d)} median={fmt(med_d)} n={n} wins_b={wins_b} wins_a={wins_a} ties={ties}\n"
            )
    f.write("\n")

    f.write("4) concat-vs-rm SRCC_filt (by train)\n")
    for grid_name, _, _ in GRIDS_CONSISTENCY:
        f.write(f"\n  grid={grid_name}\n")
        lines = []
        for tr in trains:
            vals = [c["consistency"].get(grid_name, np.nan) for c in cases if int(c["train"]) == tr]
            m, med, n = mean_median(vals)
            lines.append((m, tr, med, n))
        lines.sort(key=lambda x: (-(x[0] if np.isfinite(x[0]) else -1e9), x[1]))
        for m, tr, med, n in lines:
            f.write(f"  train{tr} {TRAIN_LABEL.get(tr)}: mean={fmt(m)} median={fmt(med)} n={n}\n")
    f.write("\n")


def _case_key(c: Dict) -> Tuple[int, str, str, str]:
    # match between block sizes (same train/sys/mode/variant)
    return (int(c["train"]), str(c["sys"]), str(c["mode"]), str(c["variant"]))


def report_blocksize_comparison(
    f,
    dataset_base: str,
    cases_1: List[Dict],
    cases_20: List[Dict],
    block_a: int = 1,
    block_b: int = 20,
) -> None:
    # build maps
    m1 = {_case_key(c): c for c in cases_1}
    m2 = {_case_key(c): c for c in cases_20}
    keys = sorted(set(m1.keys()) & set(m2.keys()))
    trains = sorted(set(k[0] for k in keys))

    f.write("#" * 60 + "\n")
    f.write(f"{dataset_base.upper()} — block-size comparison ({block_b}s vs {block_a}s)\n")
    f.write("#" * 60 + "\n\n")

    if not keys:
        f.write("no overlapping cases between block sizes.\n\n")
        return

    # helper: get value for grid from a case
    def get_val(case: Dict, group: str, grid: str) -> float:
        v = case[group].get(grid, np.nan)
        return float(v) if v is not None else float("nan")

    # compare for both groups
    compare_specs = []
    for grid_name, _, _ in GRIDS_TO_TARGET:
        compare_specs.append(("to_target", grid_name))
    for grid_name, _, _ in GRIDS_CONSISTENCY:
        compare_specs.append(("consistency", grid_name))

    f.write("A) mean/median delta = (block20 - block1)  (by train, grid)\n\n")
    for group, grid in compare_specs:
        f.write(f"  grid={grid}\n")
        for tr in trains:
            deltas = []
            for k in keys:
                if k[0] != tr:
                    continue
                v1 = get_val(m1[k], group, grid)
                v2 = get_val(m2[k], group, grid)
                if np.isfinite(v1) and np.isfinite(v2):
                    deltas.append(v2 - v1)
            md, med, n = mean_median(deltas)
            f.write(f"    train{tr} {TRAIN_LABEL.get(tr, f'train{tr}')}: mean_delta={fmt(md)} median_delta={fmt(med)} n={n}\n")
        f.write("\n")

    f.write("B) agreement SRCC between block1-values and block20-values (by train, grid)\n\n")
    for group, grid in compare_specs:
        f.write(f"  grid={grid}\n")
        for tr in trains:
            a = []
            b = []
            for k in keys:
                if k[0] != tr:
                    continue
                v1 = get_val(m1[k], group, grid)
                v2 = get_val(m2[k], group, grid)
                if np.isfinite(v1) and np.isfinite(v2):
                    a.append(v1)
                    b.append(v2)
            r = srcc(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
            f.write(f"    train{tr} {TRAIN_LABEL.get(tr, f'train{tr}')}: SRCC(block1,block20)={fmt(r)} n={len(a)}\n")
        f.write("\n")

    f.write("C) win counts: how often block20 > block1 (by train, grid)\n\n")
    for group, grid in compare_specs:
        f.write(f"  grid={grid}\n")
        for tr in trains:
            w20 = 0
            w1 = 0
            ties = 0
            n = 0
            for k in keys:
                if k[0] != tr:
                    continue
                v1 = get_val(m1[k], group, grid)
                v2 = get_val(m2[k], group, grid)
                if not (np.isfinite(v1) and np.isfinite(v2)):
                    continue
                n += 1
                if v2 > v1:
                    w20 += 1
                elif v1 > v2:
                    w1 += 1
                else:
                    ties += 1
            f.write(f"    train{tr} {TRAIN_LABEL.get(tr, f'train{tr}')}: wins_20={w20} wins_1={w1} ties={ties} n={n}\n")
        f.write("\n")


def main() -> None:
    results_block = find_results_block_root()
    ds_dirs = find_dataset_dirs(results_block)

    # collect all cases grouped by base dataset and block seconds
    grouped: Dict[str, Dict[Optional[int], List[Dict]]] = {"bvcc": {}, "somos": {}}

    for ds_dir in ds_dirs:
        base = base_dataset_name(ds_dir.name)
        blk = block_seconds_from_ds_dir(ds_dir.name)  # None if legacy
        cases = compute_cases(ds_dir, ds_dir.name)
        grouped.setdefault(base, {})
        grouped[base].setdefault(blk, [])
        grouped[base][blk].extend(cases)

    out_path = results_block / OUT_TXT
    with open(out_path, "w", encoding="utf-8") as f:
        # 1) normal reports per folder (like before)
        for base in ["bvcc", "somos"]:
            blk_map = grouped.get(base, {})
            for blk, cases in sorted(blk_map.items(), key=lambda x: (-1 if x[0] is None else x[0])):
                title = base.upper() if blk is None else f"{base.upper()}_{blk}"
                report_dataset_like_before(f, title, cases)

        # 2) block1 vs block20 compare (only if both exist)
        for base in ["bvcc", "somos"]:
            blk_map = grouped.get(base, {})
            if 1 in blk_map and 20 in blk_map:
                report_blocksize_comparison(
                    f,
                    dataset_base=base,
                    cases_1=blk_map[1],
                    cases_20=blk_map[20],
                    block_a=1,
                    block_b=20,
                )

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()