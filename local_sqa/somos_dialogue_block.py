from __future__ import annotations

import csv
import itertools
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

import audiofile

from rVADfast import rVADfast
from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio


# -------------------------
# config
# -------------------------

TRAIN_IDS = [18, 19, 23, 24, 26, 27]
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

SOMOS_TEST_LIST = Path("/net/db/somos/training_files/split1/clean/test_mos_list.txt")
SOMOS_WAV_ROOT = Path("/net/db/somos/audios")

SYSTEM_IDS = ["057", "061", "110", "124", "191"]
MODES_TO_RUN = ["forward", "reverse", "random"]

TARGET_SECONDS = 180.0
BLOCK_SECONDS = 1.0  # <-- requested 1s
START_INDEX = 0
MAX_FILES_PER_SIDE = 5000

ALLOW_REPEAT = True
SKIP_IF_DONE = True
RANDOM_SEED = 1234
FALLBACK_TO_CPU = False

VARIANTS = ["original", "no_pause", "long_pause", "long_pause_noise"]

# pause configs for variants
INSERT_PAUSE_SECONDS = 5.0
NOISE_STD = 0.003

# rVAD params
PAD_MS = 40
MERGE_GAP_MS = 120
MIN_SPEECH_MS = 80
MIN_PROC_MS_FALLBACK = 250


# -------------------------
# utils
# -------------------------

def cpu_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def gpu_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


AUDIO_LOADER = LoadAudio(
    audio_path_keys="audio_path.observation",
    target_sampling_rate=SAMPLING_RATE,
    resample=True,
)


def load_wav(path: Path) -> np.ndarray:
    ex = {"audio_path": {"observation": str(path)}}
    ex = AUDIO_LOADER(ex)
    wav = ex["audio"].astype(np.float32)
    if wav.ndim != 1:
        raise ValueError(f"expected 1d audio, got {wav.shape} for {path}")
    return wav


def durw_mean(values: List[float], durations: List[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(durations, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(d) & (d > 0)
    if not np.any(m):
        return float("nan")
    v = v[m]
    d = d[m]
    return float(np.sum(v * d) / np.sum(d))


def safe_infer_global(predictor_gpu, predictor_cpu, wav: np.ndarray, tag: str):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError("wav must be 1d")

    wav_batch = wav[None, :]
    num_samples = [int(wav.shape[0])]

    cpu_before = cpu_memory_mb()
    gpu_reset()
    t0 = time.time()

    try:
        with torch.inference_mode():
            preds, _, _ = predictor_gpu(wav_batch, num_samples)

        t1 = time.time()
        cpu_after = cpu_memory_mb()

        runtime = t1 - t0
        mem = gpu_peak_mb() + max(cpu_after - cpu_before, 0.0)

        mos = float(np.asarray(preds).reshape(-1)[0])
        return True, runtime, mem, mos, "gpu"

    except RuntimeError as e:
        msg = str(e).lower()
        print(f"[{tag}] runtimeerror: {e}")

        if not FALLBACK_TO_CPU or predictor_cpu is None:
            return False, None, None, None, "fail"

        if "out of memory" in msg or "cudnn" in msg or "non-contiguous" in msg:
            print(f"[{tag}] -> cpu fallback...")
            try:
                with torch.inference_mode():
                    preds, _, _ = predictor_cpu(wav_batch, num_samples)
                mos = float(np.asarray(preds).reshape(-1)[0])
                return True, None, None, mos, "cpu"
            except Exception as e2:
                print(f"[{tag}] cpu fallback failed: {e2}")

        return False, None, None, None, "fail"


def parse_somos_list(list_path: Path) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for i, ln in enumerate(list_path.read_text().splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        if i == 0 and ln.lower().startswith("utteranceid"):
            continue
        utt, mos = ln.split(",")
        items.append((utt.strip(), float(mos)))
    return items


def extract_system_id(utterance_id: str) -> str:
    u = utterance_id.strip()
    if u.endswith(".wav"):
        u = u[:-4]
    if "_" not in u:
        return "unknown"
    tail = u.split("_")[-1]
    if tail.isdigit():
        return tail.zfill(3)
    return "unknown"


def utt_to_path(utt: str) -> Path:
    u = utt.strip()
    if not u.endswith(".wav"):
        u = u + ".wav"
    return SOMOS_WAV_ROOT / u


def _make_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    return zlib.adler32(s) & 0xFFFFFFFF


def order_dialogue(packet: List["Seg"], mode: str, seed_tag: str) -> List["Seg"]:
    if mode == "forward":
        return packet
    if mode == "reverse":
        return list(reversed(packet))
    if mode == "random":
        seed = (_make_seed(seed_tag) ^ RANDOM_SEED) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(packet)).tolist()
        return [packet[i] for i in idx]
    raise ValueError(mode)


# -------------------------
# variants (rvad + pause)
# -------------------------

RVAD = rVADfast()


def _rvad_to_intervals_seconds(y: np.ndarray, sr: int):
    labels, ts = RVAD(y.astype(np.float64), sr)

    labels = np.asarray(labels).astype(bool)
    ts = np.asarray(ts, dtype=np.float64)

    if len(ts) == len(labels):
        ts = np.concatenate([ts, [len(y) / sr]])
    if len(ts) != len(labels) + 1:
        raise RuntimeError(f"unexpected rVADfast output: len(ts)={len(ts)} len(labels)={len(labels)}")

    segs = []
    in_seg = False
    s0 = 0.0

    for i, v in enumerate(labels):
        if v and not in_seg:
            s0 = float(ts[i])
            in_seg = True
        if in_seg and (not v):
            e0 = float(ts[i])
            if e0 > s0:
                segs.append((s0, e0))
            in_seg = False

    if in_seg:
        e0 = float(ts[-1])
        if e0 > s0:
            segs.append((s0, e0))

    return segs


def vad_intervals_rvad(y: np.ndarray, sr: int):
    segs_sec = _rvad_to_intervals_seconds(y, sr)
    if not segs_sec:
        return []

    pad = int((PAD_MS / 1000.0) * sr)
    merge_gap = int((MERGE_GAP_MS / 1000.0) * sr)
    min_len = int((MIN_SPEECH_MS / 1000.0) * sr)

    segs = []
    for s, e in segs_sec:
        a = max(int(round(s * sr)) - pad, 0)
        b = min(int(round(e * sr)) + pad, len(y))
        if b - a >= min_len:
            segs.append((a, b))

    if not segs:
        return []

    segs.sort()
    merged = [segs[0]]
    for a, b in segs[1:]:
        pa, pb = merged[-1]
        if a <= pb + merge_gap:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))

    return merged


def _make_pause(sr: int, seconds: float, kind: str, rng: np.random.Generator):
    n = int(round(seconds * sr))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if kind == "zero":
        return np.zeros(n, dtype=np.float32)
    if kind == "noise":
        x = rng.normal(0.0, NOISE_STD, size=n).astype(np.float32)
        return np.clip(x, -1.0, 1.0)
    raise ValueError(kind)


def _build_processed(y: np.ndarray, sr: int, intervals, pause_s: float, pause_kind: str, seed: int):
    if not intervals:
        return np.zeros(0, dtype=np.float32)

    rng = np.random.default_rng(seed)
    pause = _make_pause(sr, pause_s, pause_kind, rng)

    chunks = []
    for k, (s, e) in enumerate(intervals):
        chunks.append(y[s:e])
        if k < len(intervals) - 1 and len(pause) > 0:
            chunks.append(pause)

    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def make_variants(wav: np.ndarray, sr: int, seed: int):
    out = {"original": wav}
    intervals = vad_intervals_rvad(wav, sr)
    min_proc = int((MIN_PROC_MS_FALLBACK / 1000.0) * sr)

    if not intervals:
        out["no_pause"] = wav
        out["long_pause"] = wav
        out["long_pause_noise"] = wav
        return out

    y_cut = _build_processed(wav, sr, intervals, pause_s=0.0, pause_kind="zero", seed=seed ^ 0xA1B2C3D4)
    y_zero = _build_processed(wav, sr, intervals, pause_s=INSERT_PAUSE_SECONDS, pause_kind="zero", seed=seed ^ 0x11223344)
    y_noise = _build_processed(wav, sr, intervals, pause_s=INSERT_PAUSE_SECONDS, pause_kind="noise", seed=seed ^ 0x55667788)

    if len(y_cut) < min_proc:
        y_cut = wav
    if len(y_zero) < min_proc:
        y_zero = wav
    if len(y_noise) < min_proc:
        y_noise = wav

    out["no_pause"] = y_cut
    out["long_pause"] = y_zero
    out["long_pause_noise"] = y_noise
    return out


# -------------------------
# dialogue + block logic
# -------------------------

@dataclass
class Seg:
    utt: str
    mos_t: float
    path: Path
    dur_s: float
    variants: Dict[str, np.ndarray]


def build_dialogue_packet(items: List[Tuple[str, float]], sys_a: str, sys_b: str) -> List[Tuple[str, float, Path, float]]:
    a_list = []
    b_list = []
    for utt, mos in items:
        sid = extract_system_id(utt)
        if sid == sys_a:
            a_list.append((utt, mos))
        elif sid == sys_b:
            b_list.append((utt, mos))

    a_list = a_list[START_INDEX:START_INDEX + MAX_FILES_PER_SIDE]
    b_list = b_list[START_INDEX:START_INDEX + MAX_FILES_PER_SIDE]

    if not a_list or not b_list:
        return []

    out = []
    total_s = 0.0
    ia, ib = 0, 0

    while total_s < TARGET_SECONDS and (ia < len(a_list) or ib < len(b_list)):
        for pick_a in (True, False):  # A then B
            if total_s >= TARGET_SECONDS:
                break

            if pick_a:
                if ia >= len(a_list):
                    if not ALLOW_REPEAT:
                        continue
                    ia = 0
                utt, mos = a_list[ia]
                ia += 1
            else:
                if ib >= len(b_list):
                    if not ALLOW_REPEAT:
                        continue
                    ib = 0
                utt, mos = b_list[ib]
                ib += 1

            p = utt_to_path(utt)
            if not p.exists():
                continue

            try:
                info = audiofile.info(str(p))
                dur = float(info.duration)
            except Exception:
                y = load_wav(p)
                dur = float(len(y) / SAMPLING_RATE)

            out.append((utt, float(mos), p, float(dur)))
            total_s += float(dur)

    return out


def compute_blocks(
    segs: List[Seg],
    seg_pred_raw: List[float],
    variant: str,
    block_seconds: float,
    predictor_gpu,
    predictor_cpu,
    tag_prefix: str,
) -> Tuple[List[List[object]], List[List[object]]]:
    block_rm_rows: List[List[object]] = []
    block_concat_rows: List[List[object]] = []

    i = 0
    seg_off = 0
    seg_rem_s = 0.0
    bid = 0

    while i < len(segs):
        b_start = float(bid * block_seconds)
        need_s = float(block_seconds)
        used_s = 0.0

        audio_parts: List[np.ndarray] = []
        pred_parts: List[float] = []
        tgt_parts: List[float] = []
        dur_parts: List[float] = []
        pieces = 0

        while need_s > 1e-9 and i < len(segs):
            s = segs[i]
            wav = s.variants[variant]
            pred = float(seg_pred_raw[i])
            tgt = float(s.mos_t)

            if seg_rem_s <= 0.0:
                seg_rem_s = float(s.dur_s)
                seg_off = 0

            take_s = min(need_s, seg_rem_s)
            if take_s <= 1e-12:
                i += 1
                seg_rem_s = 0.0
                seg_off = 0
                continue

            rem_samples = max(len(wav) - seg_off, 0)
            if rem_samples <= 0:
                i += 1
                seg_rem_s = 0.0
                seg_off = 0
                continue

            frac = take_s / max(seg_rem_s, 1e-12)
            take_samples = int(round(frac * rem_samples))
            take_samples = max(1, min(take_samples, rem_samples))

            part = wav[seg_off:seg_off + take_samples]
            if part.size > 0:
                audio_parts.append(part)
                pred_parts.append(pred)
                tgt_parts.append(tgt)
                dur_parts.append(take_s)
                pieces += 1

            seg_off += take_samples
            seg_rem_s -= take_s
            need_s -= take_s
            used_s += take_s

            if seg_rem_s <= 1e-9 or seg_off >= len(wav):
                i += 1
                seg_rem_s = 0.0
                seg_off = 0

        if used_s <= 1e-9:
            break

        b_end = b_start + used_s
        b_mid = 0.5 * (b_start + b_end)

        pred_blk_rm = durw_mean(pred_parts, dur_parts)
        tgt_blk = durw_mean(tgt_parts, dur_parts)

        block_rm_rows.append([
            round(b_start, 6),
            round(b_end, 6),
            round(b_mid, 6),
            round(float(pred_blk_rm), 6) if np.isfinite(pred_blk_rm) else "",
            round(float(tgt_blk), 6) if np.isfinite(tgt_blk) else "",
            round(float(used_s), 6),
            int(pieces),
            round(float(block_seconds), 6),
        ])

        wav_cat = np.concatenate(audio_parts).astype(np.float32) if audio_parts else np.zeros(0, dtype=np.float32)
        ok, _, _, mos_pred_blk, _ = safe_infer_global(
            predictor_gpu, predictor_cpu, wav_cat, f"{tag_prefix}:block{bid:04d}:{variant}"
        )
        if not ok:
            break

        block_concat_rows.append([
            round(b_start, 6),
            round(b_end, 6),
            round(b_mid, 6),
            round(float(mos_pred_blk), 6),
            round(float(tgt_blk), 6) if np.isfinite(tgt_blk) else "",
            round(float(used_s), 6),
            int(pieces),
            round(float(block_seconds), 6),
        ])

        bid += 1

    return block_rm_rows, block_concat_rows


# -------------------------
# output layout
# -------------------------

def build_run_dir(model_dir: Path) -> Path:
    ckpt_tag = Path(CHECKPOINT_NAME).stem
    run_root = model_dir / "results_block"
    run_root.mkdir(parents=True, exist_ok=True)

    run_dir = run_root / f"somos_dialogue_{ckpt_tag}_t{int(TARGET_SECONDS)}_s{START_INDEX}_block{int(BLOCK_SECONDS)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "packets").mkdir(parents=True, exist_ok=True)
    return run_dir


def out_dir_for(run_dir: Path, mode: str, sys_a: str, sys_b: str) -> Path:
    tag = f"{sys_a}__{sys_b}"
    if mode == "forward":
        d = run_dir / f"dlg_{tag}"
    elif mode == "reverse":
        d = run_dir / f"rev_dlg_{tag}"
    elif mode == "random":
        d = run_dir / f"rand_dlg_{tag}"
    else:
        raise ValueError(mode)
    d.mkdir(parents=True, exist_ok=True)
    return d


def variant_subdir(base_out_dir: Path, variant: str) -> Path:
    d = base_out_dir / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_packet_csv(path: Path, packet: List[Tuple[str, float, str, float]], meta: Dict[str, object]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for k, v in meta.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["utt", "mos_target", "path", "dur_s"])
        for utt, mos_t, p, dur in packet:
            w.writerow([utt, mos_t, p, dur])


# -------------------------
# main per training
# -------------------------

def run_training(train_id: int) -> None:
    model_dir = Path(f"/net/vol/zigor/checkpoints/{train_id}")
    run_dir = build_run_dir(model_dir)

    print("\n" + "=" * 38)
    print(f"[TRAINING] {train_id}  MODEL_DIR={model_dir}")
    print(f"[OUT]      {run_dir}")
    print("=" * 38)

    items = parse_somos_list(SOMOS_TEST_LIST)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_gpu = SpeechQualityPredictor(
        storage_dir=str(model_dir),
        checkpoint_name=CHECKPOINT_NAME,
        return_numpy=True,
        device=device,
    )

    predictor_cpu = None
    if FALLBACK_TO_CPU and device == "cuda":
        predictor_cpu = SpeechQualityPredictor(
            storage_dir=str(model_dir),
            checkpoint_name=CHECKPOINT_NAME,
            return_numpy=True,
            device="cpu",
        )

    pairs = list(itertools.combinations(SYSTEM_IDS, 2))

    for sys_a, sys_b in pairs:
        packet = build_dialogue_packet(items, sys_a, sys_b)
        if not packet:
            print(f"[SKIP] pair {sys_a}__{sys_b} (no data)")
            continue

        meta = {
            "train_id": train_id,
            "sys_a": sys_a,
            "sys_b": sys_b,
            "target_seconds": TARGET_SECONDS,
            "block_seconds": BLOCK_SECONDS,
            "start_index": START_INDEX,
            "list_path": str(SOMOS_TEST_LIST),
            "wav_root": str(SOMOS_WAV_ROOT),
        }

        for mode in MODES_TO_RUN:
            base_out = out_dir_for(run_dir, mode, sys_a, sys_b)

            if SKIP_IF_DONE:
                done = True
                for v in VARIANTS:
                    vd = base_out / v
                    if not (vd / "global_mos_concat.csv").exists():
                        done = False
                        break
                    if not (vd / "global_mos_running_mean.csv").exists():
                        done = False
                        break
                    if not (vd / "block_mos_concat.csv").exists():
                        done = False
                        break
                    if not (vd / "block_mos_running_mean.csv").exists():
                        done = False
                        break
                if done:
                    print(f"[SKIP] {mode} {sys_a}__{sys_b} (already done)")
                    continue

            pkt_path = run_dir / "packets" / f"packet_dialogue_train{train_id}_{sys_a}__{sys_b}.csv"
            if not pkt_path.exists():
                write_packet_csv(pkt_path, [(u, m, str(p), d) for (u, m, p, d) in packet], meta)

            segs: List[Seg] = []
            for utt, mos_t, p, dur in packet:
                wav = load_wav(p)
                seed = _make_seed("somos_dialogue", str(train_id), sys_a, sys_b, utt)
                variants = make_variants(wav, SAMPLING_RATE, seed)
                segs.append(Seg(utt=utt, mos_t=float(mos_t), path=p, dur_s=float(dur), variants=variants))

            seed_tag = f"{train_id}|{sys_a}|{sys_b}|{mode}"
            segs = order_dialogue(segs, mode, seed_tag)

            for variant in VARIANTS:
                out_dir = variant_subdir(base_out, variant)
                p_gc = out_dir / "global_mos_concat.csv"
                p_gr = out_dir / "global_mos_running_mean.csv"
                p_bc = out_dir / "block_mos_concat.csv"
                p_br = out_dir / "block_mos_running_mean.csv"

                if SKIP_IF_DONE and p_gc.exists() and p_gr.exists() and p_bc.exists() and p_br.exists():
                    continue

                seg_pred_raw: List[float] = []
                pred_prefix: List[float] = []
                tgt_prefix: List[float] = []
                dur_prefix: List[float] = []
                elapsed = 0.0
                rm_rows = []

                for i, s in enumerate(segs, start=1):
                    ok, _, _, mos_pred, _ = safe_infer_global(
                        predictor_gpu, predictor_cpu, s.variants[variant],
                        f"dlg:{train_id}:{sys_a}__{sys_b}:{mode}:{variant}:seg{i}"
                    )
                    if not ok:
                        break

                    seg_pred_raw.append(float(mos_pred))

                    pred_prefix.append(float(mos_pred))
                    tgt_prefix.append(float(s.mos_t))
                    dur_prefix.append(float(s.dur_s))
                    elapsed += float(s.dur_s)

                    pred_rm = durw_mean(pred_prefix, dur_prefix)
                    tgt_rm = durw_mean(tgt_prefix, dur_prefix)
                    rm_rows.append([elapsed, float(pred_rm), float(tgt_rm)])

                if not rm_rows:
                    continue

                concat_wavs = []
                concat_secs = 0.0
                gc_rows = []
                tgt_c = []
                dur_c = []

                for i, s in enumerate(segs[:len(seg_pred_raw)], start=1):
                    concat_wavs.append(s.variants[variant])
                    concat_secs += float(s.dur_s)

                    tgt_c.append(float(s.mos_t))
                    dur_c.append(float(s.dur_s))

                    wav_cat = np.concatenate(concat_wavs).astype(np.float32)
                    ok, _, _, mos_pred, _ = safe_infer_global(
                        predictor_gpu, predictor_cpu, wav_cat,
                        f"dlg:{train_id}:{sys_a}__{sys_b}:{mode}:{variant}:concat{i}"
                    )
                    if not ok:
                        break

                    tgt_concat = durw_mean(tgt_c, dur_c)
                    gc_rows.append([concat_secs, float(mos_pred), float(tgt_concat)])

                if not gc_rows:
                    continue

                segs_ok = segs[:len(seg_pred_raw)]
                br_rows, bc_rows = compute_blocks(
                    segs=segs_ok,
                    seg_pred_raw=seg_pred_raw,
                    variant=variant,
                    block_seconds=BLOCK_SECONDS,
                    predictor_gpu=predictor_gpu,
                    predictor_cpu=predictor_cpu,
                    tag_prefix=f"dlg:{train_id}:{sys_a}__{sys_b}:{mode}",
                )

                with open(p_gc, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["seconds", "mos_pred_concat", "mos_target_concat_durw"])
                    for sec, mp, mt in gc_rows:
                        w.writerow([round(float(sec), 6), round(float(mp), 6), round(float(mt), 6)])

                with open(p_gr, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["elapsed_s", "mos_pred_rm_durw", "mos_target_rm_durw"])
                    for t, mp, mt in rm_rows:
                        w.writerow([round(float(t), 6), round(float(mp), 6), round(float(mt), 6)])

                with open(p_bc, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "block_start_s", "block_end_s", "block_mid_s",
                        "mos_pred_block_concat", "mos_target_block_durw",
                        "used_s", "n_pieces", "block_seconds"
                    ])
                    for r in bc_rows:
                        w.writerow(r)

                with open(p_br, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "block_start_s", "block_end_s", "block_mid_s",
                        "mos_pred_block_rm_durw", "mos_target_block_durw",
                        "used_s", "n_pieces", "block_seconds"
                    ])
                    for r in br_rows:
                        w.writerow(r)

                print(f"[OK] train{train_id} {sys_a}__{sys_b} {mode} {variant} -> {out_dir}")


def main() -> None:
    for tid in TRAIN_IDS:
        try:
            run_training(tid)
        except Exception as e:
            print(f"[ERROR] train{tid}: {e}")


if __name__ == "__main__":
    main()