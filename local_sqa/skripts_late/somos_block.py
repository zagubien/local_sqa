from __future__ import annotations

import csv
import time
import zlib
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

TRAIN_RUN_IDS = [18, 19, 23, 24, 26, 27]
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

SOMOS_TEST_LIST = Path("/net/db/somos/training_files/split1/clean/test_mos_list.txt")
SOMOS_WAV_ROOT = Path("/net/db/somos/audios")

SYSTEM_IDS = ["061", "057", "110", "124", "191"]
MODES_TO_RUN = ["forward", "reverse", "random"]

TARGET_SECONDS = 180.0
BLOCK_SECONDS = 5.0

START_INDEX = 0
MAX_FILES = 5000

RANDOM_SEED = 1234
FALLBACK_TO_CPU = False
ALLOW_REPEAT = True
SKIP_IF_DONE = True

# pause configs
INSERT_PAUSE_SECONDS = 5.0
NOISE_STD = 0.003

PAD_MS = 40
MERGE_GAP_MS = 120
MIN_SPEECH_MS = 80
MIN_PROC_MS_FALLBACK = 250

VARIANTS = [
    "original",
    "no_pause",
    "long_pause",
    "long_pause_noise",
]

# csv names
CSV_GLOBAL_CONCAT = "global_mos_concat.csv"
CSV_GLOBAL_RM = "global_mos_running_mean.csv"
CSV_BLOCK_CONCAT = "block_mos_concat.csv"
CSV_BLOCK_RM = "block_mos_running_mean.csv"


# -------------------------
# utils
# -------------------------

def cpu_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def gpu_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


AUDIO_LOADER = LoadAudio(
    audio_path_keys="audio_path.observation",
    target_sampling_rate=SAMPLING_RATE,
    resample=True,
)


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


def load_wav(path: Path) -> np.ndarray:
    ex = {"audio_path": {"observation": str(path)}}
    ex = AUDIO_LOADER(ex)
    wav = ex["audio"].astype(np.float32)
    if wav.ndim != 1:
        raise ValueError(f"expected 1D audio, got {wav.shape} for {path}")
    return wav


def durw_mean(values, durations) -> float:
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(durations, dtype=np.float64)
    w = np.maximum(d, 1e-12)
    return float(np.sum(v * w) / np.sum(w))


def safe_infer_global(predictor_gpu, predictor_cpu, wav: np.ndarray, tag: str):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"expected 1D waveform, got {wav.shape}")

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


def build_paths(model_dir: Path) -> Tuple[str, Path, Path]:
    ckpt_tag = Path(CHECKPOINT_NAME).stem

    run_root = model_dir / "results_block"
    run_root.mkdir(parents=True, exist_ok=True)

    run_dir = run_root / f"somos_{ckpt_tag}_t{int(TARGET_SECONDS)}_s{START_INDEX}_block{int(BLOCK_SECONDS)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    packet_dir = run_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)

    return ckpt_tag, run_dir, packet_dir


def packet_path(packet_dir: Path, ckpt_tag: str, system_id: str) -> Path:
    return packet_dir / f"packet_somos_{ckpt_tag}_sys{system_id}_t{int(TARGET_SECONDS)}_s{START_INDEX}.csv"


def build_packet(items: List[Tuple[str, float]], system_id: str, target_seconds: float):
    filtered = []
    for utt, mos in items:
        sid = extract_system_id(utt)
        if sid == system_id:
            filtered.append((utt, mos))

    filtered = filtered[START_INDEX:START_INDEX + MAX_FILES]

    packet = []
    total_s = 0.0

    for utt, mos in filtered:
        p = utt_to_path(utt)
        if not p.exists():
            continue

        try:
            info = audiofile.info(str(p))
            dur = float(info.duration)
        except Exception:
            y = load_wav(p)
            dur = float(len(y) / SAMPLING_RATE)

        packet.append((utt, float(mos), str(p), float(dur)))
        total_s += float(dur)

        if total_s >= target_seconds:
            break

        if not ALLOW_REPEAT:
            break

    return packet, float(total_s), len(filtered)


def save_packet(p: Path, system_id: str, packet, total_s: float, sys_count: int):
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system_id", system_id])
        w.writerow(["list_path", str(SOMOS_TEST_LIST)])
        w.writerow(["wav_root", str(SOMOS_WAV_ROOT)])
        w.writerow(["target_seconds", TARGET_SECONDS])
        w.writerow(["start_index", START_INDEX])
        w.writerow(["sys_count", sys_count])
        w.writerow(["packet_len", len(packet)])
        w.writerow(["packet_total_s", total_s])
        w.writerow([])
        w.writerow(["utt", "mos_target", "path", "dur_s"])
        for utt, mos_t, path_str, dur in packet:
            w.writerow([utt, mos_t, path_str, dur])


def load_packet(p: Path):
    rows = p.read_text().splitlines()
    header = "utt,mos_target,path,dur_s"
    i0 = None
    for i, r in enumerate(rows):
        if r.strip() == header:
            i0 = i + 1
            break
    if i0 is None:
        raise RuntimeError(f"invalid packet file: {p}")

    packet = []
    for r in rows[i0:]:
        r = r.strip()
        if not r:
            continue
        utt, mos_t, path_str, dur = r.split(",", 3)
        packet.append((utt, float(mos_t), Path(path_str), float(dur)))
    return packet


def order_packet(packet, mode: str, system_id: str):
    if mode == "forward":
        return packet
    if mode == "reverse":
        return list(reversed(packet))
    if mode == "random":
        h = zlib.adler32(system_id.encode("utf-8")) & 0xFFFFFFFF
        seed = (RANDOM_SEED ^ h) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(packet)).tolist()
        return [packet[i] for i in idx]
    raise ValueError(f"unknown mode: {mode}")


def out_dir_for(run_dir: Path, mode: str, system_id: str) -> Path:
    if mode == "forward":
        return run_dir / f"sys{system_id}"
    if mode == "reverse":
        return run_dir / f"rev_sys{system_id}"
    if mode == "random":
        return run_dir / f"rand_sys{system_id}"
    raise ValueError(mode)


def variant_subdir(base_out_dir: Path, variant: str) -> Path:
    d = base_out_dir / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


# -------------------------
# VAD + variants
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


def _make_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    return zlib.adler32(s) & 0xFFFFFFFF


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
        return np.zeros(0, dtype=np.float32), []

    rng = np.random.default_rng(seed)
    pause = _make_pause(sr, pause_s, pause_kind, rng)

    chunks = []
    for k, (s, e) in enumerate(intervals):
        chunks.append(y[s:e])
        if k < len(intervals) - 1 and len(pause) > 0:
            chunks.append(pause)

    y_proc = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
    return y_proc, []


def make_variants(wav: np.ndarray, sr: int, seed: int):
    out = {"original": wav}

    intervals = vad_intervals_rvad(wav, sr)
    min_proc = int((MIN_PROC_MS_FALLBACK / 1000.0) * sr)

    if not intervals:
        out["no_pause"] = wav
        out["long_pause"] = wav
        out["long_pause_noise"] = wav
        return out

    y_cut, _ = _build_processed(wav, sr, intervals, pause_s=0.0, pause_kind="zero", seed=seed ^ 0xA1B2C3D4)
    y_zero, _ = _build_processed(wav, sr, intervals, pause_s=INSERT_PAUSE_SECONDS, pause_kind="zero", seed=seed ^ 0x11223344)
    y_noise, _ = _build_processed(wav, sr, intervals, pause_s=INSERT_PAUSE_SECONDS, pause_kind="noise", seed=seed ^ 0x55667788)

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
# block logic
# -------------------------

def assign_blocks(durations: List[float], block_s: float) -> List[int]:
    ids = []
    t = 0.0
    for d in durations:
        bid = int(t // block_s)
        ids.append(bid)
        t += float(d)
    return ids


def block_bounds(bid: int, block_s: float) -> Tuple[float, float, float]:
    s = bid * block_s
    e = (bid + 1) * block_s
    mid = 0.5 * (s + e)
    return s, e, mid


# -------------------------
# main run
# -------------------------

def run_one(items, predictor_gpu, predictor_cpu, ckpt_tag: str, run_dir: Path, packet_dir: Path, system_id: str, mode: str):
    base_out = out_dir_for(run_dir, mode, system_id)
    base_out.mkdir(parents=True, exist_ok=True)

    if SKIP_IF_DONE:
        all_done = True
        for v in VARIANTS:
            vd = base_out / v
            if not (vd / CSV_GLOBAL_CONCAT).exists():
                all_done = False
                break
            if not (vd / CSV_GLOBAL_RM).exists():
                all_done = False
                break
            if not (vd / CSV_BLOCK_CONCAT).exists():
                all_done = False
                break
            if not (vd / CSV_BLOCK_RM).exists():
                all_done = False
                break
        if all_done:
            print(f"[SKIP] {mode} | {system_id} (all variants already done)")
            return

    p_pkt = packet_path(packet_dir, ckpt_tag, system_id)
    if not p_pkt.exists():
        pkt_raw, total_s, sys_count = build_packet(items, system_id, TARGET_SECONDS)
        save_packet(p_pkt, system_id, pkt_raw, total_s, sys_count)
        print(f"[sys{system_id}] created packet: {p_pkt.name}  total_s={total_s:.1f}  base_n={sys_count}")
    else:
        print(f"[sys{system_id}] using packet:   {p_pkt.name}")

    packet = load_packet(p_pkt)
    packet = order_packet(packet, mode, system_id)

    print(f"\n=== {mode.upper()} | sys{system_id} ===")
    print(f"packet_len={len(packet)}  first={packet[0][0]}  last={packet[-1][0]}")
    print(f"out={base_out}")

    segs = []
    for (utt, mos_t, p, dur) in packet:
        wav = load_wav(p)
        seed = _make_seed("somos", system_id, mode, utt)
        variants = make_variants(wav, SAMPLING_RATE, seed)

        segs.append({
            "utt": utt,
            "mos_t": float(mos_t),
            "dur_orig": float(dur),
            "path": str(p),
            "variants": variants,
        })

    durs = [float(s["dur_orig"]) for s in segs]
    block_ids = assign_blocks(durs, BLOCK_SECONDS)
    unique_blocks = sorted(set(block_ids))

    for variant in VARIANTS:
        out_dir = variant_subdir(base_out, variant)
        p_global_concat = out_dir / CSV_GLOBAL_CONCAT
        p_global_rm = out_dir / CSV_GLOBAL_RM
        p_block_concat = out_dir / CSV_BLOCK_CONCAT
        p_block_rm = out_dir / CSV_BLOCK_RM

        if SKIP_IF_DONE and p_global_concat.exists() and p_global_rm.exists() and p_block_concat.exists() and p_block_rm.exists():
            print(f"[SKIP] {mode} | sys{system_id} | {variant}")
            continue

        print(f"\n--- variant: {variant} ---")

        elapsed = 0.0
        pred_list, tgt_list, dur_list = [], [], []
        global_rm_rows = []

        seg_pred = []
        ok_up_to = 0

        for j, s in enumerate(segs, start=1):
            wav_v = s["variants"][variant]
            ok, _, _, mos_pred, _ = safe_infer_global(
                predictor_gpu, predictor_cpu, wav_v, f"somos:{system_id}:{mode}:{variant}:seg{j}"
            )
            if not ok:
                print(f"stopping at segment {j} (segment inference) for variant={variant}")
                break

            ok_up_to = j
            seg_pred.append(float(mos_pred))

            pred_list.append(float(mos_pred))
            tgt_list.append(float(s["mos_t"]))
            dur_list.append(float(s["dur_orig"]))
            elapsed += float(s["dur_orig"])

            pred_rm = durw_mean(pred_list, dur_list)
            tgt_rm = durw_mean(tgt_list, dur_list)
            global_rm_rows.append([elapsed, pred_rm, tgt_rm])

        if len(global_rm_rows) == 0:
            print(f"no segments inferred -> skip variant={variant}")
            continue

        concat_wavs = []
        concat_secs = 0.0
        global_concat_rows = []
        tgt_prefix, dur_prefix = [], []

        for n, s in enumerate(segs[:ok_up_to], start=1):
            wav_v = s["variants"][variant]
            concat_wavs.append(wav_v)
            concat_secs += float(s["dur_orig"])

            tgt_prefix.append(float(s["mos_t"]))
            dur_prefix.append(float(s["dur_orig"]))

            wav_cat = np.concatenate(concat_wavs).astype(np.float32) if concat_wavs else np.zeros(0, dtype=np.float32)

            ok, _, _, mos_pred, _ = safe_infer_global(
                predictor_gpu, predictor_cpu, wav_cat, f"somos:{system_id}:{mode}:{variant}:concat{n}"
            )
            if not ok:
                print(f"stopping at n={n} (global concat) for variant={variant}")
                break

            tgt_concat = durw_mean(tgt_prefix, dur_prefix)
            global_concat_rows.append([concat_secs, float(mos_pred), float(tgt_concat)])

        if len(global_concat_rows) == 0:
            print(f"global concat failed entirely -> skip writing for variant={variant}")
            continue

        with open(p_global_concat, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seconds", "mos_pred_concat", "mos_target_concat_durw"])
            for sec, mp, mt in global_concat_rows:
                w.writerow([round(float(sec), 6), round(float(mp), 6), round(float(mt), 6)])

        with open(p_global_rm, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["elapsed_s", "mos_pred_rm_durw", "mos_target_rm_durw"])
            for t, mp, mt in global_rm_rows:
                w.writerow([round(float(t), 6), round(float(mp), 6), round(float(mt), 6)])

        segs_ok = segs[:ok_up_to]
        durs_ok = durs[:ok_up_to]
        bids_ok = block_ids[:ok_up_to]
        pred_ok = seg_pred[:ok_up_to]

        block_rm_rows = []
        block_concat_rows = []

        block_to_idx: Dict[int, List[int]] = {}
        for i_seg, bid in enumerate(bids_ok):
            block_to_idx.setdefault(int(bid), []).append(i_seg)

        for bid in unique_blocks:
            idxs = block_to_idx.get(int(bid), [])
            if not idxs:
                continue

            s_block, e_block, mid = block_bounds(int(bid), BLOCK_SECONDS)

            v_pred = [pred_ok[i] for i in idxs]
            v_tgt = [float(segs_ok[i]["mos_t"]) for i in idxs]
            v_dur = [float(durs_ok[i]) for i in idxs]
            pred_blk_rm = durw_mean(v_pred, v_dur)
            tgt_blk = durw_mean(v_tgt, v_dur)

            block_rm_rows.append([s_block, e_block, mid, float(pred_blk_rm), float(tgt_blk)])

            wavs = [segs_ok[i]["variants"][variant] for i in idxs]
            wav_cat = np.concatenate(wavs).astype(np.float32) if wavs else np.zeros(0, dtype=np.float32)
            ok, _, _, mos_pred_blk, _ = safe_infer_global(
                predictor_gpu, predictor_cpu, wav_cat, f"somos:{system_id}:{mode}:{variant}:block{bid}"
            )
            if not ok:
                print(f"[WARN] block concat failed bid={bid} -> skip this block")
                continue

            block_concat_rows.append([s_block, e_block, mid, float(mos_pred_blk), float(tgt_blk)])

        with open(p_block_concat, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["block_start_s", "block_end_s", "block_mid_s", "mos_pred_block_concat", "mos_target_block_durw"])
            for a, b, m, mp, mt in block_concat_rows:
                w.writerow([round(float(a), 6), round(float(b), 6), round(float(m), 6), round(float(mp), 6), round(float(mt), 6)])

        with open(p_block_rm, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["block_start_s", "block_end_s", "block_mid_s", "mos_pred_block_rm_durw", "mos_target_block_durw"])
            for a, b, m, mp, mt in block_rm_rows:
                w.writerow([round(float(a), 6), round(float(b), 6), round(float(m), 6), round(float(mp), 6), round(float(mt), 6)])

        print("done:")
        print(p_global_concat)
        print(p_global_rm)
        print(p_block_concat)
        print(p_block_rm)


def run_one_training(model_dir: Path, items) -> None:
    ckpt_tag, run_dir, packet_dir = build_paths(model_dir)

    print("\n======================================")
    print(f"[TRAINING] MODEL_DIR={model_dir}")
    print(f"[OUT]      RUN_DIR={run_dir}")
    print("======================================\n")

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

    for system_id in SYSTEM_IDS:
        for mode in MODES_TO_RUN:
            try:
                run_one(items, predictor_gpu, predictor_cpu, ckpt_tag, run_dir, packet_dir, system_id, mode)
            except Exception as e:
                print(f"\n[ERROR] MODEL_DIR={model_dir} sys{system_id} / {mode}: {e}\n")


def run_all_trainings() -> None:
    items = parse_somos_list(SOMOS_TEST_LIST)

    for rid in TRAIN_RUN_IDS:
        md = Path(f"/net/vol/zigor/checkpoints/{rid}")
        if not md.exists():
            print(f"[SKIP] missing: {md}")
            continue
        run_one_training(md, items)


if __name__ == "__main__":
    run_all_trainings()