from __future__ import annotations

import argparse
import csv
import time
import zlib
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import psutil
import torch

import audiofile

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio


DEFAULT_MODEL_DIR = "/net/vol/zigor/checkpoints/18"
DEFAULT_CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

DEFAULT_SOMOS_TEST_LIST = Path("/net/db/somos/training_files/split1/clean/test_mos_list.txt")
DEFAULT_SOMOS_WAV_ROOT = Path("/net/db/somos/audios")

SYSTEM_IDS = ["061", "057", "110", "124", "191"]

SCHEDULES = [
    ("one_sentence", 1),
    ("three_sentences", 3),
    ("five_sentences", 5),
]

PAUSE_MODES = [
    ("pause_zero", "zero"),
    ("pause_noise", "noise"),
]

AUDIO_LOADER = LoadAudio(
    audio_path_keys="audio_path.observation",
    target_sampling_rate=SAMPLING_RATE,
    resample=True,
)


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


def parse_somos_list(list_path: Path) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for i, ln in enumerate(list_path.read_text().splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        if i == 0 and ln.lower().startswith("utteranceid"):
            continue
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        utt = parts[0].strip()
        mos = float(parts[1])
        items.append((utt, mos))
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


def load_wav(path: Path) -> np.ndarray:
    ex = {"audio_path": {"observation": str(path)}}
    ex = AUDIO_LOADER(ex)
    wav = ex["audio"].astype(np.float32)
    if wav.ndim != 1:
        raise ValueError(f"expected 1D audio, got {wav.shape} for {path}")
    return wav


def wav_duration_s(path: Path) -> float:
    try:
        return float(audiofile.duration(str(path)))
    except Exception:
        y = load_wav(path)
        return float(len(y) / SAMPLING_RATE)


def durw_mean(values, durations) -> float:
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(durations, dtype=np.float64)
    w = np.maximum(d, 1e-12)
    return float(np.sum(v * w) / np.sum(w))


def safe_infer_global(predictor_gpu, predictor_cpu, wav: np.ndarray, tag: str, fallback_to_cpu: bool):
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

        if not fallback_to_cpu or predictor_cpu is None:
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


def resolve_somos_path(wav_root: Path, utt: str) -> Optional[Path]:
    p = wav_root / utt
    if p.exists():
        return p
    if not utt.endswith(".wav"):
        p2 = wav_root / f"{utt}.wav"
        if p2.exists():
            return p2
    return None


def collect_system_files(items: List[Tuple[str, float]], wav_root: Path, system_id: str):
    out = []
    sid = str(system_id).zfill(3)
    for utt, mos in items:
        if extract_system_id(utt) != sid:
            continue
        p = resolve_somos_path(wav_root, utt)
        if p is None:
            continue
        dur = wav_duration_s(p)
        out.append((p.name, float(mos), p, float(dur)))
    if not out:
        raise ValueError(f"no files found for system {sid}")
    return out


def build_mixed_packet(sys_a, sys_b, sysA_tag: str, sysB_tag: str, block: int,
                       target_seconds: float, start_index: int, max_files: int, allow_repeat: bool):
    a0 = start_index % len(sys_a)
    b0 = start_index % len(sys_b)

    a = sys_a[a0:] + sys_a[:a0]
    b = sys_b[b0:] + sys_b[:b0]

    ia = 0
    ib = 0
    total_s = 0.0
    packet = []

    while total_s < target_seconds and len(packet) < max_files:
        for _ in range(block):
            if total_s >= target_seconds or len(packet) >= max_files:
                break
            if ia >= len(a):
                if not allow_repeat:
                    break
                ia = 0
            fn, mos, p, dur = a[ia]
            ia += 1
            packet.append((sysA_tag, fn, float(mos), str(p), float(dur)))
            total_s += float(dur)

        if total_s >= target_seconds or len(packet) >= max_files:
            break

        for _ in range(block):
            if total_s >= target_seconds or len(packet) >= max_files:
                break
            if ib >= len(b):
                if not allow_repeat:
                    break
                ib = 0
            fn, mos, p, dur = b[ib]
            ib += 1
            packet.append((sysB_tag, fn, float(mos), str(p), float(dur)))
            total_s += float(dur)

        if not allow_repeat:
            if ia >= len(a) or ib >= len(b):
                break

    return packet, float(total_s), len(sys_a), len(sys_b)


def save_packet_csv(path: Path, sysA: str, sysB: str, schedule_name: str, block: int, packet,
                    total_s: float, nA: int, nB: int,
                    target_seconds: float, start_index: int, allow_repeat: bool, list_path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", f"{sysA}_{sysB}"])
        w.writerow(["schedule", schedule_name])
        w.writerow(["block_size", block])
        w.writerow(["list_path", str(list_path)])
        w.writerow(["target_seconds", float(target_seconds)])
        w.writerow(["start_index", int(start_index)])
        w.writerow(["allow_repeat", int(bool(allow_repeat))])
        w.writerow(["sysA_count", nA])
        w.writerow(["sysB_count", nB])
        w.writerow(["packet_len", len(packet)])
        w.writerow(["packet_total_s_orig", total_s])
        w.writerow([])
        w.writerow(["idx", "src_system", "fn", "mos_target", "path", "dur_s"])
        for i, (src, fn, mos_t, p, dur) in enumerate(packet):
            w.writerow([i, src, fn, mos_t, p, dur])


def load_packet_csv(path: Path):
    rows = path.read_text().splitlines()
    header = "idx,src_system,fn,mos_target,path,dur_s"
    i0 = None
    for i, r in enumerate(rows):
        if r.strip() == header:
            i0 = i + 1
            break
    if i0 is None:
        raise RuntimeError(f"invalid packet file: {path}")

    packet = []
    for r in rows[i0:]:
        r = r.strip()
        if not r:
            continue
        idx, src, fn, mos_t, p, dur = r.split(",", 5)
        packet.append((src, fn, float(mos_t), Path(p), float(dur)))
    return packet


def make_pause(sr: int, seconds: float, kind: str, seed: int, noise_std: float):
    n = int(round(seconds * sr))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if kind == "zero":
        return np.zeros(n, dtype=np.float32)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, noise_std, size=n).astype(np.float32)
        return np.clip(x, -1.0, 1.0)
    raise ValueError(kind)


def run_pair_schedule(predictor_gpu, predictor_cpu, run_root: Path,
                      sysA_tag: str, sysB_tag: str,
                      schedule_name: str, block: int,
                      sysA_list, sysB_list,
                      target_seconds: float, start_index: int, max_files: int, allow_repeat: bool,
                      pause_seconds: float, noise_std: float,
                      skip_if_done: bool, fallback_to_cpu: bool,
                      list_path: Path):
    pair_name = f"{sysA_tag}_{sysB_tag}"
    pair_dir = run_root / schedule_name / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    pkt_path = pair_dir / "packet.csv"
    if not pkt_path.exists():
        packet_raw, total_s, nA, nB = build_mixed_packet(
            sysA_list, sysB_list, sysA_tag, sysB_tag, block,
            target_seconds=target_seconds,
            start_index=start_index,
            max_files=max_files,
            allow_repeat=allow_repeat,
        )
        save_packet_csv(
            pkt_path, sysA_tag, sysB_tag, schedule_name, block,
            packet_raw, total_s, nA, nB,
            target_seconds=target_seconds,
            start_index=start_index,
            allow_repeat=allow_repeat,
            list_path=list_path,
        )
        print(f"[packet] created {pkt_path}  total_s={total_s:.1f}  len={len(packet_raw)}")
    else:
        print(f"[packet] using   {pkt_path}")

    packet = load_packet_csv(pkt_path)
    if not packet:
        print(f"[skip] empty packet for {pair_name} / {schedule_name}")
        return

    for pause_dirname, pause_kind in PAUSE_MODES:
        out_dir = pair_dir / pause_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        p_rm = out_dir / "global_mos_running_mean.csv"
        p_concat = out_dir / "global_mos_concat.csv"
        p_timeline = out_dir / "timeline.csv"

        if skip_if_done and p_rm.exists() and p_concat.exists() and p_timeline.exists():
            print(f"[skip] {schedule_name} | {pair_name} | {pause_dirname}")
            continue

        print(f"\n=== {schedule_name} | {pair_name} | {pause_dirname} ===")
        print(f"out: {out_dir}")

        sr = SAMPLING_RATE

        cur = 0
        with open(p_timeline, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "src_system", "fn", "start_s", "end_s", "pause_after_s"])
            for i, (src, fn, mos_t, path, dur) in enumerate(packet):
                start = cur / sr
                end = (cur + int(round(dur * sr))) / sr
                pause_after = pause_seconds if i < len(packet) - 1 else 0.0
                w.writerow([i, src, fn, round(start, 6), round(end, 6), round(pause_after, 6)])
                cur += int(round(dur * sr))
                if i < len(packet) - 1:
                    cur += int(round(pause_seconds * sr))

        elapsed_orig = 0.0
        pred_list, tgt_list, dur_list = [], [], []

        with open(p_rm, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "elapsed_orig_s", "mos_pred_rm_durw", "mos_target_rm_durw",
                "idx", "src_system", "fn", "mos_pred", "mos_target", "dur_s",
                "runtime_s", "mem_mb", "device"
            ])

            for i, (src, fn, mos_t, path, dur) in enumerate(packet):
                y = load_wav(path)
                ok, rt, mem, mos_pred, where = safe_infer_global(
                    predictor_gpu, predictor_cpu, y,
                    f"rm:somos:{schedule_name}:{pair_name}:{pause_dirname}:{i}",
                    fallback_to_cpu=fallback_to_cpu,
                )
                if not ok:
                    print(f"[rm] stop at idx={i} ({fn})")
                    break

                elapsed_orig += float(dur)
                pred_list.append(float(mos_pred))
                tgt_list.append(float(mos_t))
                dur_list.append(float(dur))

                pred_rm = durw_mean(pred_list, dur_list)
                tgt_rm = durw_mean(tgt_list, dur_list)

                w.writerow([
                    round(elapsed_orig, 6),
                    round(pred_rm, 6),
                    round(tgt_rm, 6),
                    i,
                    src,
                    fn,
                    round(float(mos_pred), 6),
                    round(float(mos_t), 6),
                    round(float(dur), 6),
                    None if rt is None else round(float(rt), 6),
                    None if mem is None else round(float(mem), 6),
                    where,
                ])

        chunks = []
        secs_orig = 0.0
        secs_with_pause = 0.0
        tgt_prefix, dur_prefix = [], []

        with open(p_concat, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "seconds_with_pause", "seconds_orig", "mos_pred_concat", "mos_target_concat_durw",
                "idx", "src_system", "fn", "runtime_s", "mem_mb", "device"
            ])

            for i, (src, fn, mos_t, path, dur) in enumerate(packet):
                y = load_wav(path)
                chunks.append(y)

                secs_orig += float(dur)
                secs_with_pause += float(dur)

                tgt_prefix.append(float(mos_t))
                dur_prefix.append(float(dur))
                tgt_concat = durw_mean(tgt_prefix, dur_prefix)

                if i < len(packet) - 1:
                    seed = zlib.adler32(f"somos|{schedule_name}|{pair_name}|{pause_kind}|{i}".encode("utf-8")) & 0xFFFFFFFF
                    p = make_pause(sr, pause_seconds, pause_kind, seed, noise_std=noise_std)
                    chunks.append(p)
                    secs_with_pause += float(pause_seconds)

                y_cat = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)

                ok, rt, mem, mos_pred, where = safe_infer_global(
                    predictor_gpu, predictor_cpu, y_cat,
                    f"cat:somos:{schedule_name}:{pair_name}:{pause_dirname}:{i}",
                    fallback_to_cpu=fallback_to_cpu,
                )
                if not ok:
                    print(f"[concat] stop at idx={i} ({fn})")
                    break

                w.writerow([
                    round(secs_with_pause, 6),
                    round(secs_orig, 6),
                    round(float(mos_pred), 6),
                    round(float(tgt_concat), 6),
                    i,
                    src,
                    fn,
                    None if rt is None else round(float(rt), 6),
                    None if mem is None else round(float(mem), 6),
                    where,
                ])

        print(f"done: {p_rm.name}, {p_concat.name}, {p_timeline.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_NAME)

    ap.add_argument("--somos_list", type=str, default=str(DEFAULT_SOMOS_TEST_LIST))
    ap.add_argument("--somos_root", type=str, default=str(DEFAULT_SOMOS_WAV_ROOT))

    ap.add_argument("--target_seconds", type=float, default=180.0)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--max_files", type=int, default=5000)
    ap.add_argument("--allow_repeat", type=int, default=1)

    ap.add_argument("--pause_seconds", type=float, default=5.0)
    ap.add_argument("--noise_std", type=float, default=0.003)

    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--fallback_cpu", type=int, default=0)
    args = ap.parse_args()

    model_dir = args.model_dir
    ckpt = args.ckpt

    somos_list_path = Path(args.somos_list)
    somos_wav_root = Path(args.somos_root)

    target_seconds = float(args.target_seconds)
    start_index = int(args.start_index)
    max_files = int(args.max_files)
    allow_repeat = bool(int(args.allow_repeat))

    pause_seconds = float(args.pause_seconds)
    noise_std = float(args.noise_std)

    skip_if_done = bool(int(args.skip))
    fallback_to_cpu = bool(int(args.fallback_cpu))

    run_root = Path(model_dir) / "results" / "somos_dialogue"
    run_root.mkdir(parents=True, exist_ok=True)

    print("model_dir:", model_dir)
    print("ckpt:", ckpt)
    print("run_root:", run_root)
    print("somos_list:", somos_list_path)
    print("somos_root:", somos_wav_root)
    print("target_seconds:", target_seconds)
    print("start_index:", start_index)
    print("max_files:", max_files)
    print("allow_repeat:", allow_repeat)
    print("pause_seconds:", pause_seconds)
    print("noise_std:", noise_std)
    print("skip_if_done:", skip_if_done)
    print("fallback_to_cpu:", fallback_to_cpu)
    print("schedules:", SCHEDULES)
    print("pause_modes:", PAUSE_MODES)
    print("systems:", SYSTEM_IDS)
    print()

    items = parse_somos_list(somos_list_path)

    sys_map = {}
    for sid in SYSTEM_IDS:
        sid3 = str(sid).zfill(3)
        tag = f"sys{sid3}"
        print(f"indexing {tag} ...")
        sys_map[tag] = collect_system_files(items, somos_wav_root, sid3)
        print(f"  files: {len(sys_map[tag])}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_gpu = SpeechQualityPredictor(
        storage_dir=model_dir,
        checkpoint_name=ckpt,
        return_numpy=True,
        device=device,
    )

    predictor_cpu = None
    if fallback_to_cpu and device == "cuda":
        predictor_cpu = SpeechQualityPredictor(
            storage_dir=model_dir,
            checkpoint_name=ckpt,
            return_numpy=True,
            device="cpu",
        )

    tags = list(sys_map.keys())
    pairs = list(combinations(tags, 2))
    print(f"\nall pairs: {len(pairs)}")

    for sysA_tag, sysB_tag in pairs:
        for schedule_name, block in SCHEDULES:
            try:
                run_pair_schedule(
                    predictor_gpu,
                    predictor_cpu,
                    run_root,
                    sysA_tag,
                    sysB_tag,
                    schedule_name,
                    block,
                    sys_map[sysA_tag],
                    sys_map[sysB_tag],
                    target_seconds=target_seconds,
                    start_index=start_index,
                    max_files=max_files,
                    allow_repeat=allow_repeat,
                    pause_seconds=pause_seconds,
                    noise_std=noise_std,
                    skip_if_done=skip_if_done,
                    fallback_to_cpu=fallback_to_cpu,
                    list_path=somos_list_path,
                )
            except Exception as e:
                print(f"\n[error] {sysA_tag}_{sysB_tag} / {schedule_name}: {e}\n")


if __name__ == "__main__":
    main()
