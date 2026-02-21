#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import socket
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import psutil
import torch

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio


MODEL_DIR_DEFAULT = "/net/vol/zigor/checkpoints/26"
CHECKPOINT_NAME_DEFAULT = "ckpt_best_SRCC.pth"

BVCC_TEST_LIST_DEFAULT = "/net/db/BVCC/main/DATA/sets/test_mos_list.txt"
BVCC_WAV_ROOT_DEFAULT = "/net/db/BVCC/main/DATA/wav"

SOMOS_TEST_LIST_DEFAULT = "/net/db/somos/training_files/split1/clean/test_mos_list.txt"
SOMOS_WAV_ROOT_DEFAULT = "/net/db/somos/audios"

BVCC_SYSTEMS_DEFAULT = ["sys78aec", "sys83aed", "sys91caa", "sys6c11c", "sys8f532", "sysd81da"]
SOMOS_SYSTEMS_DEFAULT = ["061", "057", "110", "124", "191"]

FALLBACK_TO_CPU = False

AUDIO_LOADER = LoadAudio(
    audio_path_keys="audio_path.observation",
    target_sampling_rate=SAMPLING_RATE,
    resample=True,
)


@dataclass
class PacketItem:
    key: str
    mos: float
    path: Path
    dur_s: float


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


def ensure_file(p: Path, name: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"missing {name}: {p}")


def parse_list_generic(list_path: Path) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for ln in list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s:
            continue

        if "," in s:
            a, b = s.split(",", 1)
            key = a.strip()
            mos_s = b.strip()
        else:
            parts = s.split()
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            mos_s = parts[-1].strip()

        try:
            mos = float(mos_s)
        except Exception:
            continue

        out.append((key, mos))
    return out


def wav_duration_seconds(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        return 0.0 if sr <= 0 else float(n) / float(sr)


def load_wav_16k(path: Path) -> np.ndarray:
    ex = {"audio_path": {"observation": str(path)}}
    ex = AUDIO_LOADER(ex)
    wav = ex["audio"].astype(np.float32)
    if wav.ndim != 1:
        raise ValueError(f"expected 1D audio, got {wav.shape} for {path}")
    return wav


def durw_mean(values: List[float], durations: List[float]) -> float:
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


def resolve_audio_path(key: str, wav_root: Path) -> Path:
    p = Path(key)

    if p.is_absolute() and p.exists():
        return p

    p2 = wav_root / key
    if p2.exists():
        return p2

    if p.suffix == "":
        p2w = wav_root / f"{key}.wav"
        if p2w.exists():
            return p2w

    bn = p.name if p.suffix else f"{p.name}.wav"
    hit = next(wav_root.rglob(bn), None)
    if hit is not None and hit.exists():
        return hit

    raise FileNotFoundError(f"audio not found for key={key} (root={wav_root})")


def bvcc_matcher(key: str, system_id: str) -> bool:
    return key.startswith(f"{system_id}-")


def somos_matcher(key: str, system_id: str) -> bool:
    stem = Path(key).stem
    return stem.endswith(f"_{system_id}")


def build_packet_forward(
    items: List[Tuple[str, float]],
    system_id: str,
    wav_root: Path,
    target_seconds: float,
    start_index: int,
    max_files: int,
    allow_repeat: bool,
    matcher,
) -> List[PacketItem]:
    filtered = [(k, mos) for (k, mos) in items if matcher(k, system_id)]
    if not filtered:
        raise RuntimeError(f"no items for system_id={system_id}")

    start = start_index % len(filtered)
    base = filtered[start:] + filtered[:start]

    packet: List[PacketItem] = []
    total = 0.0

    while total < target_seconds and len(packet) < max_files:
        for key, mos in base:
            if total >= target_seconds or len(packet) >= max_files:
                break
            ap = resolve_audio_path(key, wav_root)
            dur = wav_duration_seconds(ap)
            packet.append(PacketItem(key=key, mos=float(mos), path=ap, dur_s=float(dur)))
            total += float(dur)
        if not allow_repeat:
            break

    return packet


def auto_find_tts_root(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg)

    here = Path(__file__).resolve()
    cands = [
        Path.cwd() / "data" / "tts",
        here.parent / "data" / "tts",
        here.parent.parent / "data" / "tts",
        here.parent.parent.parent / "data" / "tts",
    ]
    for c in cands:
        if c.exists():
            return c
    return Path.cwd() / "data" / "tts"


def find_tts_wav(tts_dir: Path, orig_key: str) -> Path:
    stem = Path(orig_key).stem
    pats = [
        f"{stem}__tts*.wav",
        f"{stem}*tts*.wav",
        f"{stem}*.wav",
    ]

    hits: List[Path] = []
    for pat in pats:
        hits.extend(sorted(tts_dir.glob(pat)))

    if not hits:
        for p in sorted(tts_dir.glob("*.wav")):
            if stem in p.stem:
                hits.append(p)

    if not hits:
        raise FileNotFoundError(f"no tts wav found for {orig_key} in {tts_dir}")

    return hits[0]


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def pick_index(n: int, pick: str, needle_index: Optional[int]) -> int:
    if needle_index is not None:
        if needle_index < 0 or needle_index >= n:
            raise ValueError(f"needle_index out of range: {needle_index} (packet_len={n})")
        return int(needle_index)

    if pick == "middle":
        return n // 2
    if pick == "first":
        return 0
    if pick == "last":
        return n - 1
    raise ValueError(pick)


def run_one_system(
    *,
    dataset: str,
    system_id: str,
    items: List[Tuple[str, float]],
    wav_root: Path,
    matcher,
    predictor_gpu,
    predictor_cpu,
    tts_root: Path,
    out_dir: Path,
    target_seconds: float,
    start_index: int,
    max_files: int,
    allow_repeat: bool,
    pick: str,
    needle_index: Optional[int],
    skip_missing_tts: bool,
) -> None:
    packet = build_packet_forward(
        items=items,
        system_id=system_id,
        wav_root=wav_root,
        target_seconds=target_seconds,
        start_index=start_index,
        max_files=max_files,
        allow_repeat=allow_repeat,
        matcher=matcher,
    )

    ni = pick_index(len(packet), pick, needle_index)
    needle = packet[ni]

    tts_dir = tts_root / dataset / system_id / "forward"
    if not tts_dir.exists():
        msg = f"[SKIP] {dataset}:{system_id} missing tts dir: {tts_dir}"
        if skip_missing_tts:
            print(msg)
            return
        raise FileNotFoundError(msg)

    try:
        tts_wav = find_tts_wav(tts_dir, needle.key)
    except FileNotFoundError as e:
        if skip_missing_tts:
            print(f"[SKIP] {dataset}:{system_id} {e}")
            return
        raise

    out_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "dataset": dataset,
        "system_id": system_id,
        "mode": "forward",
        "variant": "original_only",
        "target_seconds": float(target_seconds),
        "start_index": int(start_index),
        "packet_len": int(len(packet)),
        "packet_total_s": float(sum(x.dur_s for x in packet)),
        "needle_pick": pick,
        "needle_index": int(ni),
        "needle_key": needle.key,
        "needle_original_path": str(needle.path),
        "needle_tts_path": str(tts_wav),
        "tts_dir": str(tts_dir),
        "host": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "needle_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    seg_wavs: List[np.ndarray] = []
    seg_dur_orig: List[float] = []
    seg_tgt: List[float] = []
    seg_keys: List[str] = []
    seg_is_tts: List[int] = []

    for idx, it in enumerate(packet):
        use_tts = idx == ni
        p = tts_wav if use_tts else it.path
        wav = load_wav_16k(p)

        seg_wavs.append(wav)
        seg_dur_orig.append(float(it.dur_s))
        seg_tgt.append(float(it.mos))
        seg_keys.append(it.key)
        seg_is_tts.append(1 if use_tts else 0)

    elapsed = 0.0
    pred_list: List[float] = []
    tgt_list: List[float] = []
    dur_list: List[float] = []
    rm_rows: List[List[object]] = []

    for j, (wav, dur, tgt, key, is_tts) in enumerate(
        zip(seg_wavs, seg_dur_orig, seg_tgt, seg_keys, seg_is_tts), start=1
    ):
        ok, rt, mem, mos_pred, where = safe_infer_global(
            predictor_gpu, predictor_cpu, wav,
            f"{dataset}:{system_id}:forward:rm:seg{j}"
        )
        if not ok:
            print(f"[WARN] {dataset}:{system_id} stop rm at seg{j}")
            break

        pred_list.append(float(mos_pred))
        tgt_list.append(float(tgt))
        dur_list.append(float(dur))
        elapsed += float(dur)

        pred_rm = durw_mean(pred_list, dur_list)
        tgt_rm = durw_mean(tgt_list, dur_list)
        rm_rows.append([round(elapsed, 6), round(pred_rm, 6), round(tgt_rm, 6), key, int(is_tts)])

    if not rm_rows:
        print(f"[WARN] {dataset}:{system_id} no rm rows")
        return

    concat_rows: List[List[object]] = []
    concat_list: List[np.ndarray] = []
    elapsed2 = 0.0
    tgt_prefix: List[float] = []
    dur_prefix: List[float] = []

    for n, (wav, dur, tgt, key, is_tts) in enumerate(
        zip(seg_wavs, seg_dur_orig, seg_tgt, seg_keys, seg_is_tts), start=1
    ):
        concat_list.append(wav)
        elapsed2 += float(dur)

        tgt_prefix.append(float(tgt))
        dur_prefix.append(float(dur))

        wav_cat = np.concatenate(concat_list).astype(np.float32)
        ok, rt, mem, mos_pred, where = safe_infer_global(
            predictor_gpu, predictor_cpu, wav_cat,
            f"{dataset}:{system_id}:forward:concat:n{n}"
        )
        if not ok:
            print(f"[WARN] {dataset}:{system_id} stop concat at n={n}")
            break

        tgt_cat = durw_mean(tgt_prefix, dur_prefix)
        concat_rows.append([round(elapsed2, 6), round(float(mos_pred), 6), round(float(tgt_cat), 6), key, int(is_tts)])

    if not concat_rows:
        print(f"[WARN] {dataset}:{system_id} no concat rows")
        return

    write_csv(
        out_dir / "global_mos_running_mean.csv",
        ["elapsed_s_orig", "mos_pred_rm_durw", "mos_target_rm_durw", "key", "is_tts_replaced_segment"],
        rm_rows,
    )
    write_csv(
        out_dir / "global_mos_concat.csv",
        ["seconds_orig", "mos_pred_concat", "mos_target_concat_durw", "key_last_added", "is_tts_replaced_segment"],
        concat_rows,
    )

    print(f"[OK] {dataset}:{system_id} -> {out_dir}")


def parse_systems_arg(s: str, defaults: List[str]) -> List[str]:
    if not s.strip():
        return defaults
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT)
    ap.add_argument("--checkpoint-name", type=str, default=CHECKPOINT_NAME_DEFAULT)

    ap.add_argument("--dataset", type=str, choices=["bvcc", "somos", "both"], default="both")
    ap.add_argument("--systems", type=str, default="")

    ap.add_argument("--bvcc-test-list", type=str, default=BVCC_TEST_LIST_DEFAULT)
    ap.add_argument("--bvcc-wav-root", type=str, default=BVCC_WAV_ROOT_DEFAULT)

    ap.add_argument("--somos-test-list", type=str, default=SOMOS_TEST_LIST_DEFAULT)
    ap.add_argument("--somos-wav-root", type=str, default=SOMOS_WAV_ROOT_DEFAULT)

    ap.add_argument("--target-seconds", type=float, default=180.0)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=5000)
    ap.add_argument("--allow-repeat", action="store_true", default=True)

    ap.add_argument("--pick", type=str, choices=["middle", "first", "last"], default="middle")
    ap.add_argument("--needle-index", type=int, default=None)

    ap.add_argument("--tts-root", type=str, default=None)
    ap.add_argument("--skip-missing-tts", action="store_true", default=True)
    ap.add_argument("--out-tag", type=str, default="")

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    ckpt_name = args.checkpoint_name
    ckpt_stem = Path(ckpt_name).stem

    tts_root = auto_find_tts_root(args.tts_root)

    host = socket.gethostname()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.out_tag}" if args.out_tag else ""
    run_name = f"needle_tts_fwd_orig_{ckpt_stem}_t{int(args.target_seconds)}_s{int(args.start_index)}{tag}_{host}_{stamp}"

    run_dir = model_dir / "results" / "TTS" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("run_dir:", run_dir)
    print("tts_root:", tts_root)
    print("dataset:", args.dataset)
    print("pick:", args.pick, "needle_index:", args.needle_index)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_gpu = SpeechQualityPredictor(
        storage_dir=str(model_dir),
        checkpoint_name=ckpt_name,
        return_numpy=True,
        device=device,
    )

    predictor_cpu = None
    if FALLBACK_TO_CPU and device == "cuda":
        predictor_cpu = SpeechQualityPredictor(
            storage_dir=str(model_dir),
            checkpoint_name=ckpt_name,
            return_numpy=True,
            device="cpu",
        )

    if args.dataset in ("bvcc", "both"):
        bvcc_test_list = Path(args.bvcc_test_list)
        bvcc_wav_root = Path(args.bvcc_wav_root)
        ensure_file(bvcc_test_list, "BVCC_TEST_LIST")
        ensure_file(bvcc_wav_root, "BVCC_WAV_ROOT")

        items = parse_list_generic(bvcc_test_list)
        systems = parse_systems_arg(args.systems, BVCC_SYSTEMS_DEFAULT)
        for sid in systems:
            out = run_dir / "bvcc" / sid / "forward"
            run_one_system(
                dataset="bvcc",
                system_id=sid,
                items=items,
                wav_root=bvcc_wav_root,
                matcher=bvcc_matcher,
                predictor_gpu=predictor_gpu,
                predictor_cpu=predictor_cpu,
                tts_root=tts_root,
                out_dir=out,
                target_seconds=args.target_seconds,
                start_index=args.start_index,
                max_files=args.max_files,
                allow_repeat=args.allow_repeat,
                pick=args.pick,
                needle_index=args.needle_index,
                skip_missing_tts=args.skip_missing_tts,
            )

    if args.dataset in ("somos", "both"):
        somos_test_list = Path(args.somos_test_list)
        somos_wav_root = Path(args.somos_wav_root)
        ensure_file(somos_test_list, "SOMOS_TEST_LIST")
        ensure_file(somos_wav_root, "SOMOS_WAV_ROOT")

        items = parse_list_generic(somos_test_list)
        systems = parse_systems_arg(args.systems, SOMOS_SYSTEMS_DEFAULT)
        for sid in systems:
            sid3 = str(sid).zfill(3)
            out = run_dir / "somos" / sid3 / "forward"
            run_one_system(
                dataset="somos",
                system_id=sid3,
                items=items,
                wav_root=somos_wav_root,
                matcher=somos_matcher,
                predictor_gpu=predictor_gpu,
                predictor_cpu=predictor_cpu,
                tts_root=tts_root,
                out_dir=out,
                target_seconds=args.target_seconds,
                start_index=args.start_index,
                max_files=args.max_files,
                allow_repeat=args.allow_repeat,
                pick=args.pick,
                needle_index=args.needle_index,
                skip_missing_tts=args.skip_missing_tts,
            )

    print("\nDONE:", run_dir)


if __name__ == "__main__":
    main()
