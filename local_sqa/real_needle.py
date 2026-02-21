from __future__ import annotations

import argparse
import csv
import json
import socket
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio


# defaults (override via cli if needed)
MODEL_DIR_DEFAULT = "/net/vol/zigor/checkpoints/26"
CHECKPOINT_NAME_DEFAULT = "ckpt_best_SRCC.pth"

BVCC_TEST_LIST_DEFAULT = "/net/db/BVCC/main/DATA/sets/test_mos_list.txt"
BVCC_WAV_ROOT_DEFAULT = "/net/db/BVCC/main/DATA/wav"

SOMOS_TEST_LIST_DEFAULT = "/net/db/somos/training_files/split1/clean/test_mos_list.txt"
SOMOS_WAV_ROOT_DEFAULT = "/net/db/somos/audios"

BVCC_SYSTEMS_DEFAULT = ["sys78aec", "sys83aed", "sys91caa", "sys6c11c", "sys8f532", "sysd81da"]
SOMOS_SYSTEMS_DEFAULT = ["057", "061", "110", "124", "191"]

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


def safe_infer_global(predictor_gpu, predictor_cpu, wav: np.ndarray, tag: str, fallback_to_cpu: bool) -> Tuple[bool, Optional[float], Optional[float], Optional[float], str]:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"expected 1d waveform, got {wav.shape}")

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

        if (not fallback_to_cpu) or predictor_cpu is None:
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


def parse_list_generic(list_path: Path) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for i, ln in enumerate(list_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        if i == 0 and (ln.lower().startswith("utteranceid") or ln.lower().startswith("wav")):
            continue
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        k = parts[0].strip()
        try:
            v = float(parts[1].strip())
        except Exception:
            continue
        if k:
            m[k] = v
    return m


def extract_bvcc_system_id(wav_name: str) -> str:
    # sysXXXX-uttYYYY.wav -> sysXXXX
    base = Path(wav_name).name
    if "-" not in base:
        return "unknown"
    return base.split("-", 1)[0].strip()


def extract_somos_system_id(utterance_id: str) -> str:
    # something_like_xxx_061.wav -> 061
    u = utterance_id.strip()
    if u.endswith(".wav"):
        u = u[:-4]
    if "_" not in u:
        return "unknown"
    tail = u.split("_")[-1]
    if tail.isdigit():
        return tail.zfill(3)
    return "unknown"


def sort_key_tts_name(p: Path) -> Tuple[int, str]:
    # name like: <stem>__tts__i0007.wav
    name = p.name
    i = name.rfind("__tts__i")
    if i >= 0:
        j = i + len("__tts__i")
        k = j
        while k < len(name) and name[k].isdigit():
            k += 1
        if k > j:
            try:
                return (int(name[j:k]), name)
            except Exception:
                pass
    return (10**9, name)


@dataclass
class Seg:
    key: str
    src_wav_name: str
    wav_path: Path
    mos_target: float
    dur_s: float
    is_needle: bool


def tts_dir_for(tts_root: Path, dataset: str, system_id: str) -> Path:
    return tts_root / dataset / system_id / "forward"


def orig_name_from_tts(tts_name: str) -> str:
    # <stem>__tts__i0007.wav -> <stem>.wav
    base = Path(tts_name).name
    if "__tts__" in base:
        stem = base.split("__tts__", 1)[0]
        return f"{stem}.wav"
    return base


def pick_bvcc_needle(bvcc_list: Dict[str, float], wav_root: Path, mos_min: float, mos_max: float, prefer_name: Optional[str]) -> Tuple[str, float, Path]:
    if prefer_name:
        mos = bvcc_list.get(prefer_name, None)
        p = wav_root / prefer_name
        if mos is None or not p.exists():
            raise RuntimeError(f"preferred needle not found: {prefer_name} (mos={mos}) path={p}")
        if not (mos_min < mos < mos_max):
            raise RuntimeError(f"preferred needle mos not in range ({mos_min},{mos_max}): {prefer_name} mos={mos}")
        return prefer_name, float(mos), p

    cand: List[Tuple[float, str]] = []
    for k, mos in bvcc_list.items():
        if mos_min < mos < mos_max:
            p = wav_root / k
            if p.exists():
                cand.append((mos, k))
    if not cand:
        raise RuntimeError(f"no BVCC needle found with mos in ({mos_min},{mos_max}) under {wav_root}")

    cand.sort(key=lambda x: (x[0], x[1]))  # lowest mos first
    mos, name = cand[0]
    return name, float(mos), wav_root / name


def build_packet_from_tts(
    dataset: str,
    system_id: str,
    tts_root: Path,
    mos_map: Dict[str, float],
    target_seconds: float,
    start_index: int,
    allow_repeat: bool,
    max_files: int,
) -> List[Tuple[str, str, Path, float]]:
    # returns list of (key, src_wav_name, tts_path, mos_target)
    tdir = tts_dir_for(tts_root, dataset, system_id)
    if not tdir.exists():
        return []

    wavs = sorted([p for p in tdir.glob("*.wav") if p.is_file()], key=sort_key_tts_name)
    if not wavs:
        return []

    items: List[Tuple[str, str, Path, float]] = []
    for p in wavs:
        src = orig_name_from_tts(p.name)
        mos = mos_map.get(src, float("nan"))
        items.append((p.name, src, p, float(mos)))

    start = start_index % len(items)
    base = items[start:] + items[:start]

    out: List[Tuple[str, str, Path, float]] = []
    total_s = 0.0
    loops = 0
    while total_s < target_seconds and len(out) < max_files:
        for key, src, p, mos in base:
            if total_s >= target_seconds or len(out) >= max_files:
                break
            try:
                y = load_wav(p)
                dur = float(len(y) / SAMPLING_RATE)
            except Exception:
                continue
            out.append((key, src, p, mos))
            total_s += dur
        loops += 1
        if not allow_repeat:
            break
        if loops > 1000:
            break

    return out


def write_packet_csv(out_path: Path, segs: List[Seg], needle_info: Dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["needle_wav", needle_info.get("needle_wav", "")])
        w.writerow(["needle_mos", needle_info.get("needle_mos", "")])
        w.writerow(["needle_src_path", needle_info.get("needle_src_path", "")])
        w.writerow([])
        w.writerow(["idx", "key", "src_wav_name", "wav_path", "mos_target", "dur_s", "is_needle"])
        for i, s in enumerate(segs):
            w.writerow([i, s.key, s.src_wav_name, str(s.wav_path), s.mos_target, round(s.dur_s, 6), int(s.is_needle)])


def run_one_system(
    dataset: str,
    system_id: str,
    seg_tpl: List[Tuple[str, str, Path, float]],
    needle_name: str,
    needle_mos: float,
    needle_path: Path,
    predictor_gpu,
    predictor_cpu,
    out_dir: Path,
    fallback_to_cpu: bool,
    dry_run: bool,
) -> None:
    if not seg_tpl:
        print(f"[skip] {dataset}:{system_id} no tts wavs found")
        return

    # load all segment wavs once
    segs: List[Seg] = []
    for key, src, p, mos in seg_tpl:
        y = load_wav(p)
        dur = float(len(y) / SAMPLING_RATE)
        segs.append(Seg(key=key, src_wav_name=src, wav_path=p, mos_target=float(mos), dur_s=dur, is_needle=False))

    mid = len(segs) // 2
    replaced = segs[mid]

    # replace middle with real needle
    y_need = load_wav(needle_path)
    dur_need = float(len(y_need) / SAMPLING_RATE)
    segs[mid] = Seg(
        key=needle_name,
        src_wav_name=needle_name,
        wav_path=needle_path,
        mos_target=float(needle_mos),
        dur_s=dur_need,
        is_needle=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": dataset,
        "system_id": system_id,
        "out_dir": str(out_dir),
        "mid_index": int(mid),
        "replaced_tts_key": replaced.key,
        "replaced_src_wav_name": replaced.src_wav_name,
        "replaced_tts_path": str(replaced.wav_path),
        "needle_wav": needle_name,
        "needle_mos": float(needle_mos),
        "needle_src_path": str(needle_path),
        "n_segments": len(segs),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    write_packet_csv(out_dir / "packet.csv", segs, meta)

    if dry_run:
        print(f"[DRY] {dataset}:{system_id} segs={len(segs)} mid={mid} needle={needle_name} replace={replaced.key}")
        return

    # running mean (dur-weighted)
    elapsed = 0.0
    pred_list: List[float] = []
    tgt_list: List[float] = []
    dur_list: List[float] = []
    rm_rows: List[List[object]] = []

    for i, s in enumerate(segs, start=1):
        y = load_wav(s.wav_path)
        ok, rt, mem, mos_pred, where = safe_infer_global(
            predictor_gpu, predictor_cpu, y, f"{dataset}:{system_id}:rm:{i}", fallback_to_cpu
        )
        if not ok:
            print(f"[fail] {dataset}:{system_id} running-mean failed at i={i}")
            break

        pred_list.append(float(mos_pred))
        tgt_list.append(float(s.mos_target))
        dur_list.append(float(s.dur_s))
        elapsed += float(s.dur_s)

        pred_rm = durw_mean(pred_list, dur_list)
        tgt_rm = durw_mean(tgt_list, dur_list)

        rm_rows.append([
            round(float(elapsed), 6),
            round(float(pred_rm), 6) if np.isfinite(pred_rm) else "",
            round(float(tgt_rm), 6) if np.isfinite(tgt_rm) else "",
            s.key,
            int(s.is_needle),
        ])

    # concat
    concat_wavs: List[np.ndarray] = []
    concat_secs = 0.0
    tgt_prefix: List[float] = []
    dur_prefix: List[float] = []
    concat_rows: List[List[object]] = []

    for i, s in enumerate(segs, start=1):
        y = load_wav(s.wav_path)
        concat_wavs.append(y)
        concat_secs += float(s.dur_s)

        tgt_prefix.append(float(s.mos_target))
        dur_prefix.append(float(s.dur_s))

        ycat = np.concatenate(concat_wavs).astype(np.float32) if concat_wavs else np.zeros(0, dtype=np.float32)
        ok, rt, mem, mos_pred, where = safe_infer_global(
            predictor_gpu, predictor_cpu, ycat, f"{dataset}:{system_id}:concat:{i}", fallback_to_cpu
        )
        if not ok:
            print(f"[fail] {dataset}:{system_id} concat failed at i={i}")
            break

        tgt_concat = durw_mean(tgt_prefix, dur_prefix)

        concat_rows.append([
            round(float(concat_secs), 6),
            round(float(mos_pred), 6),
            round(float(tgt_concat), 6) if np.isfinite(tgt_concat) else "",
            s.key,
            int(s.is_needle),
        ])

    # write csvs
    with (out_dir / "global_mos_running_mean.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "mos_pred_rm_durw", "mos_target_rm_durw", "key_last_added", "is_real_needle_segment"])
        for r in rm_rows:
            w.writerow(r)

    with (out_dir / "global_mos_concat.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seconds", "mos_pred_concat", "mos_target_concat_durw", "key_last_added", "is_real_needle_segment"])
        for r in concat_rows:
            w.writerow(r)

    print(f"[OK] {dataset}:{system_id} -> {out_dir}")


def make_run_dir(model_dir: Path, ckpt_name: str, target_seconds: float, start_index: int) -> Path:
    host = socket.gethostname()
    ts = time.strftime("%Y%m%d_%H%M%S")
    ck = Path(ckpt_name).stem
    tag = f"reverse_tts_one_realneedle_{ck}_t{int(target_seconds)}_s{start_index}_{host}_{ts}"
    run_dir = model_dir / "results" / "TTS" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT)
    ap.add_argument("--checkpoint-name", type=str, default=CHECKPOINT_NAME_DEFAULT)

    ap.add_argument("--tts-root", type=str, default="")
    ap.add_argument("--dataset", choices=["bvcc", "somos", "both"], default="both")
    ap.add_argument("--systems", type=str, default="")  # comma-separated, optional

    ap.add_argument("--target-seconds", type=float, default=180.0)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=5000)
    ap.add_argument("--allow-repeat", action="store_true", default=True)

    ap.add_argument("--bvcc-test-list", type=str, default=BVCC_TEST_LIST_DEFAULT)
    ap.add_argument("--bvcc-wav-root", type=str, default=BVCC_WAV_ROOT_DEFAULT)
    ap.add_argument("--somos-test-list", type=str, default=SOMOS_TEST_LIST_DEFAULT)
    ap.add_argument("--somos-wav-root", type=str, default=SOMOS_WAV_ROOT_DEFAULT)

    ap.add_argument("--needle-mos-min", type=float, default=1.0)
    ap.add_argument("--needle-mos-max", type=float, default=2.0)
    ap.add_argument("--needle-name", type=str, default="")  # optional exact file name

    ap.add_argument("--fallback-to-cpu", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    tts_root = Path(args.tts_root).expanduser() if args.tts_root else (repo_root / "tts_replacements_out")

    model_dir = Path(args.model_dir)
    run_dir = make_run_dir(model_dir, args.checkpoint_name, args.target_seconds, args.start_index)

    print("repo_root:", repo_root)
    print("tts_root:", tts_root)
    print("run_dir:", run_dir)

    # predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_gpu = SpeechQualityPredictor(
        storage_dir=str(model_dir),
        checkpoint_name=str(args.checkpoint_name),
        return_numpy=True,
        device=device,
    )

    predictor_cpu = None
    if args.fallback_to_cpu and device == "cuda":
        predictor_cpu = SpeechQualityPredictor(
            storage_dir=str(model_dir),
            checkpoint_name=str(args.checkpoint_name),
            return_numpy=True,
            device="cpu",
        )

    # pick needle from bvcc list + wav root
    bvcc_list = parse_list_generic(Path(args.bvcc_test_list))
    needle_name, needle_mos, needle_path = pick_bvcc_needle(
        bvcc_list=bvcc_list,
        wav_root=Path(args.bvcc_wav_root),
        mos_min=args.needle_mos_min,
        mos_max=args.needle_mos_max,
        prefer_name=args.needle_name.strip() or None,
    )
    print(f"needle: {needle_name}  mos={needle_mos}  path={needle_path}")

    total_done = 0

    # systems override
    sys_override = [s.strip() for s in args.systems.split(",") if s.strip()] if args.systems else None

    if args.dataset in ("bvcc", "both"):
        mos_map_bvcc = bvcc_list
        systems = sys_override if sys_override is not None else BVCC_SYSTEMS_DEFAULT
        for sid in systems:
            packet = build_packet_from_tts(
                dataset="bvcc",
                system_id=sid,
                tts_root=tts_root,
                mos_map=mos_map_bvcc,
                target_seconds=args.target_seconds,
                start_index=args.start_index,
                allow_repeat=args.allow_repeat,
                max_files=args.max_files,
            )
            out = run_dir / "bvcc" / sid / "forward"
            run_one_system(
                dataset="bvcc",
                system_id=sid,
                seg_tpl=packet,
                needle_name=needle_name,
                needle_mos=needle_mos,
                needle_path=needle_path,
                predictor_gpu=predictor_gpu,
                predictor_cpu=predictor_cpu,
                out_dir=out,
                fallback_to_cpu=args.fallback_to_cpu,
                dry_run=args.dry_run,
            )
            if packet:
                total_done += 1

    if args.dataset in ("somos", "both"):
        somos_map = parse_list_generic(Path(args.somos_test_list))
        systems = sys_override if sys_override is not None else SOMOS_SYSTEMS_DEFAULT
        for sid in systems:
            sid3 = str(sid).zfill(3)
            packet = build_packet_from_tts(
                dataset="somos",
                system_id=sid3,
                tts_root=tts_root,
                mos_map=somos_map,  # key is usually "<utt>.wav" and our src_wav_name is "<stem>.wav"
                target_seconds=args.target_seconds,
                start_index=args.start_index,
                allow_repeat=args.allow_repeat,
                max_files=args.max_files,
            )
            out = run_dir / "somos" / sid3 / "forward"
            run_one_system(
                dataset="somos",
                system_id=sid3,
                seg_tpl=packet,
                needle_name=needle_name,
                needle_mos=needle_mos,
                needle_path=needle_path,
                predictor_gpu=predictor_gpu,
                predictor_cpu=predictor_cpu,
                out_dir=out,
                fallback_to_cpu=args.fallback_to_cpu,
                dry_run=args.dry_run,
            )
            if packet:
                total_done += 1

    print(f"DONE: systems_processed={total_done}  run_dir={run_dir}")


if __name__ == "__main__":
    main()
