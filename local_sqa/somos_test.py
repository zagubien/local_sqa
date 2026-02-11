import csv
import time
import zlib
from pathlib import Path

import numpy as np
import psutil
import torch

import audiofile
from rVADfast import rVADfast

from local_sqa.modules.ssl_mos import SpeechQualityPredictor, SAMPLING_RATE
from local_sqa.modules.data_loader import LoadAudio


MODEL_DIR = "/net/vol/zigor/checkpoints/26"
CHECKPOINT_NAME = "ckpt_best_SRCC.pth"

SOMOS_TEST_LIST = Path("/net/db/somos/training_files/split1/clean/test_mos_list.txt")
# SOMOS_TEST_LIST = Path("/net/db/somos/training_files/split1/full/test_mos_list.txt")

SOMOS_WAV_ROOT = Path("/net/db/somos/audios")

SYSTEM_IDS = ["061", "057", "110", "124", "191"]
MODES_TO_RUN = ["forward", "reverse", "random"]

TARGET_SECONDS = 180.0
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


# debug artifacts
SAVE_DEBUG_MEDIA = True
DEBUG_MAX_SEGMENTS = 10
SAVE_DEBUG_COMBINED_PREVIEW = True
COMBINED_PREVIEW_SECONDS = 20.0

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


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


def parse_somos_list(list_path: Path):
    items = []
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


def build_paths():
    tag = SOMOS_TEST_LIST.parent.name
    model_tag = Path(MODEL_DIR).name
    ckpt_tag = Path(CHECKPOINT_NAME).stem

    run_root = Path(MODEL_DIR) / "results"
    run_root.mkdir(parents=True, exist_ok=True)

    run_dir = run_root / f"somos_{tag}_{ckpt_tag}_t{int(TARGET_SECONDS)}_s{START_INDEX}"
    run_dir.mkdir(parents=True, exist_ok=True)

    packet_dir = run_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)

    return tag, model_tag, ckpt_tag, run_dir, packet_dir


TAG, MODEL_TAG, CKPT_TAG, RUN_DIR, PACKET_DIR = build_paths()


def packet_path(system_id: str) -> Path:
    return PACKET_DIR / f"packet_somos_{TAG}_{CKPT_TAG}_sys{system_id}_t{int(TARGET_SECONDS)}_s{START_INDEX}.csv"


def build_packet(items, system_id: str, target_seconds: float):
    filtered = [(utt, mos) for (utt, mos) in items if extract_system_id(utt) == system_id]
    if not filtered:
        sample = [extract_system_id(u) for (u, _) in items[:20]]
        raise ValueError(f"no items found for SYSTEM_ID={system_id}. example parsed ids: {sample}")

    start = START_INDEX % len(filtered)
    base = filtered[start:] + filtered[:start]

    packet = []
    total_s = 0.0

    while total_s < target_seconds and len(packet) < MAX_FILES:
        for utt, mos_t in base:
            if total_s >= target_seconds or len(packet) >= MAX_FILES:
                break

            p = SOMOS_WAV_ROOT / utt
            if not p.exists():
                p2 = SOMOS_WAV_ROOT / f"{utt}.wav"
                if p2.exists():
                    p = p2
                else:
                    raise FileNotFoundError(f"missing audio file: {p}")

            wav = load_wav(p)
            dur = len(wav) / SAMPLING_RATE

            packet.append((utt, float(mos_t), str(p), float(dur)))
            total_s += float(dur)

        if not ALLOW_REPEAT:
            break

    return packet, float(total_s), len(filtered)


def save_packet(p: Path, system_id: str, packet, total_s: float, sys_count: int):
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_dir", str(MODEL_DIR)])
        w.writerow(["checkpoint", str(CHECKPOINT_NAME)])
        w.writerow(["system_id", system_id])
        w.writerow(["list_path", str(SOMOS_TEST_LIST)])
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


def out_dir_for(mode: str, system_id: str) -> Path:
    sid = f"sys{system_id}"
    if mode == "forward":
        return RUN_DIR / sid
    if mode == "reverse":
        return RUN_DIR / f"rev_{sid}"
    if mode == "random":
        return RUN_DIR / f"rand_{sid}"
    raise ValueError(mode)


def variant_subdir(base_out_dir: Path, variant: str) -> Path:
    d = base_out_dir / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    proc_map = []
    proc_cursor = 0

    for k, (s, e) in enumerate(intervals):
        seg = y[s:e]
        chunks.append(seg)

        ps = proc_cursor
        pe = proc_cursor + len(seg)
        proc_map.append({"type": "speech", "orig_s": s, "orig_e": e, "proc_s": ps, "proc_e": pe})
        proc_cursor = pe

        if k < len(intervals) - 1 and len(pause) > 0:
            chunks.append(pause)
            ps2 = proc_cursor
            pe2 = proc_cursor + len(pause)
            proc_map.append({"type": "pause", "proc_s": ps2, "proc_e": pe2})
            proc_cursor = pe2

    y_proc = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
    return y_proc, proc_map


def make_variants(wav: np.ndarray, sr: int, seed: int):
    out = {"original": wav}

    intervals = vad_intervals_rvad(wav, sr)
    min_proc = int((MIN_PROC_MS_FALLBACK / 1000.0) * sr)

    if not intervals:
        out["no_pause"] = wav
        out["long_pause"] = wav
        out["long_pause_noise"] = wav
        return out, intervals, [], []

    y_cut, _ = _build_processed(wav, sr, intervals, pause_s=0.0, pause_kind="zero", seed=seed ^ 0xA1B2C3D4)
    y_zero, zero_map = _build_processed(wav, sr, intervals, pause_s=INSERT_PAUSE_SECONDS, pause_kind="zero", seed=seed ^ 0x11223344)
    y_noise, noise_map = _build_processed(wav, sr, intervals, pause_s=INSERT_PAUSE_SECONDS, pause_kind="noise", seed=seed ^ 0x55667788)

    if len(y_cut) < min_proc:
        y_cut = wav
    if len(y_zero) < min_proc:
        y_zero = wav
    if len(y_noise) < min_proc:
        y_noise = wav

    out["no_pause"] = y_cut
    out["long_pause"] = y_zero
    out["long_pause_noise"] = y_noise
    return out, intervals, zero_map, noise_map


def _write_wav(path: Path, y: np.ndarray, sr: int):
    y = np.asarray(y, dtype=np.float32)
    audiofile.write(str(path), y, sr)


def _save_debug_plot(png_path: Path, y, sr, intervals, y_cut, y_zero, zero_map, y_noise, noise_map, title: str):
    if not HAS_MPL:
        return

    t = np.arange(len(y)) / sr
    tc = np.arange(len(y_cut)) / sr if len(y_cut) else np.zeros(0)
    tz = np.arange(len(y_zero)) / sr if len(y_zero) else np.zeros(0)
    tn = np.arange(len(y_noise)) / sr if len(y_noise) else np.zeros(0)

    plt.figure(figsize=(16, 11))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t, y, linewidth=0.7)
    ax1.set_title(f"{title} — original (spans = speech kept)")
    for s, e in intervals:
        ax1.axvspan(s / sr, e / sr, alpha=0.2)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("amp")

    ax2 = plt.subplot(4, 1, 2)
    if len(y_cut):
        ax2.plot(tc, y_cut, linewidth=0.7)
    ax2.set_title("no_pause")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("amp")

    ax3 = plt.subplot(4, 1, 3)
    if len(y_zero):
        ax3.plot(tz, y_zero, linewidth=0.7)
    ax3.set_title(f"long_pause ({INSERT_PAUSE_SECONDS}s zero)")
    for m in zero_map:
        if m.get("type") == "pause":
            ax3.axvspan(m["proc_s"] / sr, m["proc_e"] / sr, alpha=0.2)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("amp")

    ax4 = plt.subplot(4, 1, 4)
    if len(y_noise):
        ax4.plot(tn, y_noise, linewidth=0.7)
    ax4.set_title(f"long_pause_noise ({INSERT_PAUSE_SECONDS}s noise)")
    for m in noise_map:
        if m.get("type") == "pause":
            ax4.axvspan(m["proc_s"] / sr, m["proc_e"] / sr, alpha=0.2)
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("amp")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def run_one(items, predictor_gpu, predictor_cpu, system_id: str, mode: str):
    base_out = out_dir_for(mode, system_id)
    base_out.mkdir(parents=True, exist_ok=True)

    if SKIP_IF_DONE:
        all_done = True
        for v in VARIANTS:
            vd = base_out / v
            if not (vd / "global_mos_concat.csv").exists() or not (vd / "global_mos_running_mean.csv").exists():
                all_done = False
                break
        if all_done:
            print(f"[SKIP] {mode} | sys{system_id} (all variants already done)")
            return

    p_pkt = packet_path(system_id)
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
    debug_dir = base_out / "_debug"
    if SAVE_DEBUG_MEDIA:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for i, (utt, mos_t, p, dur) in enumerate(packet, start=1):
        wav = load_wav(p)
        seed = _make_seed("somos", TAG, MODEL_TAG, CKPT_TAG, system_id, mode, utt)
        variants, intervals, zero_map, noise_map = make_variants(wav, SAMPLING_RATE, seed)

        segs.append({
            "utt": utt,
            "mos_t": float(mos_t),
            "dur_orig": float(dur),
            "path": str(p),
            "variants": variants,
        })

        if SAVE_DEBUG_MEDIA and i <= DEBUG_MAX_SEGMENTS:
            title = f"somos-{TAG} | model{MODEL_TAG} | {CKPT_TAG} | sys{system_id} | {mode} | seg{i:03d} | {utt}"
            try:
                _write_wav(debug_dir / f"seg{i:03d}_original.wav", wav, SAMPLING_RATE)
                _write_wav(debug_dir / f"seg{i:03d}_no_pause.wav", variants["no_pause"], SAMPLING_RATE)
                _write_wav(debug_dir / f"seg{i:03d}_long_pause.wav", variants["long_pause"], SAMPLING_RATE)
                _write_wav(debug_dir / f"seg{i:03d}_long_pause_noise.wav", variants["long_pause_noise"], SAMPLING_RATE)
            except Exception as e:
                print(f"[WARN] debug wav write failed seg{i}: {e}")

            if HAS_MPL:
                try:
                    _save_debug_plot(
                        debug_dir / f"seg{i:03d}_waveforms.png",
                        wav, SAMPLING_RATE, intervals,
                        variants["no_pause"],
                        variants["long_pause"], zero_map,
                        variants["long_pause_noise"], noise_map,
                        title,
                    )
                except Exception as e:
                    print(f"[WARN] debug plot failed seg{i}: {e}")

    for variant in VARIANTS:
        out_dir = variant_subdir(base_out, variant)
        p_global_concat = out_dir / "global_mos_concat.csv"
        p_global_rm = out_dir / "global_mos_running_mean.csv"

        if SKIP_IF_DONE and p_global_concat.exists() and p_global_rm.exists():
            print(f"[SKIP] {mode} | sys{system_id} | {variant}")
            continue

        print(f"\n--- variant: {variant} ---")

        elapsed = 0.0
        pred_list, tgt_list, dur_list = [], [], []
        global_rm_rows = []

        for j, s in enumerate(segs, start=1):
            wav_v = s["variants"][variant]
            ok, rt, mem, mos_pred, where = safe_infer_global(
                predictor_gpu, predictor_cpu, wav_v, f"somos:{TAG}:m{MODEL_TAG}:{CKPT_TAG}:sys{system_id}:{mode}:{variant}:seg{j}"
            )
            if not ok:
                print(f"stopping at segment {j} (running mean) for variant={variant}")
                break

            pred_list.append(float(mos_pred))
            tgt_list.append(float(s["mos_t"]))
            dur_list.append(float(s["dur_orig"]))
            elapsed += float(s["dur_orig"])

            pred_rm = durw_mean(pred_list, dur_list)
            tgt_rm = durw_mean(tgt_list, dur_list)
            global_rm_rows.append([elapsed, pred_rm, tgt_rm])

        if len(global_rm_rows) == 0:
            print(f"no segments inferred (running mean) -> skip writing for variant={variant}")
            continue

        concat_wavs = []
        concat_secs = 0.0
        global_concat_rows = []
        tgt_prefix, dur_prefix = [], []

        for n, s in enumerate(segs, start=1):
            wav_v = s["variants"][variant]
            concat_wavs.append(wav_v)
            concat_secs += float(s["dur_orig"])

            tgt_prefix.append(float(s["mos_t"]))
            dur_prefix.append(float(s["dur_orig"]))

            wav_cat = np.concatenate(concat_wavs).astype(np.float32) if concat_wavs else np.zeros(0, dtype=np.float32)

            ok, rt, mem, mos_pred, where = safe_infer_global(
                predictor_gpu, predictor_cpu, wav_cat, f"somos:{TAG}:m{MODEL_TAG}:{CKPT_TAG}:sys{system_id}:{mode}:{variant}:concat{n}"
            )
            if not ok:
                print(f"stopping at n={n} (concat) for variant={variant}")
                break

            tgt_concat = durw_mean(tgt_prefix, dur_prefix)
            global_concat_rows.append([concat_secs, float(mos_pred), float(tgt_concat)])

        if len(global_concat_rows) == 0:
            print(f"concat failed entirely -> skip writing for variant={variant}")
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

        print("done:")
        print(p_global_concat)
        print(p_global_rm)

    if SAVE_DEBUG_MEDIA and SAVE_DEBUG_COMBINED_PREVIEW:
        try:
            sr = SAMPLING_RATE
            n_prev = int(COMBINED_PREVIEW_SECONDS * sr)
            for variant in VARIANTS:
                y_all = []
                for s in segs:
                    y_all.append(s["variants"][variant])
                comb = np.concatenate(y_all).astype(np.float32) if y_all else np.zeros(0, dtype=np.float32)
                prev = comb[:n_prev]
                _write_wav(debug_dir / f"combined_preview_{variant}_{int(COMBINED_PREVIEW_SECONDS)}s.wav", prev, sr)
        except Exception as e:
            print(f"[WARN] combined preview write failed: {e}")


def run_all():
    print("MODEL_DIR:", MODEL_DIR)
    print("CHECKPOINT:", CHECKPOINT_NAME)
    print("RUN_DIR:", RUN_DIR)
    print("TAG:", TAG)
    print("SYSTEM_IDS:", SYSTEM_IDS)
    print("MODES:", MODES_TO_RUN)
    print("TARGET_SECONDS:", TARGET_SECONDS, "ALLOW_REPEAT:", ALLOW_REPEAT)
    print("VARIANTS:", VARIANTS)
    print()

    items = parse_somos_list(SOMOS_TEST_LIST)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_gpu = SpeechQualityPredictor(
        storage_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        return_numpy=True,
        device=device,
    )

    predictor_cpu = None
    if FALLBACK_TO_CPU and device == "cuda":
        predictor_cpu = SpeechQualityPredictor(
            storage_dir=MODEL_DIR,
            checkpoint_name=CHECKPOINT_NAME,
            return_numpy=True,
            device="cpu",
        )

    for sid in SYSTEM_IDS:
        system_id = str(sid).zfill(3)
        for mode in MODES_TO_RUN:
            try:
                run_one(items, predictor_gpu, predictor_cpu, system_id, mode)
            except Exception as e:
                print(f"\n[ERROR] sys{system_id} / {mode}: {e}\n")


if __name__ == "__main__":
    run_all()
