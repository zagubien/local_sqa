from __future__ import annotations

import argparse
import csv
import re
import shutil
import socket
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BVCC_SYSTEM_IDS = [
    "sys78aec",
    "sys83aed",
    "sys91caa",
    "sys6c11c",
    "sys8f532",
    "sysd81da",
]

SOMOS_SYSTEM_IDS = ["061", "057", "110", "124", "191"]

BVCC_TEST_LIST = Path("/net/db/BVCC/main/DATA/sets/test_mos_list.txt")
BVCC_WAV_ROOT = Path("/net/db/BVCC/main/DATA/wav")
BVCC_TRANSCRIPTS_TRUTH = Path("/net/db/BVCC/main/main_track_truth_transcripts.txt")
BVCC_SECRET_UTT_MAP = Path("/net/db/BVCC/main/secret_utt_mappings.txt")
BVCC_EXTRA_TRANSCRIPTS = [
    Path("/net/db/BVCC/main/ESPnet_VCC_missing_transcripts.txt"),
]

SOMOS_WAV_ROOT = Path("/net/db/somos/audios")
SOMOS_UTT_TRANSCRIPTS = Path("/net/db/somos/all_transcripts.txt")
SOMOS_LIST_CANDIDATES = [
    Path("/net/db/somos/training_files/split1/clean/test_mos_list.txt"),
    Path("/net/db/somos/training_files/split1/full/test_mos_list.txt"),
]


@dataclass
class Item:
    key: str
    mos: float
    path: Path
    dur_s: float


def pick_existing_path(cands: List[Path]) -> Optional[Path]:
    for p in cands:
        if p.exists():
            return p
    return None


def ensure_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {name}: {path}")


def wav_duration_seconds(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        return 0.0 if sr <= 0 else float(n) / float(sr)


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


def parse_transcripts_kv(path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s:
            continue

        if "\t" in s:
            a, b = s.split("\t", 1)
        elif "|" in s:
            a, b = s.split("|", 1)
        else:
            parts = s.split(maxsplit=1)
            if len(parts) == 1:
                a, b = parts[0], ""
            else:
                a, b = parts[0], parts[1]

        key = a.strip()
        txt = b.strip()
        if key:
            m[key] = txt
    return m


def parse_bvcc_secret_utt_map(path: Path) -> Dict[str, str]:
    # format: sysXXXX-uttYYYY.wav  BC2008-....wav
    out: Dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        left = parts[0].strip()
        right = parts[1].strip()
        if left and right:
            out[left] = right
    return out


def build_bvcc_transcript_map(
    truth_path: Path,
    utt_map_path: Path,
    extra_transcripts: List[Path],
) -> Dict[str, str]:
    # truth id -> text
    truth_raw: Dict[str, str] = {}
    truth_raw.update(parse_transcripts_kv(truth_path))
    for p in extra_transcripts:
        if p.exists():
            truth_raw.update(parse_transcripts_kv(p))

    # normalize truth keys to support with/without .wav
    truth: Dict[str, str] = {}
    for k, v in truth_raw.items():
        kk = k.strip()
        if not kk:
            continue
        truth[kk] = v
        if kk.endswith(".wav"):
            truth[kk[:-4]] = v
        else:
            truth[f"{kk}.wav"] = v

    # sys-utt.wav -> truth.wav
    lut = parse_bvcc_secret_utt_map(utt_map_path)

    out: Dict[str, str] = {}
    # also keep truth in map, can be handy
    out.update(truth)

    for left, right in lut.items():
        txt = truth.get(right)
        if txt is None:
            txt = truth.get(right[:-4]) if right.endswith(".wav") else None
        if not txt:
            continue

        # direct keys
        out[left] = txt
        out[Path(left).name] = txt
        out[Path(left).stem] = txt

        # also utt-only alias
        m = re.search(r"(utt[0-9a-fA-F]+)", left)
        if m:
            u = m.group(1)
            out[u] = txt
            out[f"{u}.wav"] = txt

    return out


def transcript_for_key(key: str, tmap: Dict[str, str]) -> str:
    bn = Path(key).name
    stem = Path(key).stem

    for k in (key, bn, stem, f"{stem}.wav", f"{bn}.wav"):
        if k in tmap:
            return tmap[k]

    m = re.search(r"(utt[0-9a-fA-F]+)", bn)
    if m:
        u = m.group(1)
        for k in (u, f"{u}.wav"):
            if k in tmap:
                return tmap[k]

    return ""


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
) -> List[Item]:
    filtered = [(k, mos) for (k, mos) in items if matcher(k, system_id)]
    if not filtered:
        raise RuntimeError(f"no items for system_id={system_id}")

    start = start_index % len(filtered)
    base = filtered[start:] + filtered[:start]

    packet: List[Item] = []
    total = 0.0

    while total < target_seconds and len(packet) < max_files:
        for key, mos in base:
            if total >= target_seconds or len(packet) >= max_files:
                break
            ap = resolve_audio_path(key, wav_root)
            dur = wav_duration_seconds(ap)
            packet.append(Item(key=key, mos=float(mos), path=ap, dur_s=float(dur)))
            total += float(dur)
        if not allow_repeat:
            break

    return packet


def copy_packet(packet: List[Item], out_wav_dir: Path) -> None:
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    for it in packet:
        dst = out_wav_dir / it.path.name
        if not dst.exists():
            shutil.copy2(it.path, dst)


def write_packet_csv(packet: List[Item], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "mos", "src_path", "dur_s", "copied_wav"])
        for it in packet:
            w.writerow([it.key, f"{it.mos:.6f}", str(it.path), f"{it.dur_s:.6f}", it.path.name])


def write_transcripts_tsv(packet: List[Item], tmap: Dict[str, str], out_tsv: Path) -> Tuple[int, int]:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    total = 0
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("wav\ttranscript\n")
        for it in packet:
            total += 1
            txt = transcript_for_key(it.key, tmap)
            if not txt:
                missing += 1
            f.write(f"{it.path.name}\t{txt}\n")
    return total, missing


def tar_dir(src_dir: Path, tar_path: Path) -> None:
    import tarfile
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default="/net/vol/zigor/needle_download_prep")
    ap.add_argument("--target-seconds", type=float, default=180.0)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=5000)
    ap.add_argument("--allow-repeat", action="store_true", default=True)
    ap.add_argument("--no-tar", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    host = socket.gethostname()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"needle_{stamp}_{host}"
    out_root.mkdir(parents=True, exist_ok=True)

    print("host:", host)
    print("out_root:", out_root)

    # BVCC
    ensure_file(BVCC_TEST_LIST, "BVCC_TEST_LIST")
    ensure_file(BVCC_WAV_ROOT, "BVCC_WAV_ROOT")
    ensure_file(BVCC_TRANSCRIPTS_TRUTH, "BVCC_TRANSCRIPTS_TRUTH")
    ensure_file(BVCC_SECRET_UTT_MAP, "BVCC_SECRET_UTT_MAP")

    bvcc_items = parse_list_generic(BVCC_TEST_LIST)
    bvcc_tmap = build_bvcc_transcript_map(
        truth_path=BVCC_TRANSCRIPTS_TRUTH,
        utt_map_path=BVCC_SECRET_UTT_MAP,
        extra_transcripts=BVCC_EXTRA_TRANSCRIPTS,
    )

    bvcc_dir = out_root / "bvcc"
    (bvcc_dir / "provenance").mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        shutil.copy2(BVCC_TEST_LIST, bvcc_dir / "provenance" / BVCC_TEST_LIST.name)
        shutil.copy2(BVCC_TRANSCRIPTS_TRUTH, bvcc_dir / "provenance" / BVCC_TRANSCRIPTS_TRUTH.name)
        shutil.copy2(BVCC_SECRET_UTT_MAP, bvcc_dir / "provenance" / BVCC_SECRET_UTT_MAP.name)
        for p in BVCC_EXTRA_TRANSCRIPTS:
            if p.exists():
                shutil.copy2(p, bvcc_dir / "provenance" / p.name)

    for sid in BVCC_SYSTEM_IDS:
        pkt = build_packet_forward(
            items=bvcc_items,
            system_id=sid,
            wav_root=BVCC_WAV_ROOT,
            target_seconds=args.target_seconds,
            start_index=args.start_index,
            max_files=args.max_files,
            allow_repeat=args.allow_repeat,
            matcher=bvcc_matcher,
        )
        print(f"[bvcc] {sid}: n={len(pkt)} total_s={sum(x.dur_s for x in pkt):.2f}")

        if not args.dry_run:
            d = bvcc_dir / sid
            write_packet_csv(pkt, d / "packet.csv")
            copy_packet(pkt, d / "wav")
            total, missing = write_transcripts_tsv(pkt, bvcc_tmap, d / "transcripts.tsv")
            if missing:
                print(f"  transcripts missing: {missing}/{total}")

    # SOMOS (unchanged)
    ensure_file(SOMOS_WAV_ROOT, "SOMOS_WAV_ROOT")
    ensure_file(SOMOS_UTT_TRANSCRIPTS, "SOMOS_UTT_TRANSCRIPTS")

    somos_list = pick_existing_path(SOMOS_LIST_CANDIDATES)
    if somos_list is None:
        raise FileNotFoundError(f"no SOMOS test list found. tried: {SOMOS_LIST_CANDIDATES}")

    somos_items = parse_list_generic(somos_list)
    somos_tmap = parse_transcripts_kv(SOMOS_UTT_TRANSCRIPTS)
    for k, v in list(somos_tmap.items()):
        if not k.endswith(".wav"):
            somos_tmap[f"{k}.wav"] = v

    somos_dir = out_root / "somos"
    (somos_dir / "provenance").mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        shutil.copy2(somos_list, somos_dir / "provenance" / somos_list.name)
        shutil.copy2(SOMOS_UTT_TRANSCRIPTS, somos_dir / "provenance" / SOMOS_UTT_TRANSCRIPTS.name)

    for sid in SOMOS_SYSTEM_IDS:
        pkt = build_packet_forward(
            items=somos_items,
            system_id=sid,
            wav_root=SOMOS_WAV_ROOT,
            target_seconds=args.target_seconds,
            start_index=args.start_index,
            max_files=args.max_files,
            allow_repeat=args.allow_repeat,
            matcher=somos_matcher,
        )
        print(f"[somos] {sid}: n={len(pkt)} total_s={sum(x.dur_s for x in pkt):.2f}")

        if not args.dry_run:
            d = somos_dir / sid
            write_packet_csv(pkt, d / "packet.csv")
            copy_packet(pkt, d / "wav")
            total, missing = write_transcripts_tsv(pkt, somos_tmap, d / "transcripts.tsv")
            if missing:
                print(f"  transcripts missing: {missing}/{total}")

    if not args.dry_run and not args.no_tar:
        tar_path = out_root.parent / f"{out_root.name}.tar.gz"
        print("creating tar:", tar_path)
        tar_dir(out_root, tar_path)
        print("tar done:", tar_path)

    print("done:", out_root)


if __name__ == "__main__":
    main()
