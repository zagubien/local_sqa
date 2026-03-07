from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/Users/igorzagubien/BA/workstation/test/local_sqa")  # hart setzen, damit nix daneben geht
TARGET_DIRNAME = "_debug"

def main() -> None:
    root = ROOT.resolve()
    debug_dirs = [p for p in root.rglob(TARGET_DIRNAME) if p.is_dir() and p.name == TARGET_DIRNAME]

    print(f"root: {root}")
    print(f"found {len(debug_dirs)} '{TARGET_DIRNAME}' dirs")

    for p in sorted(debug_dirs):
        rel = p.relative_to(root)
        shutil.rmtree(p)
        print(f"[del] deleted: {rel}")

if __name__ == "__main__":
    main()