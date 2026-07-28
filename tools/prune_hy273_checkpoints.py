"""Prune large HY273 training checkpoints while preserving resume and milestones."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP_RE = re.compile(r"step_(\d+)\.pt$")


def parse_step(path: Path) -> int | None:
    match = STEP_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--keep_recent", type=int, default=8)
    parser.add_argument("--keep_every", type=int, default=50000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    step_paths: list[tuple[int, Path]] = []
    for path in model_dir.glob("step_*.pt"):
        step = parse_step(path)
        if step is not None:
            step_paths.append((step, path))
    step_paths.sort(key=lambda item: item[0])

    keep: set[Path] = set()
    if args.keep_recent > 0:
        keep.update(path for _step, path in step_paths[-args.keep_recent :])
    if args.keep_every > 0:
        keep.update(path for step, path in step_paths if step % args.keep_every == 0)

    delete = [path for _step, path in step_paths if path not in keep]
    freed_bytes = 0
    deleted: list[str] = []
    for path in delete:
        try:
            freed_bytes += path.stat().st_size
            if not args.dry_run:
                path.unlink()
            deleted.append(str(path))
        except FileNotFoundError:
            continue

    summary = {
        "model_dir": str(model_dir),
        "dry_run": bool(args.dry_run),
        "step_checkpoints_seen": len(step_paths),
        "kept_step_checkpoints": len(keep),
        "deleted_step_checkpoints": len(deleted),
        "freed_gb": freed_bytes / (1024**3),
        "kept": [str(path) for path in sorted(keep, key=lambda p: parse_step(p) or -1)],
        "deleted": deleted,
        "latest_exists": (model_dir / "latest.pt").exists(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
