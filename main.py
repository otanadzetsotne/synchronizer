#!/usr/bin/env python3
"""
sync_ssd.py - Two-way synchronization with backup to SSD
-----------------------------------------------------------------------

• Supports multiple directory pairs (LOCAL↔SSD) described in a YAML config
• Creates full file copies in the .backups/<PAIR>/<TIMESTAMP>/ directory before any
  operation that changes the SSD content
• Keeps backups for RETENTION_DAYS, then deletes them
• Has a --dry-run mode, logging, and a lock file in the system TEMP directory

Example usage:
    python sync_ssd.py --config config.yaml
    python sync_ssd.py --config config.yaml --dry-run

Configuration parameters (YAML):
--------------------------------------------------
    pairs:
      - local:  /Users/tsotne/Documents
        ssd:    /Volumes/EXT/DOCS
      - local:  /Users/tsotne/Projects
        ssd:    /Volumes/EXT/PROJ
    backup_root: /Volumes/EXT/.backups
    retention_days: 14
    delete_policy: safe   # safe | mirror | sync

Dependencies: Python≥3.8, PyYAML (pip install pyyaml)
"""
import argparse
import hashlib
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

LOG = logging.getLogger("sync_ssd")

# -----------------------  LOCK‑FILE UTILITIES  ----------------------- #

@contextmanager
def single_instance_lock(lock_path: Path):
    """Create a simple lock file; fail if it already exists"""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        LOG.debug("Acquired lock %s", lock_path)
        yield
    finally:
        try:
            lock_path.unlink()
            LOG.debug("Released lock %s", lock_path)
        except FileNotFoundError:
            pass

# -----------------------  FILE META & DIFF  ------------------------- #

class FileMeta:
    __slots__ = ("size", "mtime")

    def __init__(self, size: int, mtime: float):
        self.size = size
        self.mtime = mtime

    def newer_than(self, other: "FileMeta") -> bool:
        return self.mtime > other.mtime + 1e-3  # 1 ms tolerance


def scan_tree(root: Path) -> Dict[Path, "FileMeta"]:
    """Returns a dictionary rel_path → FileMeta"""
    meta: Dict[Path, FileMeta] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        stat = path.stat()
        meta[rel] = FileMeta(stat.st_size, stat.st_mtime)
    return meta


Action = Tuple[str, Path]  # (COPY_LOCAL_TO_SSD | COPY_SSD_TO_LOCAL | DELETE | UPDATE), rel_path


def build_diff(local: Dict[Path, FileMeta], ssd: Dict[Path, FileMeta], delete_policy: str) -> List[Action]:
    plan: List[Action] = []
    all_paths = set(local) | set(ssd)
    for rel in sorted(all_paths):
        in_local = rel in local
        in_ssd = rel in ssd
        if in_local and not in_ssd:
            plan.append(("COPY_LOCAL_TO_SSD", rel))
        elif in_ssd and not in_local:
            if delete_policy == "mirror":
                plan.append(("DELETE_SSD", rel))
            elif delete_policy == "sync":
                plan.append(("COPY_SSD_TO_LOCAL", rel))
            else:
                pass
        else:  # both present
            l_meta = local[rel]
            s_meta = ssd[rel]
            if l_meta.newer_than(s_meta):
                plan.append(("UPDATE_SSD", rel))
            elif s_meta.newer_than(l_meta):
                plan.append(("COPY_SSD_TO_LOCAL", rel))
            # else identical - do nothing
    return plan

# -----------------------  BACKUP & FILE OPS  ------------------------ #

def make_ts_folder(backup_root: Path, pair_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = backup_root / pair_name / ts
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def copy_for_backup(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_copy(src: Path, dst: Path):
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)  # atomic operation


def delete_path(path: Path):
    path.unlink()

# -----------------------  RETENTION CLEAN  -------------------------- #

def purge_old_backups(pair_backup_dir: Path, retention_days: int):
    if not pair_backup_dir.exists():
        return
    now = datetime.now()
    for ts_dir in pair_backup_dir.iterdir():
        try:
            ts = datetime.strptime(ts_dir.name, "%Y%m%d-%H%M%S")
        except ValueError:
            continue  # skip unrelated files
        if now - ts > timedelta(days=retention_days):
            shutil.rmtree(ts_dir)
            LOG.info("Removed old backup %s", ts_dir)

# -----------------------  VERIFY HELPERS   -------------------------- #

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_identical(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and sha256_of(a) == sha256_of(b)

# -----------------------  MAIN PER‑PAIR LOOP ------------------------ #

def process_pair(local_root: Path, ssd_root: Path, backup_root: Path, retention: int, delete_policy: str, dry: bool):
    pair_name = ssd_root.name
    LOG.info("=== Pair %s → %s ===", local_root, ssd_root)

    # 1) retention cleanup
    purge_old_backups(backup_root / pair_name, retention)

    # 2) scan
    local_meta = scan_tree(local_root)
    ssd_meta = scan_tree(ssd_root)

    # 3) diff
    plan = build_diff(local_meta, ssd_meta, delete_policy)
    LOG.info("Plan: %d actions", len(plan))
    for action, rel in plan:
        LOG.debug("  %s %s", action, rel)

    if dry:
        LOG.info("Dry-run mode - not executing anything"); return

    ts_folder = make_ts_folder(backup_root, pair_name)

    for action, rel in plan:
        local_path = local_root / rel
        ssd_path = ssd_root / rel
        try:
            if action in ("UPDATE_SSD", "DELETE_SSD", "COPY_LOCAL_TO_SSD"):
                # backup old version
                if ssd_path.exists():
                    copy_for_backup(ssd_path, ts_folder / rel)
            if action in ("COPY_LOCAL_TO_SSD", "UPDATE_SSD"):
                safe_copy(local_path, ssd_path)
            elif action == "COPY_SSD_TO_LOCAL":
                safe_copy(ssd_path, local_path)
            elif action == "DELETE_SSD":
                delete_path(ssd_path)
            else:
                LOG.warning("Unknown action %s", action)
        except Exception as e:
            LOG.exception("Error processing %s %s: %s", action, rel, e)

# -----------------------  ARGPARSE & ENTRY -------------------------- #

def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        level=level)


def main():
    ap = argparse.ArgumentParser(description="Sync folders with SSD backups")
    ap.add_argument("--config", required=True, type=Path, help="YAML configuration file")
    ap.add_argument("--dry-run", action="store_true", help="Show plan but do not copy")
    ap.add_argument("--verbose", action="store_true", help="Debug output")
    args = ap.parse_args()

    configure_logging(args.verbose)
    cfg = load_config(args.config)

    backup_root = Path(cfg["backup_root"]).expanduser()
    retention = int(cfg.get("retention_days", 14))
    delete_policy = cfg.get("delete_policy", "safe")

    # Use system TEMP directory for the lock file
    lock_path = Path(tempfile.gettempdir()) / "sync_ssd.lock"

    with single_instance_lock(lock_path):
        for pair in cfg["pairs"]:
            process_pair(
                Path(pair["local"]).expanduser(),
                Path(pair["ssd"]).expanduser(),
                backup_root,
                retention,
                delete_policy,
                args.dry_run,
            )
    LOG.info("All done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning("Interrupted")
