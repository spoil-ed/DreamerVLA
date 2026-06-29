#!/usr/bin/env python3
"""Periodically upload W&B offline runs from a shared disk to W&B cloud.

For the deployment where an *offline* GPU machine writes W&B offline runs onto a
disk that this *online* machine also mounts. Because the disk is shared, this
machine reads the offline runs directly and runs ``wandb sync`` on them in place
-- no copy, no SSH, no rsync.

``wandb sync --sync-all`` is idempotent: after a successful upload it writes a
``.synced`` marker next to the run and skips it on later rounds, so the loop only
uploads runs that have not been synced yet. The W&B API key must live only on
this machine (run ``wandb login``); it is never read, stored, or printed here.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


@dataclasses.dataclass
class RelayConfig:
    wandb_dir: Path
    wandb_project: str | None
    wandb_entity: str | None
    interval: int
    wandb_bin: str
    dry_run: bool
    once: bool
    log_file: Path | None
    lock_file: Path


def parse_args(argv: list[str] | None = None) -> RelayConfig:
    p = argparse.ArgumentParser(
        prog="wandb_relay_sync.py",
        description=(
            "Periodically upload W&B offline runs from a shared-disk `wandb/` "
            "directory to W&B cloud via `wandb sync`. The directory is read in "
            "place; no copy, SSH, or rsync is used."
        ),
    )
    p.add_argument("--wandb-dir", required=True,
                   help="Shared-disk `wandb/` directory that DIRECTLY contains the "
                        "offline-run-* directories (readable from this machine).")
    p.add_argument("--wandb-project", default=None, help="W&B project to upload to.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity to upload to.")
    p.add_argument("--interval", type=int, default=60,
                   help="Seconds to wait between rounds (default: 60).")
    p.add_argument("--wandb-bin", default="wandb", help="Path to wandb binary.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the wandb command only; do not execute anything.")
    p.add_argument("--once", action="store_true",
                   help="Run a single round and exit (exit code reflects success).")
    p.add_argument("--log-file", default=None,
                   help="Optional log file (always also logs to the terminal).")
    p.add_argument("--lock-file", default=None,
                   help="Lock file path (default: <wandb-dir>/.wandb_relay.lock).")
    args = p.parse_args(argv)

    wandb_dir = Path(args.wandb_dir).expanduser()
    lock_file = (
        Path(args.lock_file).expanduser()
        if args.lock_file
        else wandb_dir / ".wandb_relay.lock"
    )
    log_file = Path(args.log_file).expanduser() if args.log_file else None
    return RelayConfig(
        wandb_dir=wandb_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        interval=args.interval,
        wandb_bin=args.wandb_bin,
        dry_run=args.dry_run,
        once=args.once,
        log_file=log_file,
        lock_file=lock_file,
    )


def build_wandb_sync_command(cfg: RelayConfig) -> list[str]:
    # No path argument: `wandb sync --sync-all` is run with cwd=<wandb-dir>.parent
    # so the CLI discovers the `./wandb` subdir there. (Passing an explicit path
    # with --sync-all does not reliably discover runs across wandb versions.)
    cmd = [cfg.wandb_bin, "sync", "--sync-all", "--include-offline"]
    if cfg.wandb_project:
        cmd += ["-p", cfg.wandb_project]
    if cfg.wandb_entity:
        cmd += ["-e", cfg.wandb_entity]
    return cmd


def sync_cwd(cfg: RelayConfig) -> str:
    # Run from the parent of the shared `wandb/` dir; `wandb sync --sync-all`
    # then finds the offline runs under `./wandb`.
    return str(cfg.wandb_dir.parent)


class RelayLockError(RuntimeError):
    """Raised when another relay process already holds the lock."""


def setup_logging(log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("wandb_relay")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def acquire_lock(lock_file: Path):
    """Acquire an exclusive, non-blocking flock. Return the open handle.

    Keep the returned handle open for the process lifetime; closing it (or the
    process exiting/crashing) releases the lock automatically.
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_file, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RelayLockError(
            f"Another relay process already holds the lock at {lock_file}. "
            f"Stop it first, or pass a different --lock-file."
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _run(
    cmd: list[str],
    logger: logging.Logger,
    dry_run: bool,
    label: str,
    cwd: str | None = None,
) -> bool:
    cwd_note = f" (cwd={cwd})" if cwd else ""
    logger.info("%s command:%s %s", label, cwd_note, " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        logger.info("dry-run: not executing %s", label)
        return True
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    except FileNotFoundError as exc:
        logger.error("%s failed: executable not found: %s (%s)", label, cmd[0], exc)
        return False
    if proc.stdout:
        logger.info("%s stdout:\n%s", label, proc.stdout.rstrip())
    if proc.stderr:
        logger.info("%s stderr:\n%s", label, proc.stderr.rstrip())
    if proc.returncode != 0:
        logger.error("%s failed with exit code %d", label, proc.returncode)
        return False
    return True


def run_once(cfg: RelayConfig, logger: logging.Logger) -> bool:
    """Run a single sync round. Never raises for command failures."""
    logger.info("wandb dir (shared, in-place): %s", cfg.wandb_dir)
    if not _run(
        build_wandb_sync_command(cfg), logger, cfg.dry_run, "wandb sync",
        cwd=sync_cwd(cfg),
    ):
        logger.error("round failed during wandb sync; will retry next round")
        return False
    logger.info("round succeeded")
    return True


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    logger = setup_logging(cfg.log_file)

    if cfg.dry_run:
        logger.info("DRY-RUN: printing the wandb command only; no sync, no lock")
        run_once(cfg, logger)
        return 0

    try:
        lock_handle = acquire_lock(cfg.lock_file)
    except RelayLockError as exc:
        logger.error("%s", exc)
        return 2

    stop = {"flag": False}

    def _handle(signum, _frame):
        logger.info("received signal %d; stopping after current round", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    try:
        if cfg.once:
            return 0 if run_once(cfg, logger) else 1

        logger.info("relay started; interval=%ds (Ctrl-C / SIGTERM to stop)", cfg.interval)
        while not stop["flag"]:
            run_once(cfg, logger)
            if stop["flag"]:
                break
            logger.info("next round in %d seconds", cfg.interval)
            # Sleep in 1s slices so SIGTERM/SIGINT stops promptly.
            for _ in range(cfg.interval):
                if stop["flag"]:
                    break
                time.sleep(1)
        logger.info("relay stopped")
        return 0
    finally:
        lock_handle.close()


if __name__ == "__main__":
    sys.exit(main())
