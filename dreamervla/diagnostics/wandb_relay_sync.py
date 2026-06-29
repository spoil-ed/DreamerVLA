#!/usr/bin/env python3
"""Relay W&B offline runs from a remote (air-gapped) GPU host to W&B cloud.

Standalone CPU-side helper. It NEVER touches the training process and treats the
remote ``wandb/`` directory as a read-only data source: each round it rsyncs the
remote runs into a local mirror, then runs ``wandb sync`` against the mirror.

The W&B API key must live only on this relay machine (run ``wandb login``); it is
never read, stored, or printed by this tool.
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
    remote_host: str
    remote_user: str | None
    remote_wandb_dir: str
    local_mirror_dir: Path
    wandb_project: str | None
    wandb_entity: str | None
    interval: int
    rsync_bin: str
    wandb_bin: str
    ssh_port: int
    dry_run: bool
    once: bool
    log_file: Path | None
    excludes: list[str]
    lock_file: Path


def parse_args(argv: list[str] | None = None) -> RelayConfig:
    p = argparse.ArgumentParser(
        prog="wandb_relay_sync.py",
        description=(
            "Periodically relay W&B offline runs from a remote GPU host to W&B "
            "cloud via rsync + `wandb sync`. The remote is treated as read-only."
        ),
    )
    p.add_argument("--remote-host", required=True,
                   help="Hostname / IP of the GPU training machine.")
    p.add_argument("--remote-user", default=None,
                   help="SSH user on the GPU machine (default: ssh config user).")
    p.add_argument("--remote-wandb-dir", required=True,
                   help="Remote dir that contains the offline-run-* directories.")
    p.add_argument("--local-mirror-dir", required=True,
                   help="Local dir to mirror runs into (never auto-cleaned).")
    p.add_argument("--wandb-project", default=None, help="W&B project to upload to.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity to upload to.")
    p.add_argument("--interval", type=int, default=60,
                   help="Seconds to wait between rounds (default: 60).")
    p.add_argument("--rsync-bin", default="rsync", help="Path to rsync binary.")
    p.add_argument("--wandb-bin", default="wandb", help="Path to wandb binary.")
    p.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print rsync/wandb commands only; do not execute anything.")
    p.add_argument("--once", action="store_true",
                   help="Run a single round and exit (exit code reflects success).")
    p.add_argument("--log-file", default=None,
                   help="Optional log file (always also logs to the terminal).")
    p.add_argument("--exclude", action="append", default=[], dest="excludes",
                   metavar="PATTERN", help="rsync --exclude pattern (repeatable).")
    p.add_argument("--lock-file", default=None,
                   help="Lock file path (default: <local-mirror-dir>/.wandb_relay.lock).")
    args = p.parse_args(argv)

    local_mirror_dir = Path(args.local_mirror_dir).expanduser()
    lock_file = (
        Path(args.lock_file).expanduser()
        if args.lock_file
        else local_mirror_dir / ".wandb_relay.lock"
    )
    log_file = Path(args.log_file).expanduser() if args.log_file else None
    return RelayConfig(
        remote_host=args.remote_host,
        remote_user=args.remote_user,
        remote_wandb_dir=args.remote_wandb_dir,
        local_mirror_dir=local_mirror_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        interval=args.interval,
        rsync_bin=args.rsync_bin,
        wandb_bin=args.wandb_bin,
        ssh_port=args.ssh_port,
        dry_run=args.dry_run,
        once=args.once,
        log_file=log_file,
        excludes=list(args.excludes),
        lock_file=lock_file,
    )


def build_remote_spec(remote_host: str, remote_user: str | None) -> str:
    if remote_user:
        return f"{remote_user}@{remote_host}"
    return remote_host


def _wandb_mirror_dir(cfg: RelayConfig) -> Path:
    # rsync the remote runs into a `wandb/` subdir so `wandb sync` finds them.
    return cfg.local_mirror_dir / "wandb"


def build_rsync_command(cfg: RelayConfig) -> list[str]:
    remote_spec = build_remote_spec(cfg.remote_host, cfg.remote_user)
    # Trailing slashes: copy the *contents* of the remote wandb dir into
    # <local-mirror>/wandb/, preserving the run directory structure.
    remote_src = f"{remote_spec}:{cfg.remote_wandb_dir.rstrip('/')}/"
    local_dst = f"{str(_wandb_mirror_dir(cfg)).rstrip('/')}/"
    cmd = [
        cfg.rsync_bin,
        "-a",          # archive: preserve tree, times, symlinks, perms
        "-z",          # compress during transfer
        "--partial",   # keep partially transferred files (resume-friendly)
        "--inplace",   # update files in place (good for append-only .wandb logs)
        "-e", f"ssh -p {cfg.ssh_port}",
    ]
    for pattern in cfg.excludes:
        cmd += ["--exclude", pattern]
    cmd += [remote_src, local_dst]
    return cmd


def build_wandb_sync_command(cfg: RelayConfig) -> list[str]:
    # No path argument: `wandb sync --sync-all` is run with cwd=local_mirror_dir
    # so the CLI discovers the `wandb/` subdir there. (Passing an explicit path
    # with --sync-all does not reliably discover runs across wandb versions.)
    cmd = [cfg.wandb_bin, "sync", "--sync-all", "--include-offline"]
    if cfg.wandb_project:
        cmd += ["-p", cfg.wandb_project]
    if cfg.wandb_entity:
        cmd += ["-e", cfg.wandb_entity]
    return cmd


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
    """Run a single relay round. Never raises for command failures."""
    logger.info("remote wandb dir : %s", cfg.remote_wandb_dir)
    logger.info("local mirror dir : %s", cfg.local_mirror_dir)
    if not cfg.dry_run:
        _wandb_mirror_dir(cfg).mkdir(parents=True, exist_ok=True)

    if not _run(build_rsync_command(cfg), logger, cfg.dry_run, "rsync"):
        logger.error("round failed during rsync; will retry next round")
        return False
    wandb_cwd = str(cfg.local_mirror_dir)
    if not _run(
        build_wandb_sync_command(cfg), logger, cfg.dry_run, "wandb sync", cwd=wandb_cwd
    ):
        logger.error("round failed during wandb sync; will retry next round")
        return False
    logger.info("round succeeded")
    return True


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    logger = setup_logging(cfg.log_file)

    if cfg.dry_run:
        logger.info("DRY-RUN: printing commands only; no rsync, no wandb sync, no lock")
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
