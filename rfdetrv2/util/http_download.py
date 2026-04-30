"""Shared HTTP download helpers for pretrained weight utilities."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


@contextmanager
def exclusive_download_lock(lock_path: Path):
    """Serialize downloads of the same file (e.g. ``torchrun`` / multi-process)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        yield
        return
    import fcntl

    with open(lock_path, "a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 1024 * 1024,
    *,
    timeout: int = 120,
    user_agent: str = "RF-DETR/download",
) -> None:
    """Stream-download *url* into *dest*, writing atomically via a ``.part`` temp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  {dest.name}: {pct}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
            if total:
                print(file=sys.stderr)
        tmp.replace(dest)
    except (OSError, URLError):
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
