from __future__ import annotations

from pathlib import Path
import hashlib
import os
import shutil


APP_NAME = "mdfoam-density-analyzer"


def user_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / APP_NAME / "remote-cache"
    return Path.home() / ".cache" / APP_NAME / "remote-cache"


def app_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / APP_NAME
    return Path.home() / ".cache" / APP_NAME


def remote_cache_dir(host: str, username: str, remote_path: str) -> Path:
    key = hashlib.sha256(f"{host}|{username}|{remote_path}".encode("utf-8")).hexdigest()[:16]
    safe_host = "".join(char if char.isalnum() or char in ".-_" else "_" for char in host)
    safe_user = "".join(char if char.isalnum() or char in ".-_" else "_" for char in username)
    return user_cache_dir() / f"{safe_user}@{safe_host}-{key}"


def local_analysis_cache_dir() -> Path:
    return app_cache_dir() / "local-analysis-cache"


def clear_cache(path: Path | None = None) -> None:
    target = path or user_cache_dir()
    if target.exists():
        shutil.rmtree(target)


def clear_local_analysis_cache() -> None:
    clear_cache(local_analysis_cache_dir())


def is_cached_file_current(local_path: Path, size: int | None, mtime: int | None) -> bool:
    if not local_path.is_file():
        return False
    stat = local_path.stat()
    if size is not None and stat.st_size != size:
        return False
    if mtime is not None and abs(int(stat.st_mtime) - int(mtime)) > 1:
        return False
    return True
