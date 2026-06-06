from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable

import numpy as np

from .cache import local_analysis_cache_dir


CACHE_SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 10 * 1024**3
HASH_CHUNK_SIZE = 4 * 1024**2


class CacheCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class FileFingerprint:
    path: Path
    digest: str
    size: int
    mtime: float


class AnalysisCacheSession:
    def __init__(
        self,
        root: Path | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.root = root or local_analysis_cache_dir()
        self.max_bytes = max(0, max_bytes)
        self.log = log
        self._fingerprints: dict[Path, FileFingerprint] = {}
        self._density_hits = 0
        self._mesh_hits = 0
        self._transaction_entries: list[Path] | None = None

    def begin_case(self) -> None:
        if self._transaction_entries is not None:
            raise RuntimeError("cache transaction is already active")
        self._transaction_entries = []

    def commit_case(self) -> None:
        self._transaction_entries = None

    def rollback_case(self) -> None:
        entries = self._transaction_entries or []
        self._transaction_entries = None
        for entry in reversed(entries):
            shutil.rmtree(entry, ignore_errors=True)

    def fingerprint(
        self,
        path: Path,
        stop_requested: Callable[[], bool] = lambda: False,
    ) -> FileFingerprint:
        resolved = path.resolve()
        cached = self._fingerprints.get(resolved)
        if cached is not None:
            return cached

        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while True:
                if stop_requested():
                    raise CacheCancelled("Cache hashing was stopped")
                chunk = handle.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        stat = resolved.stat()
        fingerprint = FileFingerprint(
            path=resolved,
            digest=digest.hexdigest(),
            size=stat.st_size,
            mtime=stat.st_mtime,
        )
        self._fingerprints[resolved] = fingerprint
        return fingerprint

    def fingerprints(
        self,
        paths: list[Path],
        stop_requested: Callable[[], bool] = lambda: False,
    ) -> dict[Path, FileFingerprint]:
        result: dict[Path, FileFingerprint] = {}
        for path in paths:
            if path.is_file():
                fingerprint = self.fingerprint(path, stop_requested)
                result[fingerprint.path] = fingerprint
        return result

    def key(self, payload: Any) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def load_density(self, key: str) -> list[float] | None:
        entry = self.root / "density" / key
        try:
            metadata = self._read_metadata(entry, "density")
            array = np.load(entry / "values.npy", allow_pickle=False)
            if array.dtype != np.float64 or array.ndim != 1:
                raise ValueError("invalid density array")
            if int(metadata.get("count", -1)) != len(array):
                raise ValueError("density count mismatch")
            self._touch(entry)
            self._density_hits += 1
            return array.tolist()
        except FileNotFoundError:
            return None
        except Exception as exc:
            self._discard_corrupt(entry, "density", exc)
            return None

    def store_density(self, key: str, values: list[float]) -> None:
        array = np.asarray(values, dtype=np.float64)
        self._store_entry(
            "density",
            key,
            {"kind": "density", "count": len(array)},
            {"values.npy": lambda path: np.save(path, array, allow_pickle=False)},
        )

    def load_mesh(self, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        loaded = self._load_array_entry("mesh", key)
        if loaded is not None:
            self._mesh_hits += 1
        return loaded

    def store_mesh(
        self,
        key: str,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        self._store_array_entry("mesh", key, metadata, arrays)

    def load_result(self, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        return self._load_array_entry("result", key)

    def store_result(
        self,
        key: str,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        self._store_array_entry("result", key, metadata, arrays)
        self.prune()

    def partial_hit_summary(self) -> str:
        return f"density={self._density_hits}, mesh={self._mesh_hits}"

    def hit_counts(self) -> tuple[int, int]:
        return self._density_hits, self._mesh_hits

    def invalidate(self, kind: str, key: str, reason: Exception | str) -> None:
        entry = self.root / kind / key
        self.log(f"Local analysis cache warning ({kind}); recalculating: {reason}")
        shutil.rmtree(entry, ignore_errors=True)

    def prune(self) -> None:
        if self.max_bytes <= 0 or not self.root.exists():
            return
        try:
            entries: list[tuple[float, Path, int]] = []
            total = 0
            for kind in ("density", "mesh", "result"):
                parent = self.root / kind
                if not parent.is_dir():
                    continue
                for entry in parent.iterdir():
                    if not entry.is_dir():
                        continue
                    size = sum(
                        item.stat().st_size
                        for item in entry.rglob("*")
                        if item.is_file()
                    )
                    total += size
                    entries.append((entry.stat().st_mtime, entry, size))
            for _, entry, size in sorted(entries):
                if total <= self.max_bytes:
                    break
                shutil.rmtree(entry, ignore_errors=True)
                total -= size
        except Exception as exc:
            self.log(f"Local analysis cache cleanup warning: {exc}")

    def _load_array_entry(
        self,
        kind: str,
        key: str,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        entry = self.root / kind / key
        try:
            metadata = self._read_metadata(entry, kind)
            arrays: dict[str, np.ndarray] = {}
            array_names = metadata.get("arrays")
            if not isinstance(array_names, list):
                raise ValueError("array list is missing")
            for name in array_names:
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("invalid array name")
                arrays[name] = np.load(entry / name, allow_pickle=False)
            self._touch(entry)
            return metadata, arrays
        except FileNotFoundError:
            return None
        except Exception as exc:
            self._discard_corrupt(entry, kind, exc)
            return None

    def _store_array_entry(
        self,
        kind: str,
        key: str,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        payload = dict(metadata)
        payload["kind"] = kind
        payload["arrays"] = sorted(arrays)
        writers = {
            name: (
                lambda path, value=np.asarray(array): np.save(
                    path,
                    value,
                    allow_pickle=False,
                )
            )
            for name, array in arrays.items()
        }
        self._store_entry(kind, key, payload, writers)

    def _store_entry(
        self,
        kind: str,
        key: str,
        metadata: dict[str, Any],
        writers: dict[str, Callable[[Path], None]],
    ) -> None:
        entry = self.root / kind / key
        if entry.is_dir():
            self._touch(entry)
            return
        parent = entry.parent
        temporary: Path | None = None
        try:
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=parent))
            for name, writer in writers.items():
                writer(temporary / name)
            payload = dict(metadata)
            payload["schema_version"] = CACHE_SCHEMA_VERSION
            payload["created_at"] = time.time()
            metadata_path = temporary / "metadata.json"
            with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, entry)
                temporary = None
                if self._transaction_entries is not None:
                    self._transaction_entries.append(entry)
            except OSError:
                if not entry.is_dir():
                    raise
        except Exception as exc:
            self.log(f"Local analysis cache write warning ({kind}): {exc}")
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)

    def _read_metadata(self, entry: Path, expected_kind: str) -> dict[str, Any]:
        payload = json.loads((entry / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("metadata root is not an object")
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("cache schema mismatch")
        if payload.get("kind") != expected_kind:
            raise ValueError("cache kind mismatch")
        return payload

    def _touch(self, entry: Path) -> None:
        try:
            os.utime(entry, None)
        except OSError:
            pass

    def _discard_corrupt(self, entry: Path, kind: str, exc: Exception) -> None:
        self.log(f"Local analysis cache warning ({kind}); recalculating: {exc}")
        shutil.rmtree(entry, ignore_errors=True)
