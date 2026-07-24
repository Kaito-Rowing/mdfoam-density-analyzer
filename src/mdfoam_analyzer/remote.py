from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
import os
import posixpath
import stat

from .cache import is_cached_file_current, remote_cache_dir


MESH_FILES = ("points", "faces", "owner", "neighbour")
LAGRANGIAN_FILES = ("positions", "id", "origId", "origProcId")


class RemoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteProfile:
    name: str
    host: str
    port: int
    username: str
    key_path: str
    secret: str
    remote_path: str


@dataclass(frozen=True)
class RemoteFile:
    remote_path: str
    relative_path: str
    size: int | None
    mtime: int | None


class SshConnection:
    def __init__(self, profile: RemoteProfile) -> None:
        self.profile = profile
        self.client = None
        self.sftp = None

    def __enter__(self) -> "SshConnection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def connect(self) -> None:
        validate_private_key_path(self.profile.key_path)
        try:
            import paramiko
        except ImportError as exc:
            raise RemoteError(
                "SSH機能には paramiko が必要です。python -m pip install -r requirements.txt を実行してください。"
            ) from exc

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        key_path = self.profile.key_path.strip() or None
        secret = self.profile.secret or None
        try:
            client.connect(
                hostname=self.profile.host,
                port=self.profile.port,
                username=self.profile.username,
                key_filename=key_path,
                passphrase=secret if key_path else None,
                password=None if key_path else secret,
                look_for_keys=not bool(key_path),
                allow_agent=True,
                timeout=20,
            )
            self.client = client
            self.sftp = client.open_sftp()
        except Exception as exc:
            client.close()
            raise RemoteError(f"SSH接続に失敗しました: {exc}") from exc

    def close(self) -> None:
        if self.sftp is not None:
            self.sftp.close()
            self.sftp = None
        if self.client is not None:
            self.client.close()
            self.client = None


def validate_private_key_path(path: str) -> None:
    key_path = path.strip()
    if key_path.lower().endswith(".ppk"):
        raise RemoteError(
            ".ppk は直接読み込めません。PuTTYgenでOpenSSH形式の秘密鍵に変換してから指定してください。"
        )


def normalize_remote_path(path: str) -> str:
    value = path.strip().replace("\\", "/")
    if not value:
        return "."
    return posixpath.normpath(value)


def remote_name(path: str) -> str:
    return PurePosixPath(path).name or path.strip("/").split("/")[-1] or "remote-case"


def remote_join(*parts: str) -> str:
    return posixpath.normpath(posixpath.join(*(part for part in parts if part != "")))


def is_remote_dir(sftp, path: str) -> bool:
    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except OSError:
        return False


def is_remote_file(sftp, path: str) -> bool:
    try:
        return stat.S_ISREG(sftp.stat(path).st_mode)
    except OSError:
        return False


def list_remote_dirs(sftp, path: str) -> list[tuple[str, str]]:
    dirs: list[tuple[str, str]] = []
    for entry in sftp.listdir_attr(path):
        if entry.filename in (".", ".."):
            continue
        if stat.S_ISDIR(entry.st_mode):
            full_path = remote_join(path, entry.filename)
            dirs.append((entry.filename, full_path))
    return sorted(dirs, key=lambda item: item[0])


def discover_remote_cases(sftp, parent: str) -> list[str]:
    parent = normalize_remote_path(parent)
    if is_remote_dir(sftp, remote_join(parent, "main")):
        return [parent]

    cases = [
        child_path
        for _, child_path in list_remote_dirs(sftp, parent)
        if is_remote_dir(sftp, remote_join(child_path, "main"))
    ]
    return sorted(cases, key=remote_name)


def discover_remote_fields_for_cases(sftp, cases: list[str]) -> list[str]:
    names: set[str] = set()
    for case in cases:
        names.update(discover_remote_density_fields(sftp, remote_join(case, "main")))
    if "rhoM_water" in names:
        return ["rhoM_water"] + sorted(name for name in names if name != "rhoM_water")
    return sorted(names)


def discover_remote_density_fields(sftp, main_dir: str) -> list[str]:
    names: set[str] = set()
    for _, time_dir in remote_numeric_time_dirs(sftp, main_dir):
        for field_name in _remote_file_names(sftp, time_dir):
            if field_name.startswith(("rhoM", "rhoN")):
                names.add(field_name)
        if names:
            break

    if not names:
        for _, processor in list_remote_dirs(sftp, main_dir):
            if not remote_name(processor).startswith("processor"):
                continue
            for _, time_dir in remote_numeric_time_dirs(sftp, processor):
                for field_name in _remote_file_names(sftp, time_dir):
                    if field_name.startswith(("rhoM", "rhoN")):
                        names.add(field_name)
                if names:
                    break
            if names:
                break
    return sorted(names)


def remote_numeric_time_dirs(sftp, root: str) -> list[tuple[float, str]]:
    if not is_remote_dir(sftp, root):
        return []
    result: list[tuple[float, str]] = []
    for name, path in list_remote_dirs(sftp, root):
        try:
            time_value = float(name)
        except ValueError:
            continue
        result.append((time_value, path))
    return sorted(result, key=lambda item: item[0])


def sync_remote_case(
    sftp,
    profile: RemoteProfile,
    remote_case: str,
    density_field: str,
    include_lagrangian: bool = False,
    stop_requested=lambda: False,
    log=lambda message: None,
) -> Path:
    remote_case = normalize_remote_path(remote_case)
    local_case = remote_cache_dir(profile.host, profile.username, remote_case) / remote_name(remote_case)
    required_files = _required_remote_files(
        sftp,
        remote_case,
        density_field,
        include_lagrangian=include_lagrangian,
    )
    if not required_files:
        raise RemoteError(f"{remote_case}: 読み取り可能な {density_field} が見つかりません。")

    required_relatives = {item.relative_path for item in required_files}
    _remove_stale_files(local_case, required_relatives)

    downloaded = 0
    reused = 0
    for item in required_files:
        if stop_requested():
            break
        local_path = local_case / Path(*PurePosixPath(item.relative_path).parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if is_cached_file_current(local_path, item.size, item.mtime):
            reused += 1
            continue
        sftp.get(item.remote_path, str(local_path))
        if item.mtime is not None:
            os.utime(local_path, (item.mtime, item.mtime))
        downloaded += 1

    _write_manifest(local_case, profile, remote_case, required_files)
    log(f"{remote_name(remote_case)}: SFTP同期 {downloaded} ファイル取得, {reused} ファイル再利用")
    return local_case


def sync_remote_lagrangian_time(
    sftp,
    profile: RemoteProfile,
    remote_case: str,
    local_case: Path,
    time_name: str,
    stop_requested=lambda: False,
    log=lambda message: None,
) -> None:
    remote_case = normalize_remote_path(remote_case)
    files: list[RemoteFile] = []
    for file_name in LAGRANGIAN_FILES:
        remote_path = remote_join(
            remote_case,
            "main",
            time_name,
            "lagrangian",
            "moleculeCloud",
            file_name,
        )
        if is_remote_file(sftp, remote_path):
            files.append(
                _remote_file(
                    sftp,
                    remote_path,
                    remote_join("main", time_name, "lagrangian", "moleculeCloud", file_name),
                )
            )

    id_list_remote = remote_join(remote_case, "main", "constant", "idList")
    if is_remote_file(sftp, id_list_remote):
        files.append(
            _remote_file(
                sftp,
                id_list_remote,
                remote_join("main", "constant", "idList"),
            )
        )

    downloaded = 0
    reused = 0
    for item in files:
        if stop_requested():
            break
        local_path = local_case / Path(*PurePosixPath(item.relative_path).parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if is_cached_file_current(local_path, item.size, item.mtime):
            reused += 1
            continue
        sftp.get(item.remote_path, str(local_path))
        if item.mtime is not None:
            os.utime(local_path, (item.mtime, item.mtime))
        downloaded += 1
    if files:
        log(f"{remote_name(remote_case)} {time_name}: Lagrangian {downloaded} ファイル取得, {reused} ファイル再利用")


def _required_remote_files(
    sftp,
    remote_case: str,
    density_field: str,
    include_lagrangian: bool = False,
) -> list[RemoteFile]:
    main_dir = remote_join(remote_case, "main")
    reconstructed_times = [
        time_dir
        for _, time_dir in remote_numeric_time_dirs(sftp, main_dir)
        if is_remote_file(sftp, remote_join(time_dir, density_field))
    ]
    if reconstructed_times:
        files = _mesh_files(sftp, main_dir, "main")
        if include_lagrangian:
            files.extend(_constant_molecule_files(sftp, main_dir, "main"))
        for time_dir in reconstructed_times:
            time_name = remote_name(time_dir)
            files.append(
                _remote_file(
                    sftp,
                    remote_join(time_dir, density_field),
                    remote_join("main", time_name, density_field),
                )
            )
            if include_lagrangian:
                files.extend(
                    _lagrangian_files(
                        sftp,
                        time_dir,
                        remote_join("main", time_name),
                    )
                )
        return files

    files: list[RemoteFile] = []
    if include_lagrangian:
        files.extend(_constant_molecule_files(sftp, main_dir, "main"))
    for processor_name, processor in list_remote_dirs(sftp, main_dir):
        if not processor_name.startswith("processor"):
            continue
        processor_times = [
            time_dir
            for _, time_dir in remote_numeric_time_dirs(sftp, processor)
            if is_remote_file(sftp, remote_join(time_dir, density_field))
        ]
        if not processor_times:
            continue
        files.extend(_mesh_files(sftp, processor, remote_join("main", processor_name)))
        for time_dir in processor_times:
            time_name = remote_name(time_dir)
            files.append(
                _remote_file(
                    sftp,
                    remote_join(time_dir, density_field),
                    remote_join("main", processor_name, time_name, density_field),
                )
            )
            if include_lagrangian:
                files.extend(
                    _lagrangian_files(
                        sftp,
                        time_dir,
                        remote_join("main", processor_name, time_name),
                    )
                )
    return files


def _constant_molecule_files(
    sftp,
    remote_parent: str,
    local_parent: str,
) -> list[RemoteFile]:
    remote_path = remote_join(remote_parent, "constant", "idList")
    if not is_remote_file(sftp, remote_path):
        return []
    return [
        _remote_file(
            sftp,
            remote_path,
            remote_join(local_parent, "constant", "idList"),
        )
    ]


def _lagrangian_files(
    sftp,
    remote_time_dir: str,
    local_time_dir: str,
) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for file_name in LAGRANGIAN_FILES:
        remote_path = remote_join(
            remote_time_dir,
            "lagrangian",
            "moleculeCloud",
            file_name,
        )
        if is_remote_file(sftp, remote_path):
            files.append(
                _remote_file(
                    sftp,
                    remote_path,
                    remote_join(
                        local_time_dir,
                        "lagrangian",
                        "moleculeCloud",
                        file_name,
                    ),
                )
            )
    return files


def _mesh_files(sftp, mesh_parent: str, local_parent: str) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for file_name in MESH_FILES:
        remote_path = remote_join(mesh_parent, "constant", "polyMesh", file_name)
        if is_remote_file(sftp, remote_path):
            files.append(
                _remote_file(
                    sftp,
                    remote_path,
                    remote_join(local_parent, "constant", "polyMesh", file_name),
                )
            )
    return files


def _remote_file(sftp, remote_path: str, relative_path: str) -> RemoteFile:
    attr = sftp.stat(remote_path)
    return RemoteFile(
        remote_path=remote_path,
        relative_path=relative_path,
        size=getattr(attr, "st_size", None),
        mtime=getattr(attr, "st_mtime", None),
    )


def _remote_file_names(sftp, path: str) -> list[str]:
    names: list[str] = []
    try:
        entries = sftp.listdir_attr(path)
    except OSError:
        return names
    for entry in entries:
        if stat.S_ISREG(entry.st_mode):
            names.append(entry.filename)
    return sorted(names)


def _remove_stale_files(local_case: Path, required_relatives: set[str]) -> None:
    if not local_case.exists():
        return
    required_paths = {Path(*PurePosixPath(relative).parts) for relative in required_relatives}
    for path in sorted(local_case.rglob("*"), reverse=True):
        if path.is_file():
            relative = path.relative_to(local_case)
            if path.name == ".mdfoam_remote_manifest.json":
                continue
            if relative not in required_paths:
                path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _write_manifest(
    local_case: Path,
    profile: RemoteProfile,
    remote_case: str,
    required_files: list[RemoteFile],
) -> None:
    local_case.mkdir(parents=True, exist_ok=True)
    manifest = {
        "host": profile.host,
        "username": profile.username,
        "remote_case": remote_case,
        "files": [
            {
                "remote_path": item.remote_path,
                "relative_path": item.relative_path,
                "size": item.size,
                "mtime": item.mtime,
            }
            for item in required_files
        ],
    }
    (local_case / ".mdfoam_remote_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
