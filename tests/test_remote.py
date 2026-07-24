from __future__ import annotations

from pathlib import Path
import os
import shutil
import stat
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdfoam_analyzer.remote import (
    RemoteError,
    RemoteProfile,
    discover_remote_cases,
    discover_remote_fields_for_cases,
    sync_remote_case,
    sync_remote_lagrangian_time,
    validate_private_key_path,
)


class FakeAttr:
    def __init__(self, filename: str, mode: int, size: int = 0, mtime: int = 100) -> None:
        self.filename = filename
        self.st_mode = mode
        self.st_size = size
        self.st_mtime = mtime


class FakeSftp:
    def __init__(self, tree: dict[str, bytes | None]) -> None:
        self.tree = {self._norm(path): value for path, value in tree.items()}
        self.get_count = 0

    def listdir_attr(self, path: str) -> list[FakeAttr]:
        path = self._norm(path)
        prefix = path.rstrip("/") + "/"
        names: set[str] = set()
        for item in self.tree:
            if not item.startswith(prefix):
                continue
            rest = item[len(prefix):]
            if rest:
                names.add(rest.split("/", 1)[0])
        attrs = []
        for name in sorted(names):
            child = prefix + name
            attrs.append(self.stat(child, filename=name))
        return attrs

    def stat(self, path: str, filename: str | None = None) -> FakeAttr:
        path = self._norm(path)
        if path in self.tree and self.tree[path] is not None:
            content = self.tree[path] or b""
            return FakeAttr(filename or Path(path).name, stat.S_IFREG | 0o644, len(content), 100)
        prefix = path.rstrip("/") + "/"
        if any(item.startswith(prefix) for item in self.tree):
            return FakeAttr(filename or Path(path).name, stat.S_IFDIR | 0o755, 0, 100)
        raise OSError(path)

    def get(self, remote_path: str, local_path: str) -> None:
        remote_path = self._norm(remote_path)
        content = self.tree[remote_path]
        assert content is not None
        self.get_count += 1
        Path(local_path).write_bytes(content)

    def _norm(self, path: str) -> str:
        value = path.replace("\\", "/")
        if not value.startswith("/"):
            value = "/" + value
        return os.path.normpath(value).replace("\\", "/")


def sample_tree() -> dict[str, bytes | None]:
    return {
        "/parent/case001/main/constant/polyMesh/points": b"points",
        "/parent/case001/main/constant/polyMesh/faces": b"faces",
        "/parent/case001/main/constant/polyMesh/owner": b"owner",
        "/parent/case001/main/constant/polyMesh/neighbour": b"neighbour",
        "/parent/case001/main/constant/idList": b"idList",
        "/parent/case001/main/0/rhoM_water": b"rho0",
        "/parent/case001/main/1e-10/rhoM_water": b"rho1",
        "/parent/case001/main/1e-10/lagrangian/moleculeCloud/positions": b"positions",
        "/parent/case001/main/1e-10/lagrangian/moleculeCloud/id": b"id",
        "/parent/case001/main/1e-10/lagrangian/moleculeCloud/origId": b"origId",
        "/parent/case001/main/1e-10/lagrangian/moleculeCloud/origProcId": b"origProcId",
        "/parent/case001/main/notes/readme.txt": b"ignore",
        "/parent/case002/main/processor0/constant/polyMesh/points": b"p",
        "/parent/case002/main/processor0/constant/polyMesh/faces": b"f",
        "/parent/case002/main/processor0/constant/polyMesh/owner": b"o",
        "/parent/case002/main/processor0/0/rhoN_water": b"pn0",
    }


class RemoteDiscoveryTests(unittest.TestCase):
    def test_discovers_multi_case_parent_and_fields(self) -> None:
        sftp = FakeSftp(sample_tree())
        cases = discover_remote_cases(sftp, "/parent")
        self.assertEqual(cases, ["/parent/case001", "/parent/case002"])
        fields = discover_remote_fields_for_cases(sftp, cases)
        self.assertEqual(fields, ["rhoM_water", "rhoN_water"])

    def test_discovers_single_case_folder(self) -> None:
        sftp = FakeSftp(sample_tree())
        self.assertEqual(discover_remote_cases(sftp, "/parent/case001"), ["/parent/case001"])

    def test_rejects_putty_ppk(self) -> None:
        with self.assertRaises(RemoteError):
            validate_private_key_path(r"C:\keys\id_rsa.ppk")


class RemoteSyncTests(unittest.TestCase):
    def test_sync_downloads_required_files_and_reuses_cache(self) -> None:
        sftp = FakeSftp(sample_tree())
        profile = RemoteProfile("test", "host", 22, "user", "", "", "/parent")
        old_local_appdata = os.environ.get("LOCALAPPDATA")
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["LOCALAPPDATA"] = temp_dir
            try:
                local_case = sync_remote_case(sftp, profile, "/parent/case001", "rhoM_water")
                self.assertTrue((local_case / "main" / "0" / "rhoM_water").is_file())
                self.assertTrue((local_case / "main" / "1e-10" / "rhoM_water").is_file())
                first_count = sftp.get_count

                sync_remote_case(sftp, profile, "/parent/case001", "rhoM_water")
                self.assertEqual(sftp.get_count, first_count)

                stale = local_case / "main" / "old" / "rhoM_water"
                stale.parent.mkdir(parents=True)
                stale.write_text("stale", encoding="utf-8")
                sync_remote_case(sftp, profile, "/parent/case001", "rhoM_water")
                self.assertFalse(stale.exists())
            finally:
                if old_local_appdata is None:
                    os.environ.pop("LOCALAPPDATA", None)
                else:
                    os.environ["LOCALAPPDATA"] = old_local_appdata
                shutil.rmtree(Path(temp_dir), ignore_errors=True)

    def test_sync_lagrangian_time_downloads_positions_and_id(self) -> None:
        sftp = FakeSftp(sample_tree())
        profile = RemoteProfile("test", "host", 22, "user", "", "", "/parent")
        with tempfile.TemporaryDirectory() as temp_dir:
            local_case = Path(temp_dir) / "case001"
            sync_remote_lagrangian_time(
                sftp,
                profile,
                "/parent/case001",
                local_case,
                "1e-10",
            )
            self.assertEqual(
                (local_case / "main" / "1e-10" / "lagrangian" / "moleculeCloud" / "positions").read_bytes(),
                b"positions",
            )
            self.assertEqual(
                (local_case / "main" / "1e-10" / "lagrangian" / "moleculeCloud" / "id").read_bytes(),
                b"id",
            )
            self.assertEqual(
                (local_case / "main" / "1e-10" / "lagrangian" / "moleculeCloud" / "origId").read_bytes(),
                b"origId",
            )
            self.assertEqual(
                (local_case / "main" / "constant" / "idList").read_bytes(),
                b"idList",
            )

    def test_sync_case_can_include_all_departure_inputs(self) -> None:
        sftp = FakeSftp(sample_tree())
        profile = RemoteProfile("test", "host", 22, "user", "", "", "/parent")
        old_local_appdata = os.environ.get("LOCALAPPDATA")
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["LOCALAPPDATA"] = temp_dir
            try:
                local_case = sync_remote_case(
                    sftp,
                    profile,
                    "/parent/case001",
                    "rhoM_water",
                    include_lagrangian=True,
                )
                cloud = (
                    local_case
                    / "main"
                    / "1e-10"
                    / "lagrangian"
                    / "moleculeCloud"
                )
                for name in ("positions", "id", "origId", "origProcId"):
                    self.assertTrue((cloud / name).is_file())
                self.assertTrue(
                    (local_case / "main" / "constant" / "idList").is_file()
                )
            finally:
                if old_local_appdata is None:
                    os.environ.pop("LOCALAPPDATA", None)
                else:
                    os.environ["LOCALAPPDATA"] = old_local_appdata
                shutil.rmtree(Path(temp_dir), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
