"""Unit tests for cordbeat.tools.service — service management module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cordbeat.tools.service import (
    _adapter_exe,
    _cordbeat_exe,
    _platform,
    run_service_command,
)

# ── helpers ────────────────────────────────────────────────────────────────────


class TestCordBeatExe:
    def test_returns_which_result_when_found(self) -> None:
        with patch(
            "cordbeat.tools.service.shutil.which", return_value="/usr/bin/cordbeat"
        ):
            assert _cordbeat_exe() == "/usr/bin/cordbeat"

    def test_falls_back_to_scripts_dir_when_which_fails(self, tmp_path: Path) -> None:
        fake_exe = tmp_path / "cordbeat"
        fake_exe.touch()
        with (
            patch("cordbeat.tools.service.shutil.which", return_value=None),
            patch("cordbeat.tools.service.sys.executable", str(tmp_path / "python")),
        ):
            assert _cordbeat_exe() == str(fake_exe)

    def test_falls_back_to_string_when_nothing_found(self, tmp_path: Path) -> None:
        with (
            patch("cordbeat.tools.service.shutil.which", return_value=None),
            patch("cordbeat.tools.service.sys.executable", str(tmp_path / "python")),
        ):
            assert _cordbeat_exe() == "cordbeat"


class TestAdapterExe:
    def test_returns_which_result_when_found(self) -> None:
        with patch(
            "cordbeat.tools.service.shutil.which",
            return_value="/usr/bin/cordbeat-discord",
        ):
            assert _adapter_exe("discord") == "/usr/bin/cordbeat-discord"

    def test_falls_back_to_scripts_dir(self, tmp_path: Path) -> None:
        fake = tmp_path / "cordbeat-telegram"
        fake.touch()
        with (
            patch("cordbeat.tools.service.shutil.which", return_value=None),
            patch("cordbeat.tools.service.sys.executable", str(tmp_path / "python")),
        ):
            assert _adapter_exe("telegram") == str(fake)

    def test_falls_back_to_name(self, tmp_path: Path) -> None:
        with (
            patch("cordbeat.tools.service.shutil.which", return_value=None),
            patch("cordbeat.tools.service.sys.executable", str(tmp_path / "python")),
        ):
            assert _adapter_exe("slack") == "cordbeat-slack"


class TestPlatform:
    def test_darwin(self) -> None:
        with patch.object(sys, "platform", "darwin"):
            assert _platform() == "macos"

    def test_win32(self) -> None:
        with patch.object(sys, "platform", "win32"):
            assert _platform() == "windows"

    def test_linux(self) -> None:
        with patch.object(sys, "platform", "linux"):
            assert _platform() == "linux"


# ── run_service_command ────────────────────────────────────────────────────────


class TestRunServiceCommandLinux:
    @pytest.fixture(autouse=True)
    def _linux(self) -> None:
        with patch("cordbeat.tools.service._platform", return_value="linux"):
            yield

    def test_install_runs_systemctl(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        unit_dir = tmp_path / ".config" / "systemd" / "user"
        unit_file = unit_dir / "cordbeat.service"
        with (
            patch("cordbeat.tools.service._SYSTEMD_DIR", unit_dir),
            patch("cordbeat.tools.service._SYSTEMD_UNIT", unit_file),
            patch("cordbeat.tools.service._cordbeat_exe", return_value="/bin/cordbeat"),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("install")
        assert code == 0
        assert unit_file.exists()
        assert "/bin/cordbeat" in unit_file.read_text()
        assert mock_run.call_count == 3  # daemon-reload + enable --now + restart
        out = capsys.readouterr().out
        assert "loginctl enable-linger $USER" in out

    def test_start(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("cordbeat.tools.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("start")
        assert code == 0
        mock_run.assert_called_once_with(
            ["systemctl", "--user", "start", "cordbeat"], check=True
        )

    def test_stop(self) -> None:
        with patch("cordbeat.tools.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("stop")
        assert code == 0
        mock_run.assert_called_once_with(
            ["systemctl", "--user", "stop", "cordbeat"], check=True
        )

    def test_uninstall_removes_unit_file(self, tmp_path: Path) -> None:
        unit_dir = tmp_path / ".config" / "systemd" / "user"
        unit_dir.mkdir(parents=True)
        unit_file = unit_dir / "cordbeat.service"
        unit_file.write_text("[Service]\n")
        with (
            patch("cordbeat.tools.service._SYSTEMD_UNIT", unit_file),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("uninstall")
        assert code == 0
        assert not unit_file.exists()

    def test_install_with_adapters_writes_adapter_units(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        unit_dir = tmp_path / ".config" / "systemd" / "user"
        unit_file = unit_dir / "cordbeat.service"
        with (
            patch("cordbeat.tools.service._SYSTEMD_DIR", unit_dir),
            patch("cordbeat.tools.service._SYSTEMD_UNIT", unit_file),
            patch("cordbeat.tools.service._cordbeat_exe", return_value="/bin/cordbeat"),
            patch(
                "cordbeat.tools.service._adapter_exe",
                return_value="/bin/cordbeat-discord",
            ),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("install", ["discord"])
        assert code == 0
        adapter_unit = unit_dir / "cordbeat-discord.service"
        assert adapter_unit.exists()
        assert "/bin/cordbeat-discord" in adapter_unit.read_text()
        assert "cordbeat-discord adapter service installed" in capsys.readouterr().out

    def test_status_invokes_systemctl_status(self) -> None:
        with patch("cordbeat.tools.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("status")
        assert code == 0
        mock_run.assert_called_once_with(["systemctl", "--user", "status", "cordbeat"])

    def test_uninstall_removes_adapter_units(self, tmp_path: Path) -> None:
        unit_dir = tmp_path / ".config" / "systemd" / "user"
        unit_dir.mkdir(parents=True)
        unit_file = unit_dir / "cordbeat.service"
        unit_file.write_text("[Service]\n")
        adapter_unit = unit_dir / "cordbeat-discord.service"
        adapter_unit.write_text("[Service]\n")
        with (
            patch("cordbeat.tools.service._SYSTEMD_DIR", unit_dir),
            patch("cordbeat.tools.service._SYSTEMD_UNIT", unit_file),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("uninstall")
        assert code == 0
        assert not adapter_unit.exists()
        assert not unit_file.exists()

    def test_unknown_action_returns_1(self) -> None:
        code = run_service_command("bogus")
        assert code == 1

    def test_subprocess_error_returns_nonzero(self) -> None:
        import subprocess

        with patch("cordbeat.tools.service.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "systemctl")
            code = run_service_command("start")
        assert code != 0

    def test_generic_exception_returns_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A non-CalledProcessError (e.g. missing binary) hits the catch-all.
        with patch("cordbeat.tools.service.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("systemctl missing")
            code = run_service_command("start")
        assert code == 1
        assert "Error:" in capsys.readouterr().err


class TestRunServiceCommandMacOS:
    @pytest.fixture(autouse=True)
    def _macos(self) -> None:
        with patch("cordbeat.tools.service._platform", return_value="macos"):
            yield

    def test_install_writes_plist(self, tmp_path: Path) -> None:
        plist_dir = tmp_path / "Library" / "LaunchAgents"
        plist_file = plist_dir / "com.cordbeat.agent.plist"
        with (
            patch("cordbeat.tools.service._LAUNCHD_DIR", plist_dir),
            patch("cordbeat.tools.service._LAUNCHD_PLIST", plist_file),
            patch(
                "cordbeat.tools.service._cordbeat_exe",
                return_value="/usr/local/bin/cordbeat",
            ),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("install")
        assert code == 0
        assert plist_file.exists()
        assert "/usr/local/bin/cordbeat" in plist_file.read_text()

    def test_install_with_adapters_writes_adapter_plist(self, tmp_path: Path) -> None:
        plist_dir = tmp_path / "Library" / "LaunchAgents"
        plist_file = plist_dir / "com.cordbeat.agent.plist"
        with (
            patch("cordbeat.tools.service._LAUNCHD_DIR", plist_dir),
            patch("cordbeat.tools.service._LAUNCHD_PLIST", plist_file),
            patch("cordbeat.tools.service._cordbeat_exe", return_value="/bin/cordbeat"),
            patch(
                "cordbeat.tools.service._adapter_exe",
                return_value="/bin/cordbeat-telegram",
            ),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("install", ["telegram"])
        assert code == 0
        adapter_plist = plist_dir / "com.cordbeat.telegram.plist"
        assert adapter_plist.exists()
        assert "/bin/cordbeat-telegram" in adapter_plist.read_text()

    def test_start_stop_status(self) -> None:
        with patch("cordbeat.tools.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert run_service_command("start") == 0
            assert run_service_command("stop") == 0
            assert run_service_command("status") == 0
        actions = [c.args[0][1] for c in mock_run.call_args_list]
        assert actions == ["start", "stop", "list"]

    def test_uninstall_removes_plist(self, tmp_path: Path) -> None:
        plist_dir = tmp_path / "Library" / "LaunchAgents"
        plist_dir.mkdir(parents=True)
        plist_file = plist_dir / "com.cordbeat.agent.plist"
        plist_file.write_text("<plist/>")
        adapter_plist = plist_dir / "com.cordbeat.discord.plist"
        adapter_plist.write_text("<plist/>")
        with (
            patch("cordbeat.tools.service._LAUNCHD_DIR", plist_dir),
            patch("cordbeat.tools.service._LAUNCHD_PLIST", plist_file),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("uninstall")
        assert code == 0
        assert not plist_file.exists()
        assert not adapter_plist.exists()


class TestRunServiceCommandWindows:
    @pytest.fixture(autouse=True)
    def _windows(self) -> None:
        with patch("cordbeat.tools.service._platform", return_value="windows"):
            yield

    def test_install_calls_schtasks(self, capsys: pytest.CaptureFixture[str]) -> None:
        fake_exe = r"C:\bin\cordbeat.exe"
        with (
            patch("cordbeat.tools.service._cordbeat_exe", return_value=fake_exe),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("install")
        assert code == 0
        assert mock_run.call_count == 2  # /Create + /Run
        out = capsys.readouterr().out
        assert "ONLOGON task" in out
        assert "Run whether user is logged on or not" in out

    def test_install_with_adapters_registers_adapter_task(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch(
                "cordbeat.tools.service._cordbeat_exe", return_value=r"C:\bin\cb.exe"
            ),
            patch(
                "cordbeat.tools.service._adapter_exe",
                return_value=r"C:\bin\cb-discord.exe",
            ),
            patch("cordbeat.tools.service.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("install", ["discord"])
        assert code == 0
        # core /Create + /Run, then adapter /Create + /Run
        assert mock_run.call_count == 4
        out = capsys.readouterr().out
        assert "cordbeat-discord adapter task registered" in out

    def test_start_stop_status(self) -> None:
        with patch("cordbeat.tools.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert run_service_command("start") == 0
            assert run_service_command("stop") == 0
            assert run_service_command("status") == 0
        verbs = [c.args[0][1] for c in mock_run.call_args_list]
        assert verbs == ["/Run", "/End", "/Query"]

    def test_uninstall_calls_schtasks_delete(self) -> None:
        with patch("cordbeat.tools.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            code = run_service_command("uninstall")
        assert code == 0
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("Delete" in c for c in calls)
