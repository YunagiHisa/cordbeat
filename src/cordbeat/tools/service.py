"""CordBeat service management — install / start / stop / status / uninstall.

Supports:
  Linux   : systemd user service (~/.config/systemd/user/cordbeat.service)
  macOS   : launchd user agent  (~/Library/LaunchAgents/com.cordbeat.agent.plist)
  Windows : Task Scheduler      (HKCU scheduled task "CordBeat")
"""

from __future__ import annotations

import shutil  # noqa: I001
import subprocess
import sys
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────


def _cordbeat_exe() -> str:
    """Absolute path to the 'cordbeat' executable in the active environment."""
    exe = shutil.which("cordbeat")
    if exe:
        return exe
    # Fallback: venv Scripts/bin next to current Python
    scripts = Path(sys.executable).parent
    for name in ("cordbeat", "cordbeat.exe"):
        candidate = scripts / name
        if candidate.exists():
            return str(candidate)
    return "cordbeat"  # last resort — let the shell resolve it


# ── Linux systemd ─────────────────────────────────────────────────────────────

_SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
_SYSTEMD_UNIT = _SYSTEMD_DIR / "cordbeat.service"

_SYSTEMD_TEMPLATE = """\
[Unit]
Description=CordBeat AI Agent
After=network.target

[Service]
Type=simple
ExecStart={exe}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def _systemd_install() -> None:
    _SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    _SYSTEMD_UNIT.write_text(_SYSTEMD_TEMPLATE.format(exe=_cordbeat_exe()))
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "cordbeat"], check=True)
    print("✅ CordBeat service installed and started (systemd user service).")
    print("   Logs: journalctl --user -u cordbeat -f")


def _systemd_start() -> None:
    subprocess.run(["systemctl", "--user", "start", "cordbeat"], check=True)
    print("▶  CordBeat started.")


def _systemd_stop() -> None:
    subprocess.run(["systemctl", "--user", "stop", "cordbeat"], check=True)
    print("⏹  CordBeat stopped.")


def _systemd_status() -> None:
    subprocess.run(["systemctl", "--user", "status", "cordbeat"])


def _systemd_uninstall() -> None:
    subprocess.run(["systemctl", "--user", "disable", "--now", "cordbeat"])
    if _SYSTEMD_UNIT.exists():
        _SYSTEMD_UNIT.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"])
    print("🗑  CordBeat service removed.")


# ── macOS launchd ─────────────────────────────────────────────────────────────

_LAUNCHD_DIR = Path.home() / "Library" / "LaunchAgents"
_LAUNCHD_PLIST = _LAUNCHD_DIR / "com.cordbeat.agent.plist"

_LAUNCHD_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.cordbeat.agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>{exe}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{home}/.cordbeat/cordbeat.log</string>
  <key>StandardErrorPath</key>
  <string>{home}/.cordbeat/cordbeat.log</string>
</dict>
</plist>
"""


def _launchd_install() -> None:
    _LAUNCHD_DIR.mkdir(parents=True, exist_ok=True)
    _LAUNCHD_PLIST.write_text(
        _LAUNCHD_TEMPLATE.format(exe=_cordbeat_exe(), home=Path.home())
    )
    subprocess.run(["launchctl", "load", str(_LAUNCHD_PLIST)], check=True)
    print("✅ CordBeat service installed (launchd user agent).")
    print(f"   Logs: tail -f {Path.home()}/.cordbeat/cordbeat.log")


def _launchd_start() -> None:
    subprocess.run(["launchctl", "start", "com.cordbeat.agent"], check=True)
    print("▶  CordBeat started.")


def _launchd_stop() -> None:
    subprocess.run(["launchctl", "stop", "com.cordbeat.agent"], check=True)
    print("⏹  CordBeat stopped.")


def _launchd_status() -> None:
    subprocess.run(["launchctl", "list", "com.cordbeat.agent"])


def _launchd_uninstall() -> None:
    subprocess.run(["launchctl", "unload", str(_LAUNCHD_PLIST)])
    if _LAUNCHD_PLIST.exists():
        _LAUNCHD_PLIST.unlink()
    print("🗑  CordBeat service removed.")


# ── Windows Task Scheduler ────────────────────────────────────────────────────

_TASK_NAME = "CordBeat"


def _windows_install() -> None:
    exe = _cordbeat_exe()
    # /SC ONLOGON = run every time the current user logs in
    subprocess.run(
        [
            "schtasks",
            "/Create",
            "/F",
            "/TN",
            _TASK_NAME,
            "/TR",
            exe,
            "/SC",
            "ONLOGON",
            "/RL",
            "LIMITED",
        ],
        check=True,
    )
    subprocess.run(["schtasks", "/Run", "/TN", _TASK_NAME], check=True)
    print("✅ CordBeat task registered and started (Task Scheduler).")


def _windows_start() -> None:
    subprocess.run(["schtasks", "/Run", "/TN", _TASK_NAME], check=True)
    print("▶  CordBeat started.")


def _windows_stop() -> None:
    subprocess.run(["schtasks", "/End", "/TN", _TASK_NAME], check=True)
    print("⏹  CordBeat stopped.")


def _windows_status() -> None:
    subprocess.run(["schtasks", "/Query", "/TN", _TASK_NAME, "/FO", "LIST"])


def _windows_uninstall() -> None:
    subprocess.run(["schtasks", "/Delete", "/TN", _TASK_NAME, "/F"])
    print("🗑  CordBeat task removed.")


# ── dispatch ──────────────────────────────────────────────────────────────────


def _platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


_ACTIONS: dict[str, dict[str, object]] = {
    "linux": {
        "install": _systemd_install,
        "start": _systemd_start,
        "stop": _systemd_stop,
        "status": _systemd_status,
        "uninstall": _systemd_uninstall,
    },
    "macos": {
        "install": _launchd_install,
        "start": _launchd_start,
        "stop": _launchd_stop,
        "status": _launchd_status,
        "uninstall": _launchd_uninstall,
    },
    "windows": {
        "install": _windows_install,
        "start": _windows_start,
        "stop": _windows_stop,
        "status": _windows_status,
        "uninstall": _windows_uninstall,
    },
}


def run_service_command(action: str) -> int:
    """Execute *action* (install/start/stop/status/uninstall) for this platform.

    Returns an exit code (0 = success).
    """
    platform = _platform()
    handlers = _ACTIONS[platform]

    if action not in handlers:
        valid = " | ".join(handlers)
        print(f"Unknown service action '{action}'.  Valid: {valid}", file=sys.stderr)
        return 1

    try:
        fn = handlers[action]
        assert callable(fn)
        fn()
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        return e.returncode or 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
