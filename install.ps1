#!/usr/bin/env pwsh
# CordBeat installer for Windows
# Usage: irm https://raw.githubusercontent.com/YunagiHisa/cordbeat/main/install.ps1 | iex

param(
    [string]$InstallDir = "$HOME\cordbeat"
)

$ErrorActionPreference = "Stop"
$repo = "https://github.com/YunagiHisa/cordbeat.git"

# Ctrl+C / pipeline interruption → clean exit
trap {
    Write-Host ""
    Write-Host "  Installation interrupted." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "   ██████╗ ██████╗ ██████╗ ██████╗ ██████╗ ███████╗ █████╗ ████████╗" -ForegroundColor Magenta
Write-Host "  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝" -ForegroundColor DarkMagenta
Write-Host "  ██║     ██║   ██║██████╔╝██║  ██║██████╔╝█████╗  ███████║   ██║   " -ForegroundColor Red
Write-Host "  ██║     ██║   ██║██╔══██╗██║  ██║██╔══██╗██╔══╝  ██╔══██║   ██║   " -ForegroundColor Magenta
Write-Host "  ╚██████╗╚██████╔╝██║  ██║██████╔╝██████╔╝███████╗██║  ██║   ██║   " -ForegroundColor DarkMagenta
Write-Host "   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   " -ForegroundColor Red
Write-Host ""
Write-Host "  A local-first autonomous AI agent that stays by your side." -ForegroundColor White
Write-Host ""

# ── Check git ────────────────────────────────────────────────────────────────
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "✗ git is required. Please install Git from https://git-scm.com/" -ForegroundColor Red
    exit 1
}

# ── Install / update uv ──────────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  Installing uv (Python package manager)..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
    # Reload PATH so uv is available in this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") +
                ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "  Please restart your terminal and re-run the installer." -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "  ✓ uv $(uv --version)" -ForegroundColor Green

# ── Clone or update repo ─────────────────────────────────────────────────────
if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-Host "  Updating existing installation at $InstallDir ..." -ForegroundColor Yellow
    git -C $InstallDir pull --ff-only
} else {
    Write-Host "  Cloning CordBeat to $InstallDir ..." -ForegroundColor Yellow
    git clone $repo $InstallDir
}
Write-Host "  ✓ Source ready" -ForegroundColor Green

Set-Location $InstallDir

# ── Install Python dependencies ───────────────────────────────────────────────
Write-Host "  Installing dependencies (this may take a minute)..." -ForegroundColor Yellow
uv sync --quiet
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# ── Run setup wizard (skips if config already exists) ────────────────────────
Write-Host ""
uv run cordbeat-init

Write-Host ""
Write-Host "  ✨ CordBeat is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "  To chat again:" -ForegroundColor White
Write-Host "    cd $InstallDir" -ForegroundColor Cyan
Write-Host "    uv run cordbeat-chat" -ForegroundColor Cyan
Write-Host ""
Write-Host "  To run as headless server (Discord / Telegram):" -ForegroundColor White
Write-Host "    uv run cordbeat" -ForegroundColor Cyan
Write-Host ""
Write-Host "  To update later:" -ForegroundColor White
Write-Host "    irm https://raw.githubusercontent.com/YunagiHisa/cordbeat/main/install.ps1 | iex" -ForegroundColor Cyan
Write-Host ""
