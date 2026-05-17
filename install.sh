#!/usr/bin/env bash
# CordBeat installer for Linux / macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/YunagiHisa/cordbeat/main/install.sh | bash

set -e

REPO="https://github.com/YunagiHisa/cordbeat.git"
INSTALL_DIR="${CORDBEAT_DIR:-$HOME/cordbeat}"

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
PINK='\033[95m'
NC='\033[0m'

echo ""
echo -e "${PINK}   ██████╗ ██████╗ ██████╗ ██████╗ ██████╗ ███████╗ █████╗ ████████╗${NC}"
echo -e "${PINK}  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝${NC}"
echo -e "${PINK}  ██║     ██║   ██║██████╔╝██║  ██║██████╔╝█████╗  ███████║   ██║   ${NC}"
echo -e "${PINK}  ██║     ██║   ██║██╔══██╗██║  ██║██╔══██╗██╔══╝  ██╔══██║   ██║   ${NC}"
echo -e "${PINK}  ╚██████╗╚██████╔╝██║  ██║██████╔╝██████╔╝███████╗██║  ██║   ██║   ${NC}"
echo -e "${PINK}   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝  ${NC}"
echo ""
echo "  A local-first autonomous AI agent that stays by your side."
echo ""

# ── Check git ─────────────────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
    echo -e "${RED}✗ git is required. Please install git first.${NC}"
    exit 1
fi

# ── Install / update uv ───────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}  Installing uv (Python package manager)...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
echo -e "${GREEN}  ✓ uv $(uv --version)${NC}"

# ── Clone or update repo ──────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    echo -e "${YELLOW}  Updating existing installation at $INSTALL_DIR ...${NC}"
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo -e "${YELLOW}  Cloning CordBeat to $INSTALL_DIR ...${NC}"
    git clone "$REPO" "$INSTALL_DIR"
fi
echo -e "${GREEN}  ✓ Source ready${NC}"

cd "$INSTALL_DIR"

# ── Install Python dependencies ───────────────────────────────────────────────
echo -e "${YELLOW}  Installing dependencies (this may take several minutes on first run)...${NC}"
uv sync
echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# ── Install CLI shims to ~/.local/bin ─────────────────────────────────────────
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

VENV_BIN="$INSTALL_DIR/.venv/bin"

# Create shims for all cordbeat entry points
for cmd in cordbeat cordbeat-chat cordbeat-init cordbeat-discord cordbeat-telegram \
           cordbeat-slack cordbeat-line cordbeat-whatsapp cordbeat-signal; do
    if [ -f "$VENV_BIN/$cmd" ]; then
        ln -sf "$VENV_BIN/$cmd" "$BIN_DIR/$cmd"
    fi
done
echo -e "${GREEN}  ✓ CLI commands linked to $BIN_DIR${NC}"

# Ensure ~/.local/bin is in PATH (append to shell rc if missing)
for RC in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
    if [ -f "$RC" ] && ! grep -q 'LOCAL_BIN_ADDED_BY_CORDBEAT' "$RC"; then
        printf '\n# Added by CordBeat installer (LOCAL_BIN_ADDED_BY_CORDBEAT)\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$RC"
    fi
done
export PATH="$BIN_DIR:$PATH"

# ── Run setup wizard ──────────────────────────────────────────────────────────
echo ""
cordbeat-init

echo ""
echo -e "${GREEN}  ✨ CordBeat is ready!${NC}"
echo ""
echo "  Quick start:"
echo -e "    ${CYAN}cordbeat-chat${NC}              # chat in terminal"
echo -e "    ${CYAN}cordbeat service install${NC}   # run as background service"
echo -e "    ${CYAN}cordbeat service status${NC}    # check service status"
echo ""
echo "  Adapters (after setting token in config):"
echo -e "    ${CYAN}cordbeat-discord${NC}  /  ${CYAN}cordbeat-telegram${NC}"
echo ""
echo "  To update later:"
echo -e "    ${CYAN}curl -fsSL https://raw.githubusercontent.com/YunagiHisa/cordbeat/main/install.sh | bash${NC}"
echo ""
echo -e "${YELLOW}  ⚠ Open a new terminal (or run: source ~/.bashrc) for commands to be available.${NC}"
echo ""
