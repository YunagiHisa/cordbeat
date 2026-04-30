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
P='\033[95m'   # bright pink
M='\033[35m'   # dark magenta
PD='\033[91m'  # hot pink
NC='\033[0m'

echo ""
echo -e "${P}   ██████╗ ██████╗ ██████╗ ██████╗ ██████╗ ███████╗ █████╗ ████████╗${NC}"
echo -e "${M}  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝${NC}"
echo -e "${PD}  ██║     ██║   ██║██████╔╝██║  ██║██████╔╝█████╗  ███████║   ██║   ${NC}"
echo -e "${P}  ██║     ██║   ██║██╔══██╗██║  ██║██╔══██╗██╔══╝  ██╔══██║   ██║   ${NC}"
echo -e "${M}  ╚██████╗╚██████╔╝██║  ██║██████╔╝██████╔╝███████╗██║  ██║   ██║   ${NC}"
echo -e "${PD}   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝  ${NC}"
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
echo -e "${YELLOW}  Installing dependencies (this may take a minute)...${NC}"
uv sync --quiet
echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# ── Run setup wizard ──────────────────────────────────────────────────────────
echo ""
uv run cordbeat-init

echo ""
echo -e "${GREEN}  ✨ CordBeat is ready!${NC}"
echo ""
echo "  To chat again:"
echo -e "    ${CYAN}cd $INSTALL_DIR${NC}"
echo -e "    ${CYAN}uv run cordbeat-chat${NC}"
echo ""
echo "  To run as headless server (Discord / Telegram):"
echo -e "    ${CYAN}uv run cordbeat${NC}"
echo ""
echo "  To update later:"
echo -e "    ${CYAN}curl -fsSL https://raw.githubusercontent.com/YunagiHisa/cordbeat/main/install.sh | bash${NC}"
echo ""
