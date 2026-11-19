#!/bin/sh
# Meeting Notes installer. Safe to re-run: every step checks before it acts.
#
#   ./install.sh          install everything that can be installed without a human
#   ./install.sh --link   also symlink this folder to ~/.claude/skills/meeting-notes
#
# The one step a script cannot do is the HuggingFace token (it needs you to click
# "accept" on two model licences). The last thing this prints tells you exactly
# what is left.
set -e
cd "$(dirname "$0")"
ROOT=$(pwd)
say() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

say "Checking this is an Apple Silicon Mac"
[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] \
  || fail "transcription uses mlx-whisper, which only runs on Apple Silicon Macs (M1 or later). This machine is $(uname -s) $(uname -m)."
echo "ok"

say "Checking Homebrew"
command -v brew >/dev/null 2>&1 || fail "Homebrew is missing. Install it from https://brew.sh (it asks for your Mac password), then re-run ./install.sh"
echo "ok"

for tool in ffmpeg uv; do
  say "Checking $tool"
  if command -v "$tool" >/dev/null 2>&1; then echo "ok"; else brew install "$tool"; fi
done

say "Creating the Python environment (.venv, Python 3.12)"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
echo "ok"

say "Installing Python packages (first run downloads ~1 GB, a few minutes)"
uv pip install --python .venv/bin/python -q mlx-whisper "pyannote.audio>=3.3" pillow numpy huggingface_hub
echo "ok"

say "Creating .env for your HuggingFace token"
if [ -f .env ]; then echo "exists, left as is"; else cp .env.example .env && echo "created"; fi
chmod 600 .env

chmod +x scripts/mnotes scripts/*.py install.sh 2>/dev/null || true

if [ "$1" = "--link" ]; then
  say "Linking into ~/.claude/skills"
  mkdir -p "$HOME/.claude/skills"
  T="$HOME/.claude/skills/meeting-notes"
  if [ -e "$T" ] && [ "$(cd "$T" && pwd -P)" != "$(pwd -P)" ]; then
    fail "$T already exists and is a different folder. Move it aside, then re-run with --link."
  fi
  [ -e "$T" ] || ln -s "$ROOT" "$T"
  echo "ok: $T"
fi

say "Checking the install"
scripts/mnotes doctor && exit 0
cat <<'MSG'

Installed. Speaker names need one more step that only a person can do
(about two minutes, free). See "HuggingFace token" in INSTALL.md, then run:

    scripts/mnotes doctor

Everything except speaker separation already works without it.
MSG
