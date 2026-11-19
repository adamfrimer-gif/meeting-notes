# meeting-notes

A Claude Code skill: turns a meeting recording into timestamped notes and a PDF, fully on-device.

- **Asked to install or set this up?** Follow [INSTALL.md](INSTALL.md) step by step. It's written for you, with a check after each step and the points where the user has to act marked 👤.
- **Asked to take notes from a recording?** That's [SKILL.md](SKILL.md). If `scripts/mnotes doctor` doesn't pass, finish INSTALL.md first.
- **Changing the code?** The CLI is `scripts/mnotes.py` (one file, one `cmd_*` function per stage). Always run it through `scripts/mnotes`, which uses the skill's own `.venv`. Never commit `.env`, since it holds the user's HuggingFace token.
