#!/usr/bin/env python3
"""Minimal .env loader, no dependency on python-dotenv.

Looks for `.env` in the skill root (the parent of this scripts/ directory).
A real shell variable always wins over the file, so `HF_TOKEN=... command`
still overrides for a one-off run.
"""

import os
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = SKILL_ROOT / ".env"


def load_env(path=None, override=False):
    """Read KEY=VALUE lines into os.environ. Returns the keys it set."""
    p = Path(path) if path else ENV_PATH
    if not p.exists():
        return []
    set_keys = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        # Strip matching quotes, then trailing inline comments on bare values.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].strip()
        # An unpopulated placeholder is not a value.
        if not val or (val.startswith("«") and val.endswith("»")):
            continue
        if override or not os.environ.get(key):
            os.environ[key] = val
            set_keys.append(key)
    return set_keys
