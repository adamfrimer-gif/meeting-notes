#!/usr/bin/env python3
"""Diagnose HuggingFace access for the diarization models.

Turns "it doesn't work" into a specific cause: no token, wrong token type,
or a licence gate that was never accepted.

Never prints the token value.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pyannote 4.x community pipeline, and the 3.1 stack, which needs both repos.
REPOS = [
    ("pyannote/speaker-diarization-community-1", "pyannote 4.x pipeline"),
    ("pyannote/speaker-diarization-3.1", "pyannote 3.1 pipeline"),
    ("pyannote/segmentation-3.0", "required BY 3.1"),
]


def find_token():
    from envload import ENV_PATH, load_env
    from_env_file = set(load_env())
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(name)
        if v:
            where = (f"{ENV_PATH}" if name in from_env_file
                     else f"environment variable {name}")
            return v.strip(), where
    for p in (Path.home() / ".cache/huggingface/token",
              Path.home() / ".huggingface/token"):
        if p.exists() and p.read_text().strip():
            return p.read_text().strip(), f"file {p}"
    return None, None


def main():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (GatedRepoError, RepositoryNotFoundError,
                                       HfHubHTTPError)

    token, source = find_token()
    print("=" * 68)
    if not token:
        print("RESULT: no token found.")
        print()
        from envload import ENV_PATH
        print(f"Looked in: {ENV_PATH}")
        print("           $HF_TOKEN, $HUGGINGFACE_TOKEN, $HUGGING_FACE_HUB_TOKEN,")
        print("           ~/.cache/huggingface/token, ~/.huggingface/token")
        print()
        if ENV_PATH.exists():
            print("The .env file exists but HF_TOKEN is still the «placeholder».")
            print(f"Edit {ENV_PATH} and replace it with your token.")
        else:
            print(f"Create {ENV_PATH} with a line: HF_TOKEN=hf_...")
        return 2

    print(f"Token found via {source}")
    print(f"  starts with: {token[:3]}…  length: {len(token)}")
    if not token.startswith("hf_"):
        print("  WARNING: HuggingFace tokens normally start with 'hf_'.")
        print("  If you pasted a URL, a username, or a truncated value, redo step 3.")
    print("=" * 68)

    api = HfApi()

    # 1. Is the token valid at all?
    try:
        me = api.whoami(token=token)
    except Exception as e:
        print("FAIL: the token is not valid (HuggingFace rejected it).")
        print(f"       {type(e).__name__}: {e}")
        print()
        print("Most likely: it was revoked, mistyped, or only partly copied.")
        print("Create a fresh one and set it again.")
        return 1

    print(f"OK   token is valid. Logged in as: {me.get('name')}")

    # 2. Token type. This is the usual culprit: HuggingFace now defaults new
    #    tokens to 'fine-grained', and a fine-grained token cannot read gated
    #    repos unless that specific permission box was ticked.
    auth = (me.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role")
    kind = auth.get("type") or ("fineGrained" if auth.get("fineGrained") else None)
    if role:
        print(f"     token role: {role}")
    if kind:
        print(f"     token type: {kind}")
    fine = (kind == "fineGrained") or bool(auth.get("fineGrained"))
    if fine:
        perms = (auth.get("fineGrained") or {}).get("scoped") or []
        print("     NOTE: this is a FINE-GRAINED token.")
        print("     It can only read gated repos if you ticked")
        print("     'Read access to contents of all public gated repos you can access'.")
        if not perms:
            print("     No scoped permissions visible, which usually means it cannot.")
        print("     Simplest fix: delete it and create a plain 'Read' token instead.")

    # 3. Per-repo gate status. This is what actually blocks diarization.
    print("-" * 68)
    usable = []
    for repo, why in REPOS:
        try:
            api.model_info(repo, token=token)
            print(f"OK   {repo}")
            print(f"       ({why})")
            usable.append(repo)
        except GatedRepoError:
            print(f"GATE {repo}")
            print(f"       ({why})")
            print(f"       You have NOT accepted the licence. Open:")
            print(f"       https://huggingface.co/{repo}")
            print(f"       and click 'Agree and access repository'.")
        except RepositoryNotFoundError:
            print(f"HIDE {repo}")
            print(f"       ({why})")
            print("       Reported as not-found. With a valid token this almost always")
            print("       means the gate is unaccepted OR the token cannot see gated")
            print("       repos (fine-grained token missing the gated-repo permission).")
            print(f"       Open https://huggingface.co/{repo} and accept, then")
            print("       if it still fails, create a plain 'Read' token.")
        except HfHubHTTPError as e:
            print(f"ERR  {repo}: {e}")

    print("=" * 68)
    if "pyannote/speaker-diarization-community-1" in usable:
        print("READY: community-1 is accessible. Diarization will work.")
        return 0
    if ("pyannote/speaker-diarization-3.1" in usable
            and "pyannote/segmentation-3.0" in usable):
        print("READY: the 3.1 stack is accessible. Diarization will work.")
        return 0
    if "pyannote/speaker-diarization-3.1" in usable:
        print("NOT READY: 3.1 is accessible but segmentation-3.0 is not.")
        print("           3.1 loads segmentation-3.0 internally, so you need both.")
        return 1
    print("NOT READY: no complete diarization pipeline is accessible yet.")
    print("           Accept the gates listed above, then re-run this check.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
