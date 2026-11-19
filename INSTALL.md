# Installing Meeting Notes

This guide is written so that **Claude Code can walk you through it**. Open Claude Code and say:

> Install the meeting-notes skill from github.com/adamfrimer-gif/meeting-notes. Follow its INSTALL.md.

You can also follow it yourself. It takes about 10 minutes, most of it downloads.

---

## For Claude: how to run this install

Follow the steps in order. Run each command, check the result described under it, and only then move on. Each step is safe to repeat.

- **Do the steps a script can do yourself.** Stop only at the points marked **👤 Needs the user**. There, tell the user exactly what to click, give them the links, and wait for them to say it's done.
- **Never ask the user to paste their token into the chat**, and never print it. They paste it into `.env` themselves (Step 4).
- If a step fails, read the `STOP:` or `FAIL` line, which names the cause and the fix. Fix it if you can, otherwise explain it to the user. Don't skip ahead.
- When you finish, tell the user what works and what doesn't, using the output of `scripts/mnotes doctor`.

---

## Requirements

| Need | Why | Check |
|---|---|---|
| **Apple Silicon Mac** (M1 or later) | Transcription uses `mlx-whisper`, which only runs on Apple Silicon | `uname -m` prints `arm64` |
| **Homebrew** | Installs `ffmpeg` and `uv` | `brew --version` |
| **Google Chrome** (or Chromium, Edge, Brave) | Prints the notes to PDF | Usually already installed |
| **About 5 GB free disk** | Python packages (~1 GB) and the Whisper model (~3 GB, downloaded on first use) | |
| A free **HuggingFace** account | Speaker separation uses licence-gated models | Step 4 |

Intel Macs, Windows and Linux are not supported. On those machines, Step 2 stops with a clear message.

👤 **Needs the user, only if Homebrew is missing:** installing it asks for the Mac password. Send them to [brew.sh](https://brew.sh) and wait.

---

## Step 1. Get the code into Claude's skills folder

Claude Code loads skills from `~/.claude/skills/`. Clone straight into it:

```bash
git clone https://github.com/adamfrimer-gif/meeting-notes.git ~/.claude/skills/meeting-notes
```

**Check:** `~/.claude/skills/meeting-notes/SKILL.md` exists.

- If the clone asks for a password or says `Repository not found`, the repo is private and the user needs to be added as a collaborator. With the GitHub CLI signed in, `gh repo clone adamfrimer-gif/meeting-notes ~/.claude/skills/meeting-notes` handles the login.
- If `~/.claude/skills/meeting-notes` already exists, **don't overwrite it.** Ask the user whether to move it aside. It may be an older copy with their token in `.env`.
- If the user wants the code somewhere else, clone it there and use `./install.sh --link` in Step 2. That creates the `~/.claude/skills/meeting-notes` symlink for them.

## Step 2. Run the installer

```bash
cd ~/.claude/skills/meeting-notes && ./install.sh
```

This checks the Mac, installs `ffmpeg` and `uv` with Homebrew if they're missing, creates a Python 3.12 environment in `.venv`, installs the Python packages and creates `.env` from `.env.example` (with permissions `600`). It ends by running `scripts/mnotes doctor`.

**Check:** every line under `install` says `ok`. The `huggingface` section will say the token is missing. That's expected until Step 4.

## Step 3. Restart Claude Code

New skills are picked up when a session starts. Tell the user to start a new Claude Code session (or restart the app). `/meeting-notes` should then show up.

You can do Step 4 before restarting.

## Step 4. HuggingFace token 👤 Needs the user

Speaker separation ("who said what") uses pyannote models that require accepting a licence. It's free and takes about two minutes, but **a person has to click accept**. Walk the user through it:

1. Sign in to HuggingFace, or create a free account: [huggingface.co/join](https://huggingface.co/join)
2. Open [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and accept the licence. It asks for a company and use case; any honest answer is fine.
3. Open [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept that licence too.
4. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new token** → choose the **Read** tab, not "Fine-grained". Give it any name and copy it.
5. Open the `.env` file:
   ```bash
   open -e ~/.claude/skills/meeting-notes/.env
   ```
   Replace `«your-huggingface-read-token»` on the `HF_TOKEN=` line with the token, then save. Keep the `HF_TOKEN=` part and don't add quotes.

**Check:**

```bash
~/.claude/skills/meeting-notes/scripts/mnotes doctor
```

It ends with `READY` when everything works. If it doesn't, it names the cause:

| Doctor says | Fix |
|---|---|
| `RESULT: no token found` | `.env` wasn't saved, or the `«placeholder»` is still there |
| `FAIL: the token is not valid` | Copied only part of the token, or it was revoked. Create a new one |
| `NOTE: this is a FINE-GRAINED token` | Delete it and create a **Read** token (step 4 above) |
| `GATE` or `HIDE` next to `community-1` | The licence click in step 2 didn't go through. Open the link again while signed in and click "Agree and access repository" |

`READY` is all that matters. A `GATE` line for `speaker-diarization-3.1` is normal and can be ignored, since that older pipeline is only a fallback.

**Without a token,** everything still works except speaker separation: transcripts come back as `UNKNOWN` speakers. If the call was recorded in Zoom with separate audio per participant, `mnotes tracks` can still name speakers once diarization works (see SKILL.md).

## Step 5. Optional test run

Have the user pick any short recording (a 2 to 5 minute Zoom file is ideal), then run:

```bash
MN=~/.claude/skills/meeting-notes/scripts/mnotes
$MN --work /tmp/mn_test probe "«recording»"
$MN --work /tmp/mn_test audio "«recording»"
$MN --work /tmp/mn_test diarize --speakers 2
$MN --work /tmp/mn_test transcribe --expect-languages en
```

The first `transcribe` downloads the Whisper model (~3 GB), so it's slow once. **Check:** `/tmp/mn_test/segments.json` exists and `diarize` printed talk time per speaker. Delete `/tmp/mn_test` afterwards, since it holds the recording's audio.

Done. From now on, dropping a recording into Claude Code, or saying "notes from this call", runs the skill.

---

## Updating

```bash
cd ~/.claude/skills/meeting-notes && git pull && ./install.sh
```

Your `.env` and `.venv` aren't tracked by git, so they survive updates.

## Uninstalling

**For Claude:** confirm with the user first. This removes their `.env` (their token) as well. Move things to the Trash rather than deleting them, so it can be undone:

```bash
mv ~/.claude/skills/meeting-notes ~/.Trash/meeting-notes-$(date +%Y%m%d)
```

The downloaded models (~3 GB) stay in `~/.cache/huggingface/hub/`, in the folders whose names start with `models--mlx-community--whisper` and `models--pyannote--`. Other tools may share that cache, so the user can drag just those folders to the Trash in Finder (`open ~/.cache/huggingface/hub`).

---

## Reference

### Settings in `.env`

| Variable | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace **Read** token | none (required for diarization) |
| `MNOTES_ASR_MODEL` | Whisper model. `mlx-community/whisper-medium-mlx` is faster for drafts, but worse on Hebrew | `mlx-community/whisper-large-v3-mlx` |
| `HF_HOME` | Where models are cached | `~/.cache/huggingface` |
| `CHROME` | Browser used for PDF export, if it isn't in `/Applications` | auto-detected |

A shell variable always beats `.env`, so `HF_TOKEN=… mnotes …` works for a one-off. A value still wrapped in `«guillemets»` counts as unset, so a half-finished edit fails loudly instead of silently.

Token lookup order: `--token` flag, then `HF_TOKEN` / `HUGGINGFACE_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, then `.env`, then `~/.cache/huggingface/token`.

### Troubleshooting

**`OSError: Could not load this library: …libtorchcodec_core…`.** pyannote 4.x normally decodes audio with `torchcodec`, which only supports FFmpeg 4 to 7, and Homebrew ships FFmpeg 8. `mnotes` avoids this by decoding the WAV itself, so you won't see it from `mnotes`. If it shows up from other pyannote code, that's the cause.

**Contact sheet tiles have no numbers.** Homebrew's FFmpeg is often built without the `drawtext` filter. The pipeline detects this and writes a row/column mapping to `frames.json` instead. Tiles fill left to right, top to bottom.

**`transcribe` seems stuck on first run.** It's downloading the ~3 GB Whisper model. Later runs start immediately.

**PDF export says no Chrome found.** Install Google Chrome, or add `CHROME=/path/to/browser` to `.env`.

### Performance

On an Apple M4 Max, a 6 minute recording takes about 2 minutes end to end (diarize 15 s, transcribe 78 s), and a 14 minute recording about 4 minutes. Plan on roughly a quarter of the recording's length. `mnotes probe` prints an estimate before you start.

### Security notes

- **Nothing leaves the machine** except model downloads from HuggingFace. Audio, transcripts and notes stay local.
- **The only secret is the HuggingFace Read token** in `.env`. That file is `chmod 600` and in `.gitignore`. `scripts/envload.py` reads it into the environment, and `hf_doctor.py` shows only its first three characters and its length.
- Security scanners such as SkillSpector flag "credential access" in `envload.py`, `hf_doctor.py`, `mnotes.py`, `install.sh` and these docs. That's the token handling described above, nothing more.
