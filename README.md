# Meeting Notes Pipeline

Turn a meeting or call recording into structured, goal-filtered notes: a TLDR, significant findings, decisions, open questions, action items, and embedded screenshots, all with verifiable timestamps to the source video.

**Core principle: every claim traces to a timestamp.** A summary is read by people who weren't there and will not check. An invented action item with a colleague's name on it is worse than no notes at all.

All processing is **local and offline**. Audio never leaves your machine.

## What it does

This is a Zoom/Meet/Teams recording → markdown notes → PDF pipeline, structured in **8 independent stages** that each checkpoint and resume:

1. **Probe**: extract media metadata (duration, creation time, video presence)
2. **Audio**: extract 16 kHz mono WAV
3. **Diarize**: speaker turns via `pyannote.audio`
4. **Transcribe**: per-sentence ASR with per-sentence language detection (catches mid-turn code-switching)
5. **SRT**: generate speaker-labeled transcript with bidi-safe Hebrew/Arabic wrapping
6. **Engagement**: score conversation windows to find "the moments that mattered"
7. **Frames**: extract candidate screenshots + contact sheets for vision review
8. **Cut**: extract a single full-resolution frame with optional cropping

The pipeline then produces a **PDF with embedded screenshots** that is the deliverable. Markdown alone cannot show images, and Google Docs connectors cannot upload local images programmatically.

## Install

**Requirements:** an Apple Silicon Mac (M1 or later), Homebrew, Google Chrome, about 5 GB of disk, and a free HuggingFace account for speaker separation.

**The easy way:** open Claude Code and say:

> Install the meeting-notes skill from github.com/adamfrimer-gif/meeting-notes. Follow its INSTALL.md.

Claude clones it, runs the installer, and walks you through the one manual step: accepting two model licences on HuggingFace and pasting a token into `.env`.

**By hand:**

```bash
git clone https://github.com/adamfrimer-gif/meeting-notes.git ~/.claude/skills/meeting-notes
cd ~/.claude/skills/meeting-notes && ./install.sh
```

Then do the HuggingFace step in [INSTALL.md](INSTALL.md#step-4-huggingface-token--needs-the-user) and check with `scripts/mnotes doctor`. It ends with `READY` when everything works.

### Running the pipeline

In Claude Code, drop a recording into the chat or say "notes from this call". The skill runs every stage for you. To run it by hand:

```bash
MN=~/.claude/skills/meeting-notes/scripts/mnotes
WORK=/tmp/my_meeting

$MN --work $WORK probe recording.mp4                       # duration, cost estimate
$MN --work $WORK audio recording.mp4                       # 16 kHz mono WAV
$MN --work $WORK diarize --speakers 4                      # who spoke when
$MN --work $WORK tracks "Zoom folder/Audio Record"         # optional: name speakers from Zoom per-person audio
$MN --work $WORK transcribe --expect-languages "en,he"     # per-sentence language detection
$MN --work $WORK srt --names $WORK/names.json              # transcript.srt + transcript.md
$MN --work $WORK engagement --keywords "launch,budget"     # the moments that mattered
$MN --work $WORK frames                                    # contact sheets of candidate screenshots
$MN pdf notes_v01.md                                       # HTML + PDF deliverable
```

**Every stage resumes if interrupted.** Re-run the same command to continue.

## Proofing workflow

Before a PDF ships, two independent checks run (a convention used with the Claude Code skill, not enforced by the scripts):

1. **Data proof:** an agent checks every timestamped claim, quote, number, date, name and action-item owner in the notes against `transcript.md`.
2. **Visual proof:** a second agent, blind to the first, reads the rendered PDF pages and the screenshots for cut-off figures, blank gaps, and frames that do not show what the caption says.

Findings from both are fixed (as a new `_vNN`) before delivery.

## Key principles

These are not style preferences. Each one is a failure that was measured on real audio:

| Principle | Failure mode if violated |
|-----------|--------------------------|
| **Always use `task="transcribe"`, never `translate`** | Whisper silently renders minority languages as fluent text in the majority language. Measured: "start without this" became "start with this". |
| **Detect language per sentence, not per turn** | Mid-turn code-switching gets silently translated away. |
| **Never bias Whisper with a glossary unless guarded** | A 7-term glossary can cut output from 249 characters to 65, dropping 7 of 8 sentences. |
| **Every quote is verbatim from the transcript** | "Paraphrased quote" is a contradiction. If you cannot find the sentence, it is not a quote. |
| **Do not infer speaker names from diarization** | Clustering is acoustic. "Unattributed" beats an invented name. |
| **Pass suspect flags through to the notes** | Segments marked `likely_hallucination`, `glossary_prompt_echo`, etc. are model artefacts, not speech. Never summarize from them. |
| **Specify expected languages** | On short turns, Whisper confidently hallucinates fluent minority languages. `--expect-languages en,he` flags anything else as an artefact instead of trusting it. |

(See [SKILL.md](SKILL.md) for the full playbook and measured examples.)

## Output structure

After running through all stages, a work directory contains:

```
mnotes_work/
  media.json                # video metadata
  audio.wav                 # 16 kHz mono
  diarization.json          # speaker turns + talk time
  segments.json             # per-sentence ASR + language + flags
  transcript.srt            # SRT for video players
  transcript.md             # speaker-labelled plaintext (input to summary)
  windows.json              # engagement-scored moments
  frames/                   # candidate screenshots
  contact_sheet_w01.jpg     # vision review (12 tiles per window)
  contact_sheet_w01_crops.jpg  # cropped to changed regions
  ...
```

The user then:
1. Writes a summary in markdown, citing `[mm:ss]` timestamps
2. Embeds screenshots from the contact sheets
3. Runs `scripts/mnotes pdf summary_v01.md` → produces `summary_v01.html` (images as base64) and `summary_v01.pdf`

## Measured performance

On an Apple M4 Max against a real 6 minute 16 second Zoom recording:

| Stage | Time |
|-------|------|
| Diarize (MPS) | 15 s |
| Transcribe (with per-sentence language checks) | 78 s |
| Frames | ~5 s |
| **Total** | **~2 min** (4x realtime) |

So a 60-minute meeting takes roughly 15 minutes of processing.

## Dependencies

- **ffmpeg** 4.0+ (Homebrew: `brew install ffmpeg`)
- **Apple Silicon Mac** (M1 or later), required by mlx-whisper
- **Python 3.12** (created in `.venv` by `install.sh`)
- **mlx-whisper**: Apple Silicon ASR (faster than standard Whisper)
- **pyannote.audio** 3.3+: speaker diarization (requires HuggingFace licence)
- **Pillow**: image processing
- **Google Chrome** (or Chromium, Edge, Brave) for PDF export; set `CHROME` in `.env` if it isn't in `/Applications`

## Repository layout

```
SKILL.md          the skill Claude Code loads (must stay at the repo root)
INSTALL.md        step-by-step install, written so Claude can run it
CLAUDE.md         orientation for Claude when it opens this repo
install.sh        idempotent installer: brew deps, .venv, packages, .env
scripts/mnotes    entry point; always runs with the skill's own .venv
scripts/*.py      the pipeline, PDF renderer and HuggingFace doctor
examples/sample/  a small synthetic example (no real meeting data)
```

## Examples

`examples/sample/` holds a small **synthetic** example (no real meeting data, since real recordings contain colleagues' voices): a transcript, a speaker `names.json`, the notes Markdown with one screenshot, and the rendered `notes_v01.pdf`.

Before (`transcript_v01.md`):

```
[00:19] SPEAKER_01: I'm worried about the QA window, but November 12 works.
[00:41] Alex Example: Okay, we ship November 12.
```

After (`notes_v01.md` excerpt): the decision is separated from discussion, the unconfirmed speaker stays "unattributed", and every claim carries a `[mm:ss]`.

```
- Ship on November 12 [00:41].
| unattributed | Confirm QA window | not stated |
```

Reproduce the PDF:

```bash
scripts/mnotes pdf examples/sample/notes_v01.md   # refuses to overwrite; bump _vNN first
```

## Red flags (stop and fix)

- About to write an action item with a name you inferred from diarization
- About to quote someone without the sentence in front of you
- A finding with no timestamp
- Summarizing a segment marked `suspect`
- The whole transcript came back in one language when you were told the call was multilingual
- Writing "the team decided" when the transcript only shows discussion
- Including a screenshot because a section needed one, not because the frame shows something

## Known limitations

**The crop sheet sometimes shows a webcam face rather than content:** a moving head is often the largest delta in a frame. Edge density was measured as a discriminator and the signal inverts with analysis scale (at 480px face wins, at 1440px content wins). There is no cheap reliable way to tell them apart. The contact sheet exists so a vision pass makes this call on one image. Discard the face tiles during review and move on.

## Troubleshooting

**`mnotes doctor` says "token not valid"**
- The token was revoked, mistyped, or only partly copied.
- Create a fresh one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**`mnotes doctor` says "fine-grained token"**
- HuggingFace defaults new tokens to fine-grained.
- Delete it and create a plain **Read** token instead.
- Fine-grained tokens need an extra permission box ticked and are not worth the hassle.

**Diarization works but transcription hangs**
- Run `scripts/mnotes doctor`; every `install` line should say `ok`.
- Large-v3 model (~3 GB) downloads on first `transcribe`; this is normal (measured: 78 seconds on this machine).

**FFmpeg errors about `libtorchcodec`**
- pyannote 4.x normally decodes audio via `torchcodec`, which only supports FFmpeg 4–7.
- This pipeline pre-decodes WAV with stdlib and passes a waveform, so the error doesn't happen.
- If you see it from other pyannote code, downgrade FFmpeg or update pyannote.

**`drawtext` filter missing in contact sheets**
- Homebrew FFmpeg is often built without libfreetype.
- Tile numbering is a nicety; the pipeline detects this and falls back to a row/column mapping in `frames.json`.
- Tiles fill left-to-right, top-to-bottom; use the mapping to identify one.

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` | HuggingFace read token (for gated diarization models) | from `.env` |
| `MNOTES_ASR_MODEL` | ASR model repo | `mlx-community/whisper-large-v3-mlx` |
| `HF_HOME` | HuggingFace model cache path | `~/.cache/huggingface` |
| `CHROME` | Browser for PDF export | auto-detected |

Shell variables always beat `.env`, so `HF_TOKEN=... mnotes ...` works for a one-off.

## Useful flags

```bash
mnotes <stage> --help                    # List all flags for a stage
mnotes probe recording.mp4               # Check duration before committing
mnotes transcribe --language en          # Lock to English (faster, skips language detection)
mnotes transcribe --hotwords glossary.txt  # Bias ASR toward domain terms
mnotes engagement --keywords file.txt    # Pass keywords as a file instead of comma-list
mnotes frames --radius 120               # Wider screenshot window (+/- seconds)
mnotes frames --dedupe 6.0               # Raise to drop more near-identical frames
mnotes cut 123.45 --crop --name fig.png  # Extract a single frame, cropped to delta
```

See [SKILL.md](SKILL.md) for the full playbook and [INSTALL.md](INSTALL.md) for setup and troubleshooting.

---

**Author:** Adam Frimer  
**License:** none chosen yet, so all rights are reserved by default. Ask before reusing the code.  
**Last updated:** November 2026
