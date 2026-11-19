---
name: meeting-notes
description: Use when a meeting or call recording needs to become notes, minutes, a summary, action items, decisions, or an SRT. Triggers on "summarize this meeting", "notes from this call", "what were the action items", "transcribe this recording", "make me an SRT", or a dropped .mp4/.mov/.m4a/.wav/.mkv of a call, including Zoom, Meet and Teams recordings. Also use when a transcript already exists but needs speaker attribution, multilingual handling, or screenshots pulled from the video.
---

# Meeting Notes

Turn a recording into goal-filtered notes: a 60 second TLDR, the significant findings, decisions, action items, open questions, and screenshots pulled from the moments that mattered.

**Core principle: every claim in the notes traces to a timestamp in the transcript.** A meeting summary is read by people who were not there and will not check. An invented action item with a colleague's name on it is worse than no notes at all.

All processing is local. Audio never leaves the machine.

## Non-negotiable rules

These are not style preferences. Each one is a failure that was measured on this pipeline.

| Rule | What happens if you break it |
|---|---|
| **Never pass `task="translate"`. Always `task="transcribe"`.** | Whisper silently renders Hebrew as fluent English with no marker. Measured: "אפשר להתחיל **בלי** זה" (start **without** it) became "we can start **with this**". The meaning inverted and the notes recorded the opposite of what was said. |
| **Detect language per sentence, not per file or per turn.** | A speaker who switches language mid-turn gets the minority language translated away. Turn-level detection cannot see it. `mnotes transcribe` does this by default; do not pass `--no-verify-language` on a multilingual call. |
| **Do not bias the decoder with a glossary unless you check the output.** | Measured: passing a 7 term glossary as `initial_prompt` cut a turn from 249 characters to 65, dropping 7 of 8 sentences, and on another segment made Whisper emit the glossary list itself as if someone had spoken it. `--hotwords` is opt-in and guarded. Correct terminology in a **post**-pass instead. |
| **Never present a paraphrase as a quote.** | Quotes are copied from the transcript and cleaned of filler only. If you cannot find the sentence, it is not a quote. |
| **Never assign an action item to someone whose name you inferred.** | Speaker labels come from acoustic clustering. If the mapping to a real person is not confirmed, write "unattributed" and say so. |
| **Carry `suspect` flags through to the notes.** | Segments flagged `likely_hallucination`, `glossary_prompt_echo`, `unexpected_language` or `minority_language_*` are model artefacts, not speech. Never summarize from them. |
| **Pass `--expect-languages` when you know what was spoken.** | Measured on a real 6 minute English call: short turns of filler produced confident German and Portuguese ("Obrigado.", "Und dann...") because Whisper invents fluent text on low-content audio and labels it accordingly. A one-off minority language on a short turn is an artefact, not a bilingual moment. |

**Not installed yet, or a stage fails with a missing package or token?** Run `scripts/mnotes doctor`. If it reports anything other than all `ok` and `READY`, follow [INSTALL.md](INSTALL.md) before going further. Everything except diarization works without the HuggingFace token.

## Workflow

Run everything from the **local Mac**. `MN=~/.claude/skills/meeting-notes/scripts/mnotes`

### Stage 0. Harvest context before touching the media

Do not ask the user what you can look up.

1. `$MN --work «workdir» probe «recording»` gives duration and `creation_time`.
2. If a calendar connector is available (Google Calendar, Outlook), search it around that timestamp. The event gives you the meeting title, description, agenda, and the attendee list. That is the goal and the speaker roster in one step.
3. Enrich attendees with role and team from whatever people directory or memory the user has, so the notes can say who owns what. Skip this if there is none.
4. If there is no calendar or it has nothing, ask the user for the goal and who was on the call, including roughly how many people spoke.
5. Warn before starting if `est_transcribe_min` is large. A 90 minute recording is a real wait.

**Confirm the inferred goal with the user before summarizing.** Everything downstream is filtered by it, so a wrong goal wastes the whole run.

### Stage 1. Prefer an existing transcript

Zoom, Meet and Teams often save a `.vtt` or `.txt` next to the video with **real speaker names already attached**. Look for one. Platform labels beat acoustic diarization every time and cost nothing. If found, use it for speaker turns and skip Stage 3.

Also look in the recording's folder for:
- **`chat.txt`** (Zoom): links and notes posted during the call, with local timestamps. Subtract the recording's start time to place them in the transcript.
- **`Audio Record/`** (Zoom, when "Record a separate audio file for each participant" is on): one audio file per person, named after them, such as `audioSamRivera2….m4a`. This is the best speaker evidence you can get. Keep it for Stage 3.

### Stage 2. Audio

```bash
$MN --work «workdir» audio «recording»
```

### Stage 3. Diarization

```bash
$MN --work «workdir» diarize --speakers «N»
```

Pass `--speakers` when the attendee count is known; it constrains the clustering and sharply improves accuracy. Use `--min-speakers`/`--max-speakers` when unsure.

Then map `SPEAKER_00` style labels onto real people using, in order of reliability:
1. **Per-participant audio files**, when Stage 1 found them. One command does the mapping:
   ```bash
   $MN --work «workdir» tracks "«recording folder»/Audio Record"
   ```
   For each diarized turn, the participant whose own track is loudest is the speaker. It writes `names.json`, prints the share of talk time behind each name (98 to 99% is typical), and leaves a label unnamed if the evidence is below `--min-share` (default 0.8). The names come from the file names, so check the spelling.
2. Self-identification in the transcript ("this is Sam", someone addressed by name).
3. `talk_time_sec` in `diarization.json`. Whoever ran the meeting usually dominates it.
4. The calendar attendee list.

Write the mapping to `names.json` as `{"SPEAKER_00": "Sam Rivera"}`. **Show the mapping to the user for confirmation before it reaches the notes.** Leave anyone you cannot place as `SPEAKER_0n` rather than guessing.

### Stage 4. Transcribe

```bash
$MN --work «workdir» transcribe --expect-languages "en,he"
```

Per-sentence language detection is on by default. Watch the summary line for `sentences re-decoded in another language`, which tells you code-switching was caught, and `spurious-language segments flagged`, which tells you artefacts were caught.

Always pass `--expect-languages` with the languages actually spoken. Anything outside that set gets flagged rather than trusted. Without it the pipeline still guards rare languages on short turns, but the allow-list is stricter and free.

Interrupted runs resume from the checkpoint; re-run the same command.

### Stage 5. Terminology proofing pass

Read `transcript.md` and correct domain vocabulary, then rewrite the corrections back. This is where transcription reaches S tier.

- Look up unfamiliar or garbled terms with the organisation's internal knowledge tool if one is connected rather than generic web search, since the vocabulary is usually internal. Fall back to the company's public site for public naming.
- Audit **numbers, dates, names and version strings** specifically. These are where ASR fails hardest and where a wrong summary does real damage.
- Fix only what you can justify. A plausible-sounding correction that changes meaning is worse than a visible typo. When unsure, leave the original and flag it.
- Keep a `corrections.md` log of what changed, so the user can audit it.

### Stage 6. SRT

```bash
$MN --work «workdir» srt --names names.json
```

Produces `transcript.srt` (bidi-wrapped so mixed Hebrew and English render correctly in players, with non-dominant languages tagged) and `transcript.md` (the speaker-labelled text the summary is written from).

### Stage 7. Find the moments that mattered

```bash
$MN --work «workdir» engagement --keywords "«goal keywords»"
```

Scores sliding windows on turn-taking rate, stance markers, decision language, question density and goal-keyword hits, then returns the top non-overlapping peaks. Pass keywords drawn from the **confirmed goal**, which is what makes the filtering goal-driven rather than generic.

### Stage 8. Screenshots

```bash
$MN --work «workdir» frames
```

For each peak this samples a 5 second grid across ±60 seconds, adds ffmpeg scene-change hits (a frame where the deck advanced is worth far more than an arbitrary tick), ranks candidates by edge density and sharpness so shared screens beat flat webcam grids, drops near-duplicates by visual signature, and tiles the survivors into **one contact sheet per peak**.

The dedupe matters more than it sounds. On a real call, 48 candidates collapsed to 9 distinct frames, because a shared screen barely changes for minutes at a time and timestamp spacing does not catch that. If a sheet still looks repetitive, raise `--dedupe`.

**Review the contact sheets, not the individual frames.** One image holds around 12 candidates, which is roughly a tenth of the vision cost of checking each frame.

Each window produces two sheets, and the second one usually matters more:

- `contact_sheet_wNN.jpg` shows full frames, for context.
- `contact_sheet_wNN_crops.jpg` shows each frame **cropped to what actually changed** since a few seconds earlier. On a screen share the meaningful delta is a new message or an edited line, a few percent of a 1728px wide frame and completely unreadable once the full screenshot is scaled into a document. In the crop it is legible.

Then pull the winners, in whichever form suits the point being made:

```bash
$MN --work «workdir» cut «seconds» --crop      --name «figure-01.png»   # just the delta, readable
$MN --work «workdir» cut «seconds» --highlight --name «figure-02.png»   # full frame, box around the delta
$MN --work «workdir» cut «seconds»             --name «figure-03.png»   # plain full frame
```

Use `--crop` when the content is the point, `--highlight` when the reader needs to see where in the screen it happened. `--box x,y,w,h` overrides the detected region with fractions of the frame.

**Known limitation, do not paper over it:** the crop sheet sometimes shows a webcam face rather than content, because a moving head can be the largest change in the frame. There is no reliable cheap way to tell them apart. Edge density was measured as a discriminator across three analysis resolutions and the signal inverts with scale (at 480px a webcam tile scored 4.40 against content's 2.75; at 1440px 2.15 against 2.22). The contact sheet exists precisely so a vision pass makes this call on one image. Discard the face tiles during review and move on.

If `numbered: false` in `frames.json`, this ffmpeg has no `drawtext` filter and the tiles are unlabelled. Identify tiles by the `row`/`col` mapping in `frames.json`; the grid fills left to right, top to bottom.

Skip a screenshot rather than including a dull one. A frame earns its place only if it shows something the text cannot: a slide, a diagram, a metric, a design.

Audio-only recordings skip this stage automatically.

## The summary

Write to a dated folder, versioned `_vNN`. If the user's CLAUDE.md sets folder or file-naming conventions, follow them. Otherwise create `YYYY_MM_DD_«Meeting_Title»/` in the current working directory, name files `«name»_v01.md`, and never overwrite a version: a revision is `_v02`.

Structure, in this order:

1. **Header.** Meeting title, date, duration, attendees, and the goal in one line.
2. **TLDR.** Readable in 60 seconds, roughly 120 to 150 words. What was the meeting for, what came out of it, what happens next. No timestamps here.
3. **Significant findings.** The substantive points, each with a `[mm:ss]` timestamp. Attribute a point to a person **when who said it matters**, for example when someone made the call, raised the blocker, or owns the area. Do not attribute routine back-and-forth.
4. **Decisions.** What was actually settled, kept strictly separate from what was merely discussed. If nothing was decided, say so plainly. This is the section people misreport most.
5. **Open questions and disagreements.** Unresolved threads, and points where people did not converge. Often the most valuable section and the one auto-summarizers drop.
6. **Action items.** Owner, action, and due date when stated. Mark inferred owners as inferred. Separate committed items from "someone should probably" ideas.
7. **Next steps and opportunities.** Things that would move the work forward, including ones nobody said out loud but that follow from the discussion. Label these as your suggestions, not as meeting output.
8. **Screenshots.** Placed inline next to the point they illustrate, each captioned with its timestamp.

Rules for the writing itself:

- **Exclude anything off-goal.** Weekend chat, tangents and diversions do not appear, even when entertaining. If a tangent turned out to matter, it is on-goal by definition.
- **Every factual claim carries a `[mm:ss]`.** No exceptions in findings, decisions or action items.
- **Quotes are verbatim** from the transcript, filler removed, nothing else changed.
- **Flag low-confidence audio** inline rather than asserting through it: "«unclear around 14:32»".
- **No em dashes.** This is house style; drop it if the user's own style says otherwise.
- Note the languages spoken if the call was multilingual, and quote in the original language with a translation alongside.

## Delivery

Two files are always produced in the dated folder: the Markdown source, and **a PDF with the screenshots embedded, which is the deliverable**. Markdown viewers often don't show local images, and the Google connectors can't put local images into a Doc (the Drive upload takes the whole file inline, and Docs `insertInlineImage` needs a public URL). So the PDF is what the user opens and shares.

```bash
$MN pdf «folder»/«name»_vNN.md
```

This writes a self-contained `_vNN.html` (images embedded as base64, clickable screenshots) and a print-styled A4 `_vNN.pdf` via headless Chrome, and refuses to overwrite existing versions. **Look at the PDF pages (Read with `pages`) before reporting it**: check for screenshots cut across pages or large blank gaps.

**Open the folder for the user.** They should never have to go and find it in Finder. Once the PDF has passed the page check, do both of these:

1. Run `open "«folder»"` from the local Mac, so Finder opens on the dated folder by itself. It is local and has no side effects, so no confirmation is needed.
2. Make the **first line of the completion summary** a clickable link to the folder: `📁 [«folder name»](file:///absolute/path/to/folder/)`. Use an absolute `file://` URL with a **trailing slash**, encode spaces as `%20`, and use the plain folder name as the link text. Link the PDF on its own line right below it, then the other files.

Then:

1. Don't offer a Google Doc as the default. If the user asks for one, say plainly that images can't go in through the connectors, and offer a .docx they drag into Drive or an upload through Claude in Chrome.
2. If a Doc is made, **ask whether it should be publicly available** or restricted, and confirm before changing any sharing setting.
3. Never post to Slack, Confluence or email without being asked. A meeting summary can contain sensitive material and the user decides who sees it.

Recordings contain colleagues' voices. Keep processing local, keep the work directory out of shared locations, and do not upload the audio anywhere.

## Red flags: stop and fix

- About to write an action item with a name you inferred from clustering
- About to quote someone without the sentence in front of you
- A finding with no timestamp
- Summarizing a segment marked `suspect`
- The whole transcript came back in one language when you were told the call was multilingual
- Writing "the team decided" when the transcript only shows discussion
- Including a screenshot because a section needed one, not because the frame shows something

## Rationalizations

| Excuse | Reality |
|---|---|
| "The gist is right, the exact words do not matter" | The person named in an action item disagrees. Attribution errors damage trust in every future summary. |
| "Timestamps clutter the notes" | Timestamps are what makes the notes checkable. Keep them. |
| "Diarization is close enough to name people" | Close enough puts words in a colleague's mouth. Confirm or write unattributed. |
| "It detected English, so it was English" | It detects the *majority* language of a window and translates the rest. That is the measured failure this skill exists to prevent. |
| "I will skip the terminology pass, it read fine" | Fluent and wrong is the normal ASR failure mode. Fluency is not evidence. |
| "Extracting every frame is more thorough" | It is also 100x the cost for a worse result. Contact sheets exist for this. |
| "I will just publish it to save a step" | Publishing is the user's call, always. |

## Quick reference

```bash
MN=~/.claude/skills/meeting-notes/scripts/mnotes
W=«workdir»

$MN --work $W probe «rec»            # duration, video?, creation_time, cost estimate
$MN --work $W audio «rec»            # 16kHz mono wav
$MN --work $W diarize --speakers 4   # speaker turns + talk time
$MN --work $W tracks "«rec folder»/Audio Record"  # names.json from Zoom per-person audio
$MN --work $W transcribe --expect-languages "en,he"
$MN --work $W srt --names names.json # transcript.srt + transcript.md
$MN --work $W engagement --keywords "banner,launch,budget"
$MN --work $W frames                 # contact sheets to review
$MN --work $W cut 1284 --crop --name fig-01.png
$MN pdf «folder»/notes_v01.md        # HTML + PDF deliverable
$MN doctor                           # is the install complete?
```

Every stage checkpoints and resumes. `--force` redoes one. `«stage» --help` lists tuning flags.

Useful flags: `--speakers N` (big accuracy win), `--expect-languages` (flags ASR artefacts), `--dedupe` (raise it if contact sheets look repetitive), `--min-score` (raise it to ignore low-energy stretches), `--top N` and `--radius` (screenshot breadth), `--no-verify-language` (faster, only for confirmed single-language calls), `--hotwords file` (opt-in glossary biasing, guarded against truncation).

## Measured on real audio

A 6 minute 16 second Zoom recording on an M4 Max: diarize 15 seconds, transcribe 78 seconds, 48 speaker turns, 68 sentence-level cues, 5 artefacts correctly flagged, 48 frame candidates deduped to 9. Roughly 4x realtime overall, so a 60 minute meeting is about 15 minutes of processing.
