#!/usr/bin/env python3
"""
mnotes - local meeting-recording pipeline.

Stages write checkpoints into a work directory so a long recording never has to
restart from scratch. Every stage is independently runnable.

  probe       media facts (duration, video present, creation time)
  audio       extract 16kHz mono WAV
  diarize     pyannote speaker turns  -> diarization.json
  transcribe  per-segment ASR, per-segment language detection -> segments.json
  srt         segments.json -> transcript.srt (bidi-safe)
  engagement  score conversation windows -> windows.json
  frames      candidate screenshots + contact sheets for vision review
  cut         extract one full-resolution frame
  tracks      name speakers from per-participant audio files (Zoom "Audio Record")
  pdf         render notes Markdown to HTML + PDF (runs scripts/notes_to_pdf.py)
  doctor      check the install and HuggingFace access (runs scripts/hf_doctor.py)

Nothing here talks to the network except model downloads from HuggingFace.
Audio never leaves the machine.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from envload import load_env
    load_env()          # .env in the skill root, shell vars still win
except Exception:
    pass

ASR_MODEL = os.environ.get("MNOTES_ASR_MODEL", "mlx-community/whisper-large-v3-mlx")
DIA_MODELS = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]

# Scripts whose text runs right-to-left. Used for SRT bidi wrapping.
RTL_RANGES = [
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x074F),  # Syriac
    (0xFB1D, 0xFDFF),  # Hebrew/Arabic presentation forms
    (0xFE70, 0xFEFF),
]
RLE, PDF, RLM = "\u202b", "\u202c", "\u200f"  # embedding, pop, RTL mark


def die(msg, code=1):
    print(f"mnotes: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def ffprobe_json(path, *extra):
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", *extra, str(path)]).stdout
    return json.loads(out)


def is_rtl(text):
    """True when the letters in `text` are predominantly RTL."""
    rtl = ltr = 0
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in RTL_RANGES):
            rtl += 1
        elif ch.isalpha():
            ltr += 1
    return rtl > ltr


def workdir(args):
    d = Path(args.work).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(path, what):
    p = Path(path)
    if not p.exists():
        die(f"missing {what}: {p} (run the earlier stage first)")
    return json.loads(p.read_text())


def save(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


# ----------------------------------------------------------------- probe


def cmd_probe(args):
    wd = workdir(args)
    if not Path(args.media).exists():
        die(f"no such file: {args.media}")
    meta = ffprobe_json(args.media)
    streams = meta.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        die("no audio stream found; cannot transcribe this file")
    fmt = meta.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    st = Path(args.media).stat()
    info = {
        "path": str(Path(args.media).resolve()),
        "duration_sec": round(float(fmt.get("duration", 0)), 2),
        "size_bytes": st.st_size,
        "has_video": bool(video),
        "video_codec": video[0].get("codec_name") if video else None,
        "width": video[0].get("width") if video else None,
        "height": video[0].get("height") if video else None,
        "audio_codec": audio[0].get("codec_name"),
        "audio_channels": audio[0].get("channels"),
        # creation_time is the reliable hook for finding the calendar event
        "creation_time": tags.get("creation_time"),
        "file_mtime": st.st_mtime,
    }
    d = info["duration_sec"]
    info["duration_hms"] = f"{int(d // 3600):02d}:{int(d % 3600 // 60):02d}:{int(d % 60):02d}"
    # Rough local cost so the caller can warn before a 90 minute job.
    info["est_transcribe_min"] = round(d / 60 / 8, 1)
    info["est_diarize_min"] = round(d / 60 / 12, 1)
    save(info, wd / "media.json")
    print(json.dumps(info, indent=2))


# ----------------------------------------------------------------- audio


def cmd_audio(args):
    wd = workdir(args)
    out = wd / "audio.wav"
    if out.exists() and not args.force:
        print(f"{out} exists (use --force to redo)")
        return
    # 16 kHz mono is what both whisper and pyannote want.
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.media),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)])
    with wave.open(str(out)) as w:
        dur = w.getnframes() / w.getframerate()
    print(f"wrote {out} ({dur:.1f}s, 16kHz mono)")


# ----------------------------------------------------------------- diarize


def cmd_diarize(args):
    wd = workdir(args)
    out = wd / "diarization.json"
    if out.exists() and not args.force:
        print(f"{out} exists (use --force to redo)")
        return

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    tokfile = Path.home() / ".cache/huggingface/token"
    if not token and tokfile.exists():
        token = tokfile.read_text().strip()
    if not token:
        die("no HuggingFace token. Put HF_TOKEN in the skill's .env, export it, "
            "or pass --token.\n"
            "Run scripts/hf_doctor.py to find out exactly what is missing.")

    import torch
    from pyannote.audio import Pipeline

    pipe = last_err = None
    for repo in DIA_MODELS:
        try:
            pipe = Pipeline.from_pretrained(repo, token=token)
            print(f"loaded {repo}")
            break
        except Exception as e:  # gated repo, missing licence, offline
            last_err = e
    if pipe is None:
        die(f"could not load any diarization pipeline. Last error: {last_err}\n"
            "Most likely you have not accepted the model licence yet (see SETUP.md).")

    if torch.backends.mps.is_available() and not args.cpu:
        try:
            pipe.to(torch.device("mps"))
            print("running on MPS")
        except Exception:
            print("MPS unavailable for this pipeline, using CPU")

    kwargs = {}
    if args.speakers:
        kwargs["num_speakers"] = args.speakers
    else:
        if args.min_speakers:
            kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers:
            kwargs["max_speakers"] = args.max_speakers

    audio = wd / "audio.wav"
    if not audio.exists():
        die("audio.wav not found; run `mnotes audio` first")

    # Hand pyannote a decoded waveform rather than a path.
    #
    # pyannote 4.x decodes audio through torchcodec, which only links against
    # FFmpeg 4 to 7. On a machine with FFmpeg 8 it dies with "Could not load
    # this library: libtorchcodec_core*.dylib". We already produced a 16 kHz
    # mono WAV in the previous stage, so decode it with the stdlib and skip
    # that whole dependency. Also avoids a second resample.
    import numpy as np
    with wave.open(str(audio)) as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    wav = torch.from_numpy(pcm.copy()).unsqueeze(0)   # (channel, time)
    dia_out = pipe({"waveform": wav, "sample_rate": sr}, **kwargs)

    # pyannote 4.x returns a DiarizeOutput; 3.x returned a bare Annotation.
    # Prefer the exclusive variant, which pyannote documents as "adapted to
    # downstream transcription" because it strips overlapping speech turns.
    # With overlaps left in we would transcribe the same audio twice and
    # attribute it to two different people.
    ann = None
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        cand = getattr(dia_out, attr, None)
        if cand is not None and hasattr(cand, "itertracks"):
            ann = cand
            print(f"using {attr}")
            break
    if ann is None:
        if not hasattr(dia_out, "itertracks"):
            die(f"unexpected diarization return type: {type(dia_out).__name__}")
        ann = dia_out

    turns = [{"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": spk}
             for seg, _, spk in ann.itertracks(yield_label=True)]
    turns.sort(key=lambda t: t["start"])

    # Merge same-speaker turns split by a breath, and drop specks that are
    # almost always crosstalk rather than speech.
    merged = []
    for t in turns:
        if merged and t["speaker"] == merged[-1]["speaker"] and \
                t["start"] - merged[-1]["end"] <= args.merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append(dict(t))
    merged = [t for t in merged if t["end"] - t["start"] >= args.min_turn]

    speakers = sorted({t["speaker"] for t in merged})
    talk = {s: round(sum(t["end"] - t["start"] for t in merged if t["speaker"] == s), 1)
            for s in speakers}
    result = {
        "turns": merged,
        "speakers": speakers,
        "speaker_count": len(speakers),
        # Talk time is how you map anonymous labels onto real names: the person
        # who ran the meeting almost always dominates it.
        "talk_time_sec": talk,
        "params": {"min_turn": args.min_turn, "merge_gap": args.merge_gap, **kwargs},
    }
    save(result, out)
    print(f"{len(merged)} turns, {len(speakers)} speakers, talk time: {talk}")


# ----------------------------------------------------------------- transcribe


def read_wav_slice(path, start, end):
    import numpy as np
    with wave.open(str(path)) as w:
        sr, n = w.getframerate(), w.getnframes()
        a = max(0, int(start * sr))
        b = min(n, int(end * sr))
        if b <= a:
            return None, sr
        w.setpos(a)
        raw = w.readframes(b - a)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def silence_segments(audio, max_len, min_len, noise="-32dB", min_sil=0.35):
    """Split audio at pauses using ffmpeg silencedetect.

    Used only when diarization is unavailable. Aligning cuts to pauses keeps
    sentences (and therefore languages) intact, which blind fixed windows do not.
    """
    with wave.open(str(audio)) as w:
        dur = w.getnframes() / w.getframerate()
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(audio),
         "-af", f"silencedetect=noise={noise}:d={min_sil}", "-f", "null", "-"],
        capture_output=True, text=True)
    sil = []
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start: ([0-9.\-]+)", proc.stderr)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end: ([0-9.\-]+)", proc.stderr)]
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else dur
        sil.append((max(0.0, s), min(dur, e)))

    # Speech lives in the gaps between silences.
    speech, cur = [], 0.0
    for s, e in sil:
        if s - cur > 0.05:
            speech.append([cur, s])
        cur = e
    if dur - cur > 0.05:
        speech.append([cur, dur])
    if not speech:
        speech = [[0.0, dur]]

    # Whisper degrades past ~30s of context, so break over-long runs.
    out = []
    for s, e in speech:
        if e - s <= max_len:
            out.append([s, e])
            continue
        n = int((e - s) // max_len) + 1
        step = (e - s) / n
        out.extend([[s + i * step, s + (i + 1) * step] for i in range(n)])

    return [{"start": round(s, 3), "end": round(e, 3), "speaker": "UNKNOWN"}
            for s, e in out if e - s >= min_len]


def _norm_words(text):
    return [w for w in re.split(r"[^\w֐-׿]+", text.lower()) if w]


def prompt_echo_ratio(text, terms):
    """Share of a segment's words that came from the glossary prompt.

    Whisper echoes its initial_prompt verbatim on low-content audio. That echo
    reads as real speech and would otherwise land in the meeting notes as a
    quote nobody said.
    """
    if not terms:
        return 0.0
    words = _norm_words(text)
    if not words:
        return 0.0
    vocab = set()
    for t in terms:
        vocab.update(_norm_words(t))
    return sum(1 for w in words if w in vocab) / len(words)


def _asr(pcm, model, prompt, language=None):
    import mlx_whisper
    return mlx_whisper.transcribe(
        pcm,
        path_or_hf_repo=model,
        task="transcribe",              # NEVER translate. See SKILL.md.
        language=language,              # None = detect this slice on its own
        initial_prompt=prompt,
        condition_on_previous_text=False,  # stops runaway repetition
        temperature=0.0,
    )


def asr_guarded(pcm, sr, model, prompt, language, args):
    """Transcribe, and refuse to accept a prompt that ate the transcript.

    MEASURED: passing a glossary as initial_prompt cut a 249-character result
    down to 65 characters, dropping 7 of 8 sentences, because the prompt pushed
    the decoder to an early end-of-text. So glossary biasing is opt-in only, and
    even then we re-run clean and keep whichever transcript is richer. Never
    trade transcript completeness for spelling.
    """
    r = _asr(pcm, model, prompt, language)
    if not prompt:
        return r, False
    secs = max(len(pcm) / sr, 0.01)
    rate = len(r.get("text") or "") / secs
    if rate >= args.min_char_rate:
        return r, False
    clean = _asr(pcm, model, None, language)
    if len(clean.get("text") or "") > len(r.get("text") or ""):
        return clean, True
    return r, False


_LANG_MODEL = {}


def detect_language_fast(pcm, model_repo):
    """Language ID from one encoder pass. Returns (lang, confidence).

    MEASURED at ~350ms per call against multiple seconds for a full decode, and
    still 0.99 confident on a 1.2 second clip. Cheap enough to run on every
    sentence, which is what makes catching mid-turn language switches practical.
    """
    import mlx.core as mx
    from mlx_whisper import audio as wa, decoding, load_models

    m = _LANG_MODEL.get(model_repo)
    if m is None:
        m = _LANG_MODEL[model_repo] = load_models.load_model(model_repo)
    mel = wa.log_mel_spectrogram(wa.pad_or_trim(mx.array(pcm), wa.N_SAMPLES),
                                 n_mels=m.dims.n_mels)
    _, probs = decoding.detect_language(m, mel)
    probs = probs if isinstance(probs, dict) else probs[0]
    lang, p = max(probs.items(), key=lambda kv: kv[1])
    return lang, float(p)


def cmd_transcribe(args):
    import mlx_whisper

    wd = workdir(args)
    out = wd / "segments.json"
    audio = wd / "audio.wav"
    if not audio.exists():
        die("audio.wav not found; run `mnotes audio` first")

    prompt, prompt_terms = None, []
    if args.hotwords:
        p = Path(args.hotwords)
        if not p.exists():
            die(f"hotwords file not found: {p}")
        terms = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        # Feeding the glossary in as an initial prompt biases the decoder toward
        # correct spellings. Preventing the error beats correcting it later.
        #
        # Two guards, both learned the hard way: a bare comma-joined list gets
        # echoed back verbatim as if someone had said it, and a long prompt
        # echoes more often. So phrase it as prose and cap the length.
        terms = terms[: args.max_hotwords]
        prompt_terms = terms
        prompt = "The following terms may come up: " + ", ".join(terms) + "."
        print(f"biasing decoder with {len(terms)} glossary terms "
              f"(opt-in; truncation guard active)")

    dia_path = wd / "diarization.json"
    if dia_path.exists():
        turns = load(dia_path, "diarization")["turns"]
        print(f"transcribing {len(turns)} diarized turns")
    else:
        # No diarization available. Cut on silence rather than on a blind clock:
        # a fixed grid slices sentences in half, and a half sentence in the
        # wrong language gets mis-detected and silently translated.
        turns = silence_segments(audio, args.window, args.min_turn)
        print(f"no diarization found; using {len(turns)} silence-aligned segments "
              f"(speakers will be UNKNOWN)")

    # Existing partial run? Resume it.
    done = {}
    if out.exists() and not args.force:
        prev = json.loads(out.read_text())
        for s in prev.get("segments", []):
            done[(round(s["start"], 2), round(s["end"], 2))] = s
        if done:
            print(f"resuming: {len(done)} segments already transcribed")

    segments = []
    langs = {}
    n_recovered = 0
    n_corrected = 0
    for i, t in enumerate(turns, 1):
        key = (round(t["start"], 2), round(t["end"], 2))
        if key in done:
            segments.append(done[key])
            langs[done[key].get("language")] = langs.get(done[key].get("language"), 0) + 1
            continue
        dur = t["end"] - t["start"]
        if dur < args.min_turn:
            continue
        pcm, sr = read_wav_slice(audio, t["start"], t["end"])
        if pcm is None or len(pcm) < 1600:
            continue
        try:
            r, recovered = asr_guarded(pcm, sr, args.model, prompt, args.language, args)
        except Exception as e:
            print(f"  turn {i} failed ({e}); skipping", file=sys.stderr)
            continue
        if recovered:
            print(f"  turn {i}: glossary prompt truncated output; used clean decode")
            n_recovered += 1

        text = (r.get("text") or "").strip()
        if not text:
            continue

        turn_lang = r.get("language")
        # Always split a turn on Whisper's own sentence boundaries. Two reasons:
        # a 15 second cue is useless as a subtitle, and sentence-level
        # timestamps are what let every claim in the summary cite the video.
        subs = [s for s in (r.get("segments") or [])
                if s["end"] - s["start"] >= args.min_sub and (s.get("text") or "").strip()]
        if not subs:
            subs = [{"start": 0.0, "end": dur, "text": text,
                     "no_speech_prob": 0.0, "compression_ratio": 0.0}]

        verify = (not args.language) and (not args.no_verify_language)
        for s in subs:
            stext = (s.get("text") or "").strip()
            if not stext:
                continue
            s_start, s_end = s["start"], min(s["end"], dur)
            s_dur = s_end - s_start
            lang, conf, corrected = turn_lang, None, False

            # Per-sentence language check. This is what catches a speaker
            # switching language mid-turn, which turn-level detection cannot see
            # and which otherwise gets silently translated away.
            if verify and s_dur >= args.min_sub:
                a, b = int(s_start * sr), int(s_end * sr)
                clip = pcm[a:b]
                if len(clip) >= int(0.5 * sr):
                    try:
                        dlang, conf = detect_language_fast(clip, args.model)
                        if dlang != turn_lang and conf >= args.lang_conf:
                            # Re-decode this sentence only, in its real language.
                            rr = _asr(clip, args.model, None, dlang)
                            rtext = (rr.get("text") or "").strip()
                            if rtext:
                                stext, lang, corrected = rtext, dlang, True
                    except Exception as e:
                        print(f"  turn {i}: language check failed ({e})", file=sys.stderr)

            nsp = s.get("no_speech_prob", 0) or 0
            cr = s.get("compression_ratio", 0) or 0
            echo = prompt_echo_ratio(stext, prompt_terms)

            seg = {
                "start": round(t["start"] + s_start, 2),
                "end": round(t["start"] + s_end, 2),
                "speaker": t.get("speaker", "UNKNOWN"),
                "text": stext,
                "language": lang,
                "rtl": is_rtl(stext),
                "no_speech_prob": round(nsp, 3),
                "compression_ratio": round(cr, 2),
            }
            if conf is not None:
                seg["language_confidence"] = round(conf, 3)
            if corrected:
                seg["language_corrected"] = True
                n_corrected += 1
            if echo >= args.echo_max and len(_norm_words(stext)) <= 40:
                seg["suspect"] = "glossary_prompt_echo"
                seg["echo_ratio"] = round(echo, 2)
            elif nsp > args.no_speech_max or cr > 2.6:
                # Whisper invents fluent text on near-silence and loops on music.
                seg["suspect"] = "likely_hallucination"
            if s_dur < 0.8 and conf is None:
                seg["suspect_language"] = True
            segments.append(seg)
            langs[lang] = langs.get(lang, 0) + 1

        if i % 10 == 0 or i == len(turns):
            print(f"  {i}/{len(turns)} turns, {len(segments)} cues")
            save({"segments": segments, "model": args.model}, out)  # checkpoint

    # Very short turns get their language from their neighbours rather than
    # from an unreliable one-word detection.
    dominant = max(langs, key=langs.get) if langs else None
    for i, s in enumerate(segments):
        if s.pop("suspect_language", False):
            nb = [segments[j]["language"] for j in (i - 1, i + 1)
                  if 0 <= j < len(segments)]
            s["language"] = nb[0] if nb else dominant
            s["language_inferred"] = True

    # Spurious-language guard.
    #
    # MEASURED on a real 6 minute English call: short turns of filler and
    # non-speech came back as confident German and Portuguese ("Obrigado.",
    # "Und dann...") because Whisper hallucinates fluent text on low-content
    # audio and labels it accordingly. Left alone those become "findings" in a
    # summary. A one-off minority language on a short turn is an artefact, not
    # a bilingual moment, so flag it rather than trusting it.
    expected = None
    if args.expect_languages:
        expected = {x.strip().lower() for x in args.expect_languages.split(",") if x.strip()}
    n_flagged = 0
    for s in segments:
        lang = s.get("language")
        if not lang or lang == dominant or s.get("suspect"):
            continue
        dur = s["end"] - s["start"]
        conf = s.get("language_confidence")
        rare = langs.get(lang, 0) <= args.minority_max_count
        if expected is not None and lang not in expected:
            s["suspect"] = "unexpected_language"
        elif dur < args.minority_min_dur and rare:
            s["suspect"] = "minority_language_short_turn"
        elif conf is not None and conf < args.minority_min_conf and rare:
            s["suspect"] = "minority_language_low_confidence"
        else:
            continue
        n_flagged += 1

    switches = sum(1 for a, b in zip(segments, segments[1:])
                   if a["language"] != b["language"])
    result = {
        "segments": segments,
        "model": args.model,
        "languages": langs,
        "dominant_language": dominant,
        "language_switches": switches,
        "suspect_count": sum(1 for s in segments if s.get("suspect")),
    }
    save(result, out)
    result["prompt_truncations_recovered"] = n_recovered
    result["language_corrected_segments"] = n_corrected
    result["spurious_language_flagged"] = n_flagged
    if expected is not None:
        result["expected_languages"] = sorted(expected)
    save(result, out)
    print(f"{len(segments)} segments | languages {langs} | "
          f"{switches} language switches | {result['suspect_count']} suspect"
          + (f" | {n_recovered} prompt truncations recovered" if n_recovered else "")
          + (f" | {n_corrected} sentences re-decoded in another language" if n_corrected else "")
          + (f" | {n_flagged} spurious-language segments flagged" if n_flagged else ""))


def _frange(start, stop, step):
    x = start
    while x < stop:
        yield round(x, 3)
        x += step


# ----------------------------------------------------------------- srt


def ts(sec, comma=True):
    if sec < 0:
        sec = 0
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    ms = int(round((s - int(s)) * 1000))
    if ms == 1000:
        s, ms = int(s) + 1, 0
    sep = "," if comma else "."
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{ms:03d}"


def bidi(text, rtl):
    """Wrap RTL lines so mixed Hebrew/English renders correctly in players.

    Without this, a line containing both scripts displays with the runs in the
    wrong order and looks like a transcription error when it is not.
    """
    if not rtl:
        return text
    return f"{RLM}{RLE}{text}{PDF}"


def cmd_srt(args):
    wd = workdir(args)
    data = load(wd / "segments.json", "segments.json")
    names = {}
    if args.names:
        names = load(args.names, "speaker name map")
    lines, n = [], 0
    for s in data["segments"]:
        if s.get("suspect") and args.drop_suspect:
            continue
        n += 1
        who = names.get(s["speaker"], s["speaker"])
        text = bidi(s["text"], s.get("rtl"))
        label = f"{who}: " if args.speakers_in_srt else ""
        tag = ""
        if args.tag_language and s.get("language") and \
                s["language"] != data.get("dominant_language"):
            tag = f" [{s['language']}]"
        lines.append(f"{n}\n{ts(s['start'])} --> {ts(s['end'])}\n{label}{text}{tag}\n")
    out = wd / "transcript.srt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({n} cues)")

    # A speaker-labelled plain transcript is what the summarizing pass reads.
    md = [f"[{ts(s['start'], comma=False)[:-4]}] "
          f"{names.get(s['speaker'], s['speaker'])}"
          f"{' (' + s['language'] + ')' if s.get('language') != data.get('dominant_language') else ''}: "
          f"{s['text']}"
          + ("  <!-- SUSPECT: low confidence -->" if s.get("suspect") else "")
          for s in data["segments"]]
    (wd / "transcript.md").write_text("\n\n".join(md), encoding="utf-8")
    print(f"wrote {wd / 'transcript.md'}")


# ----------------------------------------------------------------- engagement


QUESTION_HINT = re.compile(r"\?|^(what|why|how|when|who|should|can|do|does|did|is|are)\b", re.I)
AGREE_HINT = re.compile(
    r"\b(agree|exactly|right|makes sense|good point|disagree|but|however|"
    r"concern|worried|blocker|actually|no,|wait)\b", re.I)
DECISION_HINT = re.compile(
    r"\b(decide|decided|decision|let's|we should|action item|owner|deadline|"
    r"by (monday|tuesday|wednesday|thursday|friday|next week)|ship|launch)\b", re.I)


def cmd_engagement(args):
    wd = workdir(args)
    data = load(wd / "segments.json", "segments.json")
    segs = [s for s in data["segments"] if not s.get("suspect")]
    if not segs:
        die("no usable segments")
    end = max(s["end"] for s in segs)
    keywords = []
    if args.keywords:
        keywords = [k.strip().lower() for k in Path(args.keywords).read_text().split("\n") if k.strip()] \
            if Path(args.keywords).exists() else [k.strip().lower() for k in args.keywords.split(",")]

    win, step = args.window, args.step
    rows = []
    for start in _frange(0, max(end - win, 0.1), step):
        stop = start + win
        inside = [s for s in segs if s["end"] > start and s["start"] < stop]
        if not inside:
            continue
        words = sum(len(s["text"].split()) for s in inside)
        speakers = {s["speaker"] for s in inside}
        turns = sum(1 for a, b in zip(inside, inside[1:]) if a["speaker"] != b["speaker"])
        text = " ".join(s["text"] for s in inside)
        q = len(QUESTION_HINT.findall(text))
        a = len(AGREE_HINT.findall(text))
        d = len(DECISION_HINT.findall(text))
        kw = sum(text.lower().count(k) for k in keywords) if keywords else 0
        rows.append({
            "start": start, "end": round(stop, 2),
            "words": words, "wpm": round(words / win * 60, 1),
            "speakers": len(speakers), "turn_changes": turns,
            "questions": q, "stance_markers": a, "decision_markers": d,
            "keyword_hits": kw,
        })

    def norm(key):
        vals = [r[key] for r in rows]
        lo, hi = min(vals), max(vals)
        return (lambda v: 0.0) if hi == lo else (lambda v: (v - lo) / (hi - lo))

    fns = {k: norm(k) for k in
           ("wpm", "turn_changes", "questions", "stance_markers", "decision_markers", "keyword_hits")}
    # Turn-taking and stance markers are what separate a real working discussion
    # from one person reading slides aloud, so they carry the most weight.
    weights = {"turn_changes": 0.28, "stance_markers": 0.22, "decision_markers": 0.18,
               "questions": 0.14, "keyword_hits": 0.12, "wpm": 0.06}
    for r in rows:
        r["score"] = round(sum(w * fns[k](r[k]) for k, w in weights.items()), 4)

    ranked = [r for r in sorted(rows, key=lambda r: -r["score"])
              if r["score"] >= args.min_score]
    picked = []
    for r in ranked:
        # Non-overlapping peaks, so the top N are N different moments.
        if all(abs(r["start"] - p["start"]) >= args.separation for p in picked):
            picked.append(r)
        if len(picked) >= args.top:
            break
    picked.sort(key=lambda r: r["start"])
    for p in picked:
        p["hms"] = ts(p["start"], comma=False)[:-4]

    save({"windows": rows, "top": picked,
          "params": {"window": win, "step": step, "weights": weights}},
         wd / "windows.json")
    print(f"{len(rows)} windows scored; top {len(picked)}:")
    for p in picked:
        print(f"  {p['hms']}  score={p['score']:.3f}  turns={p['turn_changes']} "
              f"q={p['questions']} stance={p['stance_markers']} dec={p['decision_markers']}")


# ----------------------------------------------------------------- frames


_HAS_DRAWTEXT = None


def has_drawtext():
    """Homebrew ffmpeg is often built without libfreetype, so drawtext is absent.

    Tile numbering is a nicety; losing it must not cost us the contact sheet.
    """
    global _HAS_DRAWTEXT
    if _HAS_DRAWTEXT is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                 capture_output=True, text=True).stdout
            _HAS_DRAWTEXT = bool(re.search(r"^\s*[.TSC]+\s+drawtext\s", out, re.M))
        except Exception:
            _HAS_DRAWTEXT = False
    return _HAS_DRAWTEXT


def gray_stats(jpg):
    """Laplacian-ish variance, edge density, and a coarse visual signature.

    Slides and shared screens are full of hard text edges; a grid of webcam
    faces is comparatively smooth. This separates them for free, with no vision
    tokens and no extra dependencies.

    Returns (stats_dict, signature) where signature is a 16x16 downsample used
    to reject near-duplicate frames. The signature is deliberately not stored in
    frames.json; it is working data, not a result.
    """
    import numpy as np
    blank = {"sharpness": 0.0, "edge_density": 0.0, "brightness": 0.0}
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(jpg), "-vf", "scale=320:-1",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            check=True, capture_output=True).stdout
    except subprocess.CalledProcessError:
        return blank, None
    w = 320
    h = len(raw) // w
    if h < 16:
        return blank, None
    a = np.frombuffer(raw[:w * h], dtype=np.uint8).reshape(h, w).astype(np.float32)
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    # 16x16 block means, cheap and robust to compression noise.
    bh, bw = h // 16, w // 16
    sig = a[:bh * 16, :bw * 16].reshape(16, bh, 16, bw).mean(axis=(1, 3))
    return {
        "sharpness": round(float(lap.var()), 1),
        "edge_density": round(float((np.abs(lap) > 24).mean()), 4),
        "brightness": round(float(a.mean()), 1),
    }, sig


def gray_array(path, width=480):
    """Decode an image to a 2D float32 gray array via ffmpeg. No PIL needed."""
    import numpy as np
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={width}:-1",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            check=True, capture_output=True).stdout
    except subprocess.CalledProcessError:
        return None
    h = len(raw) // width
    if h < 8:
        return None
    return np.frombuffer(raw[:width * h], dtype=np.uint8).reshape(h, width).astype(np.float32)


def _components(mask):
    """Every connected blob in a small boolean grid, as (bbox, cells).

    Plain bbox-of-all-changed-blocks is no good: one flicker in a corner plus a
    real change elsewhere yields a box covering the whole frame. So find the
    blobs separately and let the caller choose between them.
    """
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    out = []
    if not rows or not cols:
        return out
    seen = [[False] * cols for _ in range(rows)]
    for r0 in range(rows):
        for c0 in range(cols):
            if not mask[r0][c0] or seen[r0][c0]:
                continue
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            r_lo = r_hi = r0
            c_lo = c_hi = c0
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                r_lo, r_hi = min(r_lo, r), max(r_hi, r)
                c_lo, c_hi = min(c_lo, c), max(c_hi, c)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and mask[rr][cc] and not seen[rr][cc]:
                        seen[rr][cc] = True
                        stack.append((rr, cc))
            out.append(((r_lo, r_hi, c_lo, c_hi), cells))
    return out


def change_box(ref, cur, block_thresh=6.0, grid=(24, 40), pad=0.05,
               min_frac=0.22, whole_frame_frac=0.55):
    """Locate what changed between two frames, as normalized (x, y, w, h).

    Returns (box, changed_fraction) where box is None when the change covers so
    much of the frame that cropping would lose context (a slide change), or
    when nothing changed at all.

    This is the fix for small text deltas on a shared screen: a new Slack
    message is a few percent of a 1728x1118 frame, illegible once the full
    screenshot is scaled into a document. Cropping to the delta makes it
    readable and makes near-identical frames genuinely distinguishable.
    """
    import numpy as np
    if ref is None or cur is None or ref.shape != cur.shape:
        return None, 0.0
    h, w = cur.shape
    gr, gc = grid
    bh, bw = max(h // gr, 1), max(w // gc, 1)
    rows, cols = h // bh, w // bw
    if rows < 2 or cols < 2:
        return None, 0.0
    diff = np.abs(cur - ref)[:rows * bh, :cols * bw]
    blocks = diff.reshape(rows, bh, cols, bw).mean(axis=(1, 3))
    mask = blocks > block_thresh
    frac = float(mask.mean())
    if frac <= 0.0005:
        return None, frac
    if frac >= whole_frame_frac:
        return None, frac          # everything moved; keep the full frame

    # Biggest changed region wins.
    #
    # NOT attempted here on purpose: preferring "text-like" regions over webcam
    # movement. A moving head is often the largest change, so crop sheets do
    # sometimes surface a face instead of content. Edge density was measured as
    # a discriminator and it does not work at any analysis resolution: at 480px
    # a webcam tile scored 4.40 against content's 2.75 (backwards), at 960px
    # 3.00 against 3.60 (correct), and at 1440px 2.15 against 2.22 (no
    # separation). Small text blurs on downsample while a webcam tile keeps its
    # border and label edges, so the signal inverts with scale.
    #
    # Rejecting frames automatically would therefore throw away real content.
    # The contact sheet exists so a vision pass makes this call cheaply, on one
    # image. Do not replace that with a heuristic that only looks principled.
    comps = _components(mask.tolist())
    if not comps:
        return None, frac
    best = max(comps, key=lambda bc: len(bc[1]))[0]
    r_lo, r_hi, c_lo, c_hi = best
    y0 = (r_lo * bh) / h
    y1 = ((r_hi + 1) * bh) / h
    x0 = (c_lo * bw) / w
    x1 = ((c_hi + 1) * bw) / w

    # Pad, then grow anything too small to read.
    x0, x1 = max(0.0, x0 - pad), min(1.0, x1 + pad)
    y0, y1 = max(0.0, y0 - pad), min(1.0, y1 + pad)
    if x1 - x0 < min_frac:
        cx = (x0 + x1) / 2
        x0, x1 = max(0.0, cx - min_frac / 2), min(1.0, cx + min_frac / 2)
    if y1 - y0 < min_frac:
        cy = (y0 + y1) / 2
        y0, y1 = max(0.0, cy - min_frac / 2), min(1.0, cy + min_frac / 2)

    return {"x": round(x0, 4), "y": round(y0, 4),
            "w": round(x1 - x0, 4), "h": round(y1 - y0, 4)}, frac


def px_box(box, width, height, even=True):
    """Normalized box to integer pixel crop values for ffmpeg."""
    x = int(box["x"] * width)
    y = int(box["y"] * height)
    w = max(16, int(box["w"] * width))
    h = max(16, int(box["h"] * height))
    w = min(w, width - x)
    h = min(h, height - y)
    if even:                      # h264 and some encoders dislike odd sizes
        w -= w % 2
        h -= h % 2
    return x, y, max(w, 16), max(h, 16)


def too_similar(sig, kept_sigs, threshold):
    """True when this frame is visually a near-duplicate of one already kept.

    MEASURED on a real call: a 12 tile contact sheet came back as 12 near
    identical screenshots of the same Slack window, roughly 3 distinct views
    across 12 slots. Timestamp spacing does not catch that, because a shared
    screen barely changes for minutes at a time.
    """
    import numpy as np
    if sig is None:
        return False
    for k in kept_sigs:
        if k is not None and float(np.abs(sig - k).mean()) < threshold:
            return True
    return False


def cmd_frames(args):
    wd = workdir(args)
    media = load(wd / "media.json", "media.json")
    if not media["has_video"]:
        print("audio-only recording; no frames to extract")
        save({"candidates": [], "sheets": []}, wd / "frames.json")
        return
    wins = load(wd / "windows.json", "windows.json")["top"]
    fdir = wd / "frames"
    if fdir.exists() and args.force:
        shutil.rmtree(fdir)
    fdir.mkdir(exist_ok=True)
    dur = media["duration_sec"]

    candidates, sheets = [], []
    for wi, win in enumerate(wins, 1):
        centre = win["start"] + (win["end"] - win["start"]) / 2
        lo, hi = max(0, centre - args.radius), min(dur, centre + args.radius)

        # Scene changes first: a frame where the deck advanced is worth far more
        # than an arbitrary tick of a 5 second grid.
        scene_ts = []
        try:
            proc = subprocess.run(
                ["ffmpeg", "-v", "info", "-ss", str(lo), "-t", str(hi - lo),
                 "-i", media["path"],
                 "-vf", f"select='gt(scene,{args.scene})',metadata=print",
                 "-an", "-f", "null", "-"],
                capture_output=True, text=True)
            for m in re.finditer(r"pts_time:([0-9.]+)", proc.stderr):
                scene_ts.append(round(lo + float(m.group(1)), 2))
        except Exception:
            pass

        grid = [round(t, 2) for t in _frange(lo, hi, args.interval)]
        # Dedupe: a scene change already covers a nearby grid tick.
        stamps = sorted(scene_ts)
        for g in grid:
            if all(abs(g - s) > args.interval / 2 for s in stamps):
                stamps.append(g)
        stamps = sorted(stamps)[: args.max_per_window]

        rows = []
        for t in stamps:
            name = fdir / f"w{wi:02d}_{int(t * 1000):09d}.jpg"
            try:
                run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t),
                     "-i", media["path"], "-frames:v", "1",
                     "-vf", f"scale={args.thumb}:-1", "-q:v", "4", str(name)])
            except subprocess.CalledProcessError:
                continue
            st, sig = gray_stats(name)
            rows.append({
                "window": wi, "t": t, "hms": ts(t, comma=False)[:-4],
                "file": str(name), "scene_change": t in scene_ts, **st,
                "_sig": sig,
            })

        if not rows:
            continue
        # Rank: text-heavy and sharp beats smooth talking heads; a genuine
        # scene change gets a thumb on the scale.
        for r in rows:
            # The scene-change bonus only applies to a frame that has something
            # on it. Otherwise a cut *to* a blank screen outranks a real slide.
            scene_bonus = 12 if (r["scene_change"] and r["edge_density"] >= 0.012) else 0
            r["visual_score"] = round(
                r["edge_density"] * 100 + r["sharpness"] / 200 + scene_bonus
                - (20 if r["brightness"] < 18 or r["brightness"] > 242 else 0), 2)
        rows.sort(key=lambda r: -r["visual_score"])
        # Walk best-first, skipping anything that looks like a frame we already
        # have. Better to return 4 distinct views than 12 copies of one.
        keep, kept_sigs = [], []
        for r in rows:
            if too_similar(r["_sig"], kept_sigs, args.dedupe):
                continue
            keep.append(r)
            kept_sigs.append(r["_sig"])
            if len(keep) >= args.per_sheet:
                break
        dropped = len(rows) - len(keep)
        for r in rows:
            r.pop("_sig", None)          # never serialize the signature
        keep.sort(key=lambda r: r["t"])
        if dropped:
            print(f"  window {wi}: {len(keep)} distinct frames "
                  f"({dropped} near-duplicates dropped)")

        # Work out what actually changed in each kept frame.
        #
        # Reference against the nearest slightly-earlier frame from the sampling
        # grid, NOT against the previous frame we kept. Kept frames can be 30
        # seconds apart, by which point a scrolling screen share has changed
        # almost everywhere and the "change region" degenerates to the whole
        # frame. A few seconds back isolates the actual delta, which is the
        # point: a new message or an edited line is a few percent of a 1728px
        # wide frame and unreadable once the full screenshot is scaled down.
        if not args.no_change_box:
            gap = args.change_ref_gap or max(6.0, args.interval * 2.5)
            by_t = sorted((r["t"], r["file"]) for r in rows)
            cache = {}

            def arr(path):
                if path not in cache:
                    cache[path] = gray_array(path, args.thumb)
                return cache[path]

            for r in keep:
                ref = None
                for t2, f2 in by_t:
                    if t2 >= r["t"] - 0.01:
                        break
                    if r["t"] - t2 <= gap:
                        ref = f2            # ascending, so this ends up nearest
                if ref is None:
                    continue
                box, frac = change_box(arr(ref), arr(r["file"]),
                                       block_thresh=args.change_thresh,
                                       pad=args.change_pad)
                r["changed_fraction"] = round(frac, 4)
                r["change_ref_t"] = next(t for t, f in by_t if f == ref)
                if box:
                    r["change_box"] = box

        candidates.extend(keep)

        # One contact sheet per window: vision reviews N frames in a single
        # image instead of N separate calls.
        sheet = wd / f"contact_sheet_w{wi:02d}.jpg"
        lst = fdir / f"sheet_w{wi:02d}.txt"
        lst.write_text("".join(f"file '{r['file']}'\nduration 1\n" for r in keep))
        cols = min(args.cols, len(keep))
        rowsn = (len(keep) + cols - 1) // cols
        label = ("drawtext=text='%{n}':x=6:y=6:fontsize=28:fontcolor=yellow:"
                 "box=1:boxcolor=black@0.6," if has_drawtext() else "")
        try:
            run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                 "-i", str(lst),
                 "-vf", f"scale={args.thumb}:-1,{label}"
                        f"tile={cols}x{rowsn}:padding=6:color=white",
                 "-frames:v", "1", str(sheet)])
            sheets.append({
                "window": wi, "file": str(sheet),
                "centre_hms": ts(centre, comma=False)[:-4],
                "grid": f"{cols}x{rowsn}", "numbered": has_drawtext(),
                # Tiles fill the grid left to right, top to bottom. This mapping
                # is how the caller turns "tile 7 looks good" into a timestamp,
                # and it is the only way when drawtext is unavailable.
                "tiles": [{"index": i + 1, "row": i // cols + 1, "col": i % cols + 1,
                           "t": r["t"], "hms": r["hms"], "file": r["file"]}
                          for i, r in enumerate(keep)]})
        except subprocess.CalledProcessError as e:
            print(f"  contact sheet for window {wi} failed: {e.stderr[:200]}", file=sys.stderr)

        # Second sheet, cropped to the changed region of each frame. This is the
        # one to review when the screen share barely moves, because the delta is
        # what carries the information and it is legible here.
        cropped = [r for r in keep if r.get("change_box")]
        if cropped and not args.no_change_box:
            # Even dimensions, and force_divisible_by so the scaled result can
            # never round a pixel over the pad target. Without it ffmpeg fails
            # with "Padded dimensions cannot be smaller than input dimensions".
            cw = args.crop_tile - (args.crop_tile % 2)
            ch = int(cw * 0.62)
            ch -= ch % 2
            files = []
            for r in cropped:
                x, y, bw2, bh2 = px_box(r["change_box"], media["width"], media["height"])
                cf = fdir / f"crop_w{wi:02d}_{int(r['t'] * 1000):09d}.jpg"
                try:
                    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(r["t"]),
                         "-i", media["path"], "-frames:v", "1",
                         "-vf", f"crop={bw2}:{bh2}:{x}:{y},"
                                f"scale={cw}:{ch}:force_original_aspect_ratio=decrease"
                                f":force_divisible_by=2,"
                                f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color=white",
                         "-q:v", "2", str(cf)])
                    r["crop_file"] = str(cf)
                    files.append(cf)
                except subprocess.CalledProcessError as e:
                    # Never swallow this silently; a quiet skip here once hid a
                    # filter bug that dropped every crop.
                    print(f"  crop at {r['hms']} failed: "
                          f"{e.stderr.strip().splitlines()[-1][:160]}", file=sys.stderr)
                    continue
            if files:
                csheet = wd / f"contact_sheet_w{wi:02d}_crops.jpg"
                clst = fdir / f"csheet_w{wi:02d}.txt"
                clst.write_text("".join(f"file '{f}'\nduration 1\n" for f in files))
                ccols = min(args.cols, len(files))
                crows = (len(files) + ccols - 1) // ccols
                try:
                    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                         "-safe", "0", "-i", str(clst),
                         "-vf", f"tile={ccols}x{crows}:padding=6:color=white",
                         "-frames:v", "1", str(csheet)])
                    sheets[-1]["crop_sheet"] = str(csheet)
                    sheets[-1]["crop_tiles"] = [
                        {"index": i + 1, "row": i // ccols + 1, "col": i % ccols + 1,
                         "t": r["t"], "hms": r["hms"], "box": r["change_box"],
                         "changed_fraction": r.get("changed_fraction")}
                        for i, r in enumerate(cropped) if r.get("crop_file")]
                    print(f"      + crop sheet: {csheet} ({len(files)} regions)")
                except subprocess.CalledProcessError as e:
                    print(f"  crop sheet for window {wi} failed: {e.stderr[:200]}",
                          file=sys.stderr)

    save({"candidates": candidates, "sheets": sheets}, wd / "frames.json")
    print(f"{len(candidates)} candidate frames across {len(sheets)} contact sheets")
    if sheets and not sheets[0]["numbered"]:
        print("note: this ffmpeg has no drawtext filter, so tiles are unnumbered. "
              "Use the row/col mapping in frames.json to identify a tile.")
    for s in sheets:
        print(f"  {s['file']}  grid={s['grid']} ({len(s['tiles'])} tiles, "
              f"around {s['centre_hms']})")
        for t in s["tiles"]:
            print(f"      r{t['row']}c{t['col']}  {t['hms']}")


def cmd_cut(args):
    wd = workdir(args)
    media = load(wd / "media.json", "media.json")
    outdir = wd / "selected"
    outdir.mkdir(exist_ok=True)
    t = float(args.t)
    out = outdir / (args.name or f"frame_{int(t * 1000):09d}.png")

    # Reuse the change box this timestamp already has, unless one was given.
    box = None
    if args.box:
        try:
            x, y, w, h = (float(v) for v in args.box.split(","))
            box = {"x": x, "y": y, "w": w, "h": h}
        except ValueError:
            die("--box must be four comma-separated fractions: x,y,w,h")
    elif args.crop or args.highlight:
        fj = wd / "frames.json"
        if fj.exists():
            for c in json.loads(fj.read_text()).get("candidates", []):
                if abs(c["t"] - t) < 0.35 and c.get("change_box"):
                    box = c["change_box"]
                    break
        if box is None:
            print("no change box known for this timestamp; using the full frame")

    vf = []
    if box:
        x, y, w, h = px_box(box, media["width"], media["height"])
        if args.crop:
            vf.append(f"crop={w}:{h}:{x}:{y}")
        if args.highlight:
            # Drawn before any crop so the box stays visible in context.
            vf.insert(0, f"drawbox=x={x}:y={y}:w={w}:h={h}:"
                         f"color={args.highlight_color}:t={args.highlight_width}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t),
           "-i", media["path"], "-frames:v", "1"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd.append(str(out))
    run(cmd)
    what = []
    if box and args.crop:
        what.append("cropped to change region")
    if box and args.highlight:
        what.append("change region highlighted")
    print(f"wrote {out}" + (f" ({', '.join(what)})" if what else ""))


# ----------------------------------------------------------------- tracks

def _decode_mono16k(path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def _name_from_track(path):
    """audioAdamFrimer11707420508.m4a -> 'Adam Frimer' (Zoom per-participant naming)."""
    stem = Path(path).stem
    stem = re.sub(r"^audio", "", stem)
    stem = re.sub(r"\d+$", "", stem)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem).strip() or Path(path).stem


def cmd_tracks(args):
    """Map diarization labels to people using one audio file per participant.

    Zoom's "Record a separate audio file for each participant" setting saves an
    'Audio Record' folder next to the recording. For every diarized turn, the
    participant whose own track is loudest is the speaker. This is platform
    evidence, far stronger than guessing from talk time.
    """
    import numpy as np
    wd = workdir(args)
    out = wd / "names.json"
    if out.exists() and not args.force:
        print(f"{out} exists (use --force to redo)")
        return
    dia = load(wd / "diarization.json", "diarization.json")
    folder = Path(args.folder).expanduser()
    files = sorted(f for f in folder.iterdir()
                   if f.suffix.lower() in (".m4a", ".wav", ".mp3", ".aac", ".flac"))
    if len(files) < 2:
        die(f"need at least two per-participant audio files in {folder}")
    names = [_name_from_track(f) for f in files]
    tracks = [_decode_mono16k(f) for f in files]
    votes = {}
    for t in dia["turns"]:
        s, e = int(t["start"] * 16000), int(t["end"] * 16000)
        energy = [float(np.sqrt(np.mean(tr[s:e] ** 2))) if e <= len(tr) and e > s else 0.0
                  for tr in tracks]
        v = votes.setdefault(t["speaker"], [0.0] * len(files))
        v[int(np.argmax(energy))] += t["end"] - t["start"]
    mapping, report = {}, {}
    for spk, v in sorted(votes.items()):
        total = sum(v) or 1.0
        i = int(np.argmax(v))
        share = v[i] / total
        report[spk] = {"best": names[i], "share": round(share, 3),
                       "seconds": {n: round(x, 1) for n, x in zip(names, v)}}
        if share >= args.min_share:
            mapping[spk] = names[i]
        print(f"  {spk}: {names[i]} ({share:.0%} of talk time)"
              + ("" if share >= args.min_share else "  -> left unnamed, below --min-share"))
    if len(set(mapping.values())) < len(mapping):
        print("  warning: two labels map to the same person; diarization may have split one voice")
    save(mapping, out)
    save(report, wd / "tracks.json")
    print("Show this mapping to the user before it reaches the notes.")


# ----------------------------------------------------------------- pdf / doctor

def cmd_pdf(args):
    script = Path(__file__).resolve().parent / "notes_to_pdf.py"
    sys.exit(subprocess.run([sys.executable, str(script), args.markdown]).returncode)


def cmd_doctor(args):
    import platform
    ok = True
    def line(good, what, fix=""):
        nonlocal ok
        ok = ok and good
        print(("  ok    " if good else "  FAIL  ") + what + ("" if good else f"\n        fix: {fix}"))
    print("install")
    line(platform.system() == "Darwin" and platform.machine() == "arm64",
         f"Apple Silicon Mac ({platform.system()} {platform.machine()})",
         "mlx-whisper needs an Apple Silicon Mac; this machine cannot run transcription")
    line(bool(shutil.which("ffmpeg")), "ffmpeg on PATH", "brew install ffmpeg")
    for mod, pkg in (("mlx_whisper", "mlx-whisper"), ("pyannote.audio", "pyannote.audio"),
                     ("PIL", "pillow"), ("numpy", "numpy"), ("huggingface_hub", "huggingface_hub")):
        try:
            __import__(mod)
            line(True, f"python package {pkg}")
        except Exception:
            line(False, f"python package {pkg}", "./install.sh")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from notes_to_pdf import find_chrome
    line(bool(find_chrome()), "Chrome/Chromium for PDF export",
         "install Google Chrome, or set CHROME=/path/to/chrome in .env")
    print("huggingface (speaker diarization)", flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "hf_doctor.py")])
    sys.exit(0 if ok and r.returncode == 0 else 1)


# ----------------------------------------------------------------- cli


def main():
    p = argparse.ArgumentParser(prog="mnotes", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="./mnotes_work", help="work directory for checkpoints")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("probe"); q.add_argument("media"); q.set_defaults(fn=cmd_probe)

    q = sub.add_parser("audio"); q.add_argument("media")
    q.add_argument("--force", action="store_true"); q.set_defaults(fn=cmd_audio)

    q = sub.add_parser("diarize")
    q.add_argument("--speakers", type=int, help="exact count, when you know it")
    q.add_argument("--min-speakers", type=int)
    q.add_argument("--max-speakers", type=int)
    q.add_argument("--token")
    q.add_argument("--cpu", action="store_true")
    q.add_argument("--min-turn", type=float, default=0.4)
    q.add_argument("--merge-gap", type=float, default=0.6)
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_diarize)

    q = sub.add_parser("transcribe")
    q.add_argument("--model", default=ASR_MODEL)
    q.add_argument("--language", default=None,
                   help="force a language; omit to detect PER SEGMENT (recommended)")
    q.add_argument("--hotwords", help="newline-separated glossary to bias the decoder")
    q.add_argument("--max-hotwords", type=int, default=40,
                   help="cap glossary length; long prompts get echoed more")
    q.add_argument("--window", type=float, default=25.0, help="max segment length")
    q.add_argument("--min-turn", type=float, default=0.35)
    q.add_argument("--no-speech-max", type=float, default=0.75)
    q.add_argument("--echo-max", type=float, default=0.75,
                   help="flag a segment when this share of its words came from the glossary")
    q.add_argument("--no-verify-language", action="store_true",
                   help="skip the intra-turn language re-check (faster, misses mid-turn switches)")
    q.add_argument("--min-sub", type=float, default=0.5,
                   help="shortest sentence-level cue to keep")
    q.add_argument("--expect-languages",
                   help="comma list of expected languages, e.g. 'en,he'. Anything "
                        "else gets flagged as an artefact instead of trusted.")
    q.add_argument("--minority-min-dur", type=float, default=3.0,
                   help="a rare language on a turn shorter than this is treated as an artefact")
    q.add_argument("--minority-min-conf", type=float, default=0.85,
                   help="confidence a rare language needs on a short turn")
    q.add_argument("--minority-max-count", type=int, default=3,
                   help="how few segments make a language 'rare'")
    q.add_argument("--lang-conf", type=float, default=0.6,
                   help="confidence needed before re-decoding a sentence in another language")
    q.add_argument("--min-char-rate", type=float, default=5.0,
                   help="chars/sec below which a glossary-prompted decode is retried clean")
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_transcribe)

    q = sub.add_parser("srt")
    q.add_argument("--names", help="JSON map of SPEAKER_00 -> real name")
    q.add_argument("--speakers-in-srt", action="store_true", default=True)
    q.add_argument("--tag-language", action="store_true", default=True)
    q.add_argument("--drop-suspect", action="store_true")
    q.set_defaults(fn=cmd_srt)

    q = sub.add_parser("engagement")
    q.add_argument("--window", type=float, default=45.0)
    q.add_argument("--step", type=float, default=15.0)
    q.add_argument("--top", type=int, default=6)
    q.add_argument("--separation", type=float, default=90.0)
    q.add_argument("--min-score", type=float, default=0.15,
                   help="ignore windows below this score; stops dead air becoming a peak")
    q.add_argument("--keywords", help="comma list or file of goal keywords")
    q.set_defaults(fn=cmd_engagement)

    q = sub.add_parser("frames")
    q.add_argument("--radius", type=float, default=60.0, help="+/- seconds around each peak")
    q.add_argument("--interval", type=float, default=5.0)
    q.add_argument("--scene", type=float, default=0.3, help="scene-change threshold")
    q.add_argument("--max-per-window", type=int, default=28)
    q.add_argument("--per-sheet", type=int, default=12)
    q.add_argument("--cols", type=int, default=4)
    q.add_argument("--thumb", type=int, default=480)
    q.add_argument("--no-change-box", action="store_true",
                   help="skip change-region detection and the crop sheets")
    q.add_argument("--change-thresh", type=float, default=6.0,
                   help="per-block pixel difference counted as a change")
    q.add_argument("--change-ref-gap", type=float, default=0.0,
                   help="how far back to look for the comparison frame, in "
                        "seconds (0 = auto, 2.5x --interval)")
    q.add_argument("--change-pad", type=float, default=0.05,
                   help="padding around the detected region, as a fraction")
    q.add_argument("--crop-tile", type=int, default=560,
                   help="tile width on the cropped contact sheet")
    q.add_argument("--dedupe", type=float, default=4.0,
                   help="mean-pixel difference below which two frames count as "
                        "duplicates (0 disables)")
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_frames)

    q = sub.add_parser("cut")
    q.add_argument("t")
    q.add_argument("--name")
    q.add_argument("--crop", action="store_true",
                   help="crop to the change region detected for this timestamp")
    q.add_argument("--highlight", action="store_true",
                   help="draw a box around the change region instead of cropping")
    q.add_argument("--highlight-color", default="yellow")
    q.add_argument("--highlight-width", type=int, default=5)
    q.add_argument("--box", help="override region as x,y,w,h fractions of the frame")
    q.set_defaults(fn=cmd_cut)

    q = sub.add_parser("tracks", help="name speakers from per-participant audio files")
    q.add_argument("folder", help="folder of per-participant audio, e.g. Zoom's 'Audio Record'")
    q.add_argument("--min-share", type=float, default=0.8,
                   help="leave a label unnamed unless this share of its talk time agrees")
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_tracks)

    q = sub.add_parser("pdf", help="render notes Markdown to HTML + PDF")
    q.add_argument("markdown")
    q.set_defaults(fn=cmd_pdf)

    q = sub.add_parser("doctor", help="check the install and HuggingFace access")
    q.set_defaults(fn=cmd_doctor)

    args = p.parse_args()
    if args.cmd != "doctor" and not shutil.which("ffmpeg"):
        die("ffmpeg not found on PATH (brew install ffmpeg)")
    args.fn(args)


if __name__ == "__main__":
    main()
