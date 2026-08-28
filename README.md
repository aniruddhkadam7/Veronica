# Smallbird Local

A personal, standalone build of the Smallbird desktop app. It talks directly
to your own OpenAI, Anthropic, or Gemini API key — no account, no
subscription, no SaaS backend, no Supabase, no cloud entitlement system.
This repository is completely independent of the Smallbird SaaS product and
can be built and modified on its own.

## What this is

- A Tauri (Rust + React/TypeScript) desktop app for Windows.
- Veronica: a single voice/text assistant overlay, resume/JD setup analysis,
  and Notes — all calling your configured AI provider directly.
- Speech-to-text via Groq Cloud's Whisper API (`whisper-large-v3-turbo`) —
  requires a Groq API key (Settings -> API Keys) and network access. A local
  sherpa-onnx engine still runs alongside it purely to detect when you've
  stopped speaking (voice-activity/endpoint detection); it never produces
  transcript text itself, only decides which span of audio to send to Groq.
- Text-to-speech via Deepgram Flux (`flux-sienna-en`), streamed over a
  persistent `wss://` session — requires a Deepgram API key (Settings ->
  API Keys) and network access, opt-in via the overlay's "Speak answers
  aloud" toggle. Text streams into the session sentence-by-sentence as the
  LLM answer arrives and audio streams back the same way, so playback
  starts on the first chunk, not after the whole answer finishes. A new
  question or the user speaking again interrupts (barges in on) whatever
  is still being spoken. No local TTS exists and there is no fallback if
  Deepgram is unreachable — that sentence just isn't spoken.
- Local document search/RAG (bundled, offline, no cloud dependency).
- Every API key — your AI provider (OpenAI/Anthropic/Gemini), Groq, and
  Deepgram — is entered in Settings -> API Keys and stored in Windows
  Credential Manager via the `keyring` crate — never in a file, never
  logged, never sent anywhere except directly to the provider it's for.
  This app does not read `.env` files.

## Architecture

```
Smallbird Local (this repo)
        |
        +-- Tauri Desktop App (src/, src-tauri/)
        +-- STT: local VAD/endpoint detection (sidecars/stt-sidecar/, models/stt/)
        |         + Groq Cloud transcription (src-tauri/src/stt/groq.rs)
        +-- TTS: Deepgram Flux, streamed over wss:// (src-tauri/src/tts/) -- opt-in, no local model
        +-- Local RAG / document search (sidecars/rag-lite/)
        +-- Direct AI provider client (src-tauri/src/personal/)
        +-- Your API keys, all of them (Windows Credential Manager, via Settings -> API Keys)
        |
        +-- Windows EXE
```

There is no backend server, no Fly.io, no Supabase, and no billing/
entitlement system anywhere in this repository.

## Prerequisites

- Windows 10/11 (x64)
- [Rust](https://rustup.rs/) (stable toolchain)
- [Node.js](https://nodejs.org/) 18+ and npm
- The [Tauri CLI](https://tauri.app/) prerequisites for Windows (WebView2
  runtime, MSVC build tools) — see https://tauri.app/start/prerequisites/

## Setup

```powershell
npm install
```

### Getting the model and sidecar binaries

`models/` and `sidecars/` are **not tracked in this git repository** — the
VAD model alone is over GitHub's 100MB per-file limit, and none of these are
source files anyway (they're prebuilt binary build inputs, not something
you'd normally edit). You need to place them yourself before
`npm run tauri build` will produce a working EXE:

```
models/stt/nemo-fastconformer-80ms-int8/     <- encoder.int8.onnx, decoder.int8.onnx,
                                                  joiner.int8.onnx, tokens.txt, test_wavs/
                                                  (VAD/endpoint detection only — see below)
sidecars/stt-sidecar/                        <- stt-sidecar.exe + _internal/
sidecars/rag-lite/                           <- rag-lite.exe + _internal/
```

Copy these three directories in from wherever you have a working Smallbird
desktop build (e.g. `apps/desktop/src-tauri/target/release/bundle`'s
resource inputs in the original monorepo: `models/stt/`, `packages/stt/
dist/stt-sidecar/`, and `packages/rag/dist/rag-lite/` — copy each folder's
*contents* directly into the paths above, no extra nesting). Once copied,
these never need to change again unless you intentionally rebuild the VAD
engine or the sidecar executables from source.

### Local VAD engine's Python virtualenv (`npm run tauri dev` only)

`streaming_asr_sidecar/.venv/` is **not tracked in this git repository** —
it's a Python virtualenv, so it's machine-specific (its `python.exe`
hardcodes the absolute path of the base Python install it was created from)
and 100MB+ of prebuilt binaries that add nothing to git history. Only
`streaming_asr_sidecar/sidecar.py` itself is real source.

In dev mode, if this venv doesn't exist, the app falls back to looking for
a frozen sidecar exe (release-build only) and will fail to start voice
capture with a clear error telling you the exact command to run. To set it
up yourself instead:

```powershell
py -3 -m venv streaming_asr_sidecar/.venv
streaming_asr_sidecar/.venv/Scripts/python.exe -m pip install sherpa-onnx numpy
```

Release builds (`npm run tauri build`) never need this — they bundle a
self-contained PyInstaller-frozen sidecar instead (see
`packages/stt/scripts/freeze_sidecar.py` in the original monorepo).

### Groq and Deepgram API keys (voice input/output)

Get a Groq key at [console.groq.com](https://console.groq.com) (speech-to-text)
and a Deepgram key at [console.deepgram.com](https://console.deepgram.com)
(text-to-speech, used for Flux). Both are entered the same way as your LLM
provider key — launch the app, open Settings -> API Keys, and paste each in
under the "Voice" section. Stored in Windows Credential Manager, not an env
var or file.

Without a Groq key configured, voice input will not produce any transcript
text — every detected utterance is dropped and a warning is logged (see
`src-tauri/src/stt/groq.rs`). Without a Deepgram key, enabling the overlay's
"Speak answers aloud" toggle just means no audio plays, warning logged
(see `src-tauri/src/tts/deepgram_flux.rs`). Either way, everything else
in the app (text chat, Notes, RAG search) works normally regardless.

## Development

```powershell
npm run tauri dev
```

## Building the Windows EXE

```powershell
npm run tauri build
```

Output:
- `src-tauri/target/release/desktop.exe` — the raw executable
- `src-tauri/target/release/bundle/nsis/*.exe` — the NSIS installer
  (recommended for actually installing the app)
- `src-tauri/target/release/bundle/msi/*.msi` — an MSI installer

## First run

1. Launch the app (or run the installer, then launch it).
2. Open Settings (gear icon) -> API Keys.
3. Paste in your OpenAI, Anthropic, and/or Gemini API key, and (optionally)
   your Groq and Deepgram keys under the Voice section. Each is stored
   independently in Windows Credential Manager.
4. Pick which provider to use from the header's model picker (this is
   separate from the API Keys panel — the API Keys panel only manages your
   keys, the header picker decides which LLM provider is actually used for
   a request; Groq/Deepgram have no such picker, each is simply configured
   or not).
5. Use Veronica, resume/JD setup analysis, or Notes as normal.

## Customizing

This is your personal copy — feel free to:
- Edit the prompts in `src-tauri/src/personal/prompts/` (ask, analysis,
  meeting, notes, setup).
- Add/adjust AI provider behavior in `src-tauri/src/personal/providers/`.
- Change the UI in `src/`.

Changes here never affect the Smallbird SaaS product — the two are
completely independent codebases with no shared dependency, submodule, or
deployment path.

## What's intentionally not here

This repo does not include and does not need:
- A backend server / API
- Fly.io configuration
- Supabase (database, auth, migrations)
- Stripe or any billing/subscription logic
- A usage/entitlement/minute-balance system
- Production deployment scripts or secrets

If you ever see the app behave as if it's missing one of these (it
shouldn't — sign-in and cloud sync were removed from this build entirely),
that's a bug to fix in this repo, not a missing integration to add back.
