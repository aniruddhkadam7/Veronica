# Smallbird Local

A personal, standalone build of the Smallbird desktop app. It talks directly
to your own OpenAI, Anthropic, or Gemini API key — no account, no
subscription, no SaaS backend, no Supabase, no cloud entitlement system.
This repository is completely independent of the Smallbird SaaS product and
can be built and modified on its own.

## What this is

- A Tauri (Rust + React/TypeScript) desktop app for Windows.
- Interview Mode, Meeting Mode, Notes, resume/JD setup analysis, and full
  interview scoring — all calling your configured AI provider directly.
- Local speech-to-text (bundled, offline, no cloud dependency).
- Local document search/RAG (bundled, offline, no cloud dependency).
- Your API key is stored in Windows Credential Manager via the `keyring`
  crate — never in a file, never logged, never sent anywhere except
  directly to the provider you configure.

## Architecture

```
Smallbird Local (this repo)
        |
        +-- Tauri Desktop App (src/, src-tauri/)
        +-- Local STT (sidecars/stt-sidecar/, models/stt/)
        +-- Local RAG / document search (sidecars/rag-lite/)
        +-- Direct AI provider client (src-tauri/src/personal/)
        +-- Your API keys (Windows Credential Manager)
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
STT model alone is 126MB, over GitHub's 100MB per-file limit, and none of
these are source files anyway (they're prebuilt binary build inputs, not
something you'd normally edit). You need to place them yourself before
`npm run tauri build` will produce a working EXE:

```
models/stt/nemo-fastconformer-80ms-int8/     <- encoder.int8.onnx, decoder.int8.onnx,
                                                  joiner.int8.onnx, tokens.txt, test_wavs/
sidecars/stt-sidecar/                        <- stt-sidecar.exe + _internal/
sidecars/rag-lite/                           <- rag-lite.exe + _internal/
```

Copy these three directories in from wherever you have a working Smallbird
desktop build (e.g. `apps/desktop/src-tauri/target/release/bundle`'s
resource inputs in the original monorepo: `models/stt/`, `packages/stt/
dist/stt-sidecar/`, and `packages/rag/dist/rag-lite/` — copy each folder's
*contents* directly into the paths above, no extra nesting). Once copied,
these never need to change again unless you intentionally rebuild the STT
model or the sidecar executables from source.

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
3. Paste in your OpenAI, Anthropic, and/or Gemini API key. Each is stored
   independently in Windows Credential Manager.
4. Pick which provider to use from the header's model picker (this is
   separate from the API Keys panel — the API Keys panel only manages your
   keys, the header picker decides which one is actually used for a
   request).
5. Use Interview Mode, Meeting Mode, or Notes as normal.

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
