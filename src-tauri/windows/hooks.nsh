; Installer hooks for the Windows NSIS bundle (see tauri.conf.json's
; bundle.windows.nsis.installerHooks).
;
; Why this exists: the desktop app spawns two background child processes at
; startup — the STT sidecar (stt-sidecar.exe) and the RAG-lite service
; (rag-lite.exe), both bundled as resources next to the main exe (see
; src/stt/sidecar.rs, src/rag/process.rs). Neither has a visible window, so
; a user reinstalling/updating over a still-running instance has no obvious
; way to know they need to close it first. Without this, Windows can't
; overwrite the locked DLLs those processes have open, and the installer
; fails mid-extraction with an "Error opening file for writing" prompt (seen
; in the field on a real test machine, on rag-lite's VCRUNTIME140.dll) —
; confusing for anyone who isn't a developer and doesn't know to check Task
; Manager for background exe's with no window.
;
; taskkill's exit code is deliberately ignored (nothing to do differently
; whether or not each process was actually running) — this only ever needs
; to be best-effort cleanup before extraction, never a hard failure gate.
!macro NSIS_HOOK_PREINSTALL
  ExecWait 'taskkill /F /IM "desktop.exe" /T'
  ExecWait 'taskkill /F /IM "stt-sidecar.exe" /T'
  ExecWait 'taskkill /F /IM "rag-lite.exe" /T'
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  ExecWait 'taskkill /F /IM "desktop.exe" /T'
  ExecWait 'taskkill /F /IM "stt-sidecar.exe" /T'
  ExecWait 'taskkill /F /IM "rag-lite.exe" /T'
!macroend
