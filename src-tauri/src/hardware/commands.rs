use tauri::{AppHandle, State};

use crate::rag::EmbedProcessConfig;
use crate::state::AppState;

use super::manager::{PerformanceConfig, PerformanceMode};
use super::profile::HardwareProfile;
use super::store;
use super::stt_mode::{self, ResolvedSttMode, SttModePreference};
use super::tier::HardwareTier;
use super::PerformanceState;

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PerformanceModeInfo {
    pub mode: PerformanceMode,
    pub detected_tier: HardwareTier,
    pub effective_config: PerformanceConfig,
    /// `true` only immediately after a launch where the persisted hardware
    /// fingerprint didn't match this machine's — i.e. this is a different
    /// physical PC than the one an explicit mode override was chosen on,
    /// so it was reset to Adaptive. See `hardware::store`'s module doc.
    pub hardware_refreshed: bool,
    /// `true` when this machine scored into the lowest capability tier
    /// (`HardwareTier::Entry`) — used by the UI to show a one-time,
    /// non-blocking "your hardware is below our recommended configuration"
    /// notice. Purely informational: never affects `effective_config()`,
    /// never blocks install/launch/usage.
    pub is_below_recommended: bool,
}

#[tauri::command]
pub fn get_hardware_profile(state: State<'_, PerformanceState>) -> Result<HardwareProfile, String> {
    let manager = state.0.lock().map_err(|e| e.to_string())?;
    Ok(manager.profile().clone())
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SttModeInfo {
    pub preference: SttModePreference,
    pub resolved: ResolvedSttMode,
    /// Purely informational — never a gate. See `stt_mode::local_stt_recommended`.
    pub local_stt_recommended: bool,
}

/// Read-only: reports what STT_MODE would resolve to on this machine right
/// now. Does not change which STT engine actually runs — that's still
/// `SttEngineKind::from_env` in `stt::sidecar`, unaffected by this command
/// (see `hardware::stt_mode`'s module doc for why cloud STT stays
/// unreachable regardless of this result).
#[tauri::command]
pub fn get_stt_mode_info(state: State<'_, PerformanceState>) -> Result<SttModeInfo, String> {
    let manager = state.0.lock().map_err(|e| e.to_string())?;
    let tier = manager.detected_tier();
    let preference = stt_mode::preference_from_env();
    Ok(SttModeInfo {
        preference,
        resolved: stt_mode::resolve(preference, tier),
        local_stt_recommended: stt_mode::local_stt_recommended(tier),
    })
}

#[tauri::command]
pub fn get_performance_mode(state: State<'_, PerformanceState>) -> Result<PerformanceModeInfo, String> {
    let manager = state.0.lock().map_err(|e| e.to_string())?;
    Ok(PerformanceModeInfo {
        mode: manager.mode(),
        detected_tier: manager.detected_tier(),
        effective_config: manager.effective_config(),
        hardware_refreshed: manager.hardware_refreshed(),
        is_below_recommended: manager.detected_tier() == HardwareTier::Entry,
    })
}

/// Persists the mode and updates the in-memory manager immediately (so the
/// Performance panel's UI reflects the new mode right away — no reason to
/// make the user wait on the RAG sidecar for that), then restarts the local
/// RAG sidecar with the new tier's embedding config in the background —
/// Python's `Settings` class (packages/rag/app/core/config.py) reads
/// `RAG_EMBED_BATCH_SIZE`/`RAG_TORCH_THREADS` once at process startup, so
/// applying a new value requires a fresh process (see
/// `rag::RagServiceHandle::restart`'s doc comment). This means a mode change
/// causes a brief background interruption to local document search/upload —
/// accepted as correct, simple UX rather than attempting live in-process
/// reconfiguration — but it must not block the mode switch itself from
/// appearing to take effect, especially on low-end hardware where the
/// embedding model can take tens of seconds to reload.
///
/// STT thread count does NOT trigger any sidecar restart here — it takes
/// effect on the next recording session, since forcibly killing an
/// in-progress recording's STT sidecar on a settings change would be a
/// worse user experience than a stale thread count for the rest of the
/// current session. The cloud LLM/backend architecture is untouched by mode
/// changes entirely — Smallbird's LLM call is not hardware-tiered (see
/// `hardware::manager`'s module doc).
#[tauri::command]
pub async fn set_performance_mode(
    app: AppHandle,
    state: State<'_, AppState>,
    performance: State<'_, PerformanceState>,
    mode: PerformanceMode,
) -> Result<(), String> {
    let embed_config = {
        let mut manager = performance.0.lock().map_err(|e| e.to_string())?;
        store::save_mode(&app, mode, &manager.hardware_fingerprint())?;
        manager.set_mode(mode);
        // The user just made an explicit choice on THIS machine — any
        // earlier "reset because hardware changed" notice is now stale and
        // should stop showing.
        manager.set_hardware_refreshed(false);
        let cfg = manager.effective_config();
        EmbedProcessConfig {
            embed_batch_size: cfg.rag_embed_batch_size,
            torch_threads: cfg.rag_torch_threads,
        }
    };

    // Only restart if a RAG service is actually running (venv present and
    // spawn succeeded at startup) — no-op otherwise, matching how every
    // other RAG-touching command treats an unavailable service as a
    // non-fatal "feature unavailable" state rather than an error here.
    let needs_restart = {
        let rag_slot = state.rag_service.lock().map_err(|e| e.to_string())?;
        rag_slot.is_some()
    };
    if needs_restart {
        let mut rag_slot = state.rag_service.lock().map_err(|e| e.to_string())?;
        if let Some(handle) = rag_slot.as_mut() {
            handle.restart(embed_config, Some(&app))?;
        }
        // Deliberately NOT awaited: the embedding model reload this triggers
        // can take tens of seconds on low-end hardware (see
        // docs/i3-stt-profiling-report.html's model-load figures), and this
        // command has already done everything the mode switch itself needs
        // (persisted + applied above) — making the caller wait on top of
        // that would reintroduce the exact "button looks frozen" UX problem
        // this change fixes. RAG commands already report "unavailable"
        // gracefully until the fresh process reports healthy, the same as
        // at initial startup.
        tauri::async_runtime::spawn_blocking(crate::rag::wait_until_healthy_default);
    }

    Ok(())
}
