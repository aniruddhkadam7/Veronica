//! Adaptive hardware-based performance tuning for Smallbird's local STT and RAG
//! pipelines. See docs/performance-tuning.md (once Milestone 7 lands) for
//! the benchmark evidence behind the tier→config table in `manager.rs`.
//!
//! Scope: STT (sherpa-onnx sidecar thread count) and RAG (embedding
//! batch/thread count, retrieval top_k/context/timeout) only. Smallbird's LLM
//! is a cloud API call via `apps/backend` — not touched here, and never
//! hardware-tiered (see `manager.rs`'s module doc on LLM context).

pub mod commands;
pub mod fingerprint;
pub mod gpu;
pub mod manager;
pub mod pressure;
pub mod profile;
pub mod storage;
pub mod store;
pub mod stt_rag_coordination;
pub mod telemetry;
pub mod tier;

use std::sync::Mutex;

use tauri::{AppHandle, Manager};

use manager::PerformanceManager;
use profile::CpuUsageSampler;

/// Tauri-managed state wrapping the single `PerformanceManager` instance —
/// registered directly via `.manage(...)`, following the same top-level
/// state pattern as `NotesStore`/`AgentStore`, rather than nesting inside
/// `AppState`. `Mutex`-wrapped because Tauri commands run on arbitrary
/// threads. `cpu_sampler` lives alongside the manager rather than inside
/// its `Mutex` — it only ever needs a non-blocking atomic read (see
/// `CpuUsageSampler::latest_percent`), so it doesn't need the same lock.
///
/// `telemetry_history` is the Milestone 7 latency dashboard's in-memory ring
/// buffer of recent turns (see `telemetry::TurnHistory`) — kept alongside
/// the manager/sampler rather than in `AppState` since this module already
/// owns `telemetry::PerfContext` production via `perf_context()`. It has its
/// own internal `Mutex` (not covered by the outer one), so reading/clearing
/// it never contends with a performance-mode change.
pub struct PerformanceState {
    pub manager: Mutex<PerformanceManager>,
    pub cpu_sampler: CpuUsageSampler,
    pub telemetry_history: telemetry::TurnHistory,
}

/// Detects hardware and loads the persisted mode preference. Called once
/// from `lib.rs`'s `.setup()`, since loading the persisted mode needs an
/// `AppHandle` (to resolve the app data directory) that isn't available at
/// `.manage()` time.
///
/// The hardware profile itself is always freshly detected (never cached
/// across launches — see `profile::detect`'s doc). What CAN otherwise go
/// stale is the persisted *mode* preference: an explicit `MaximumPerformance`/
/// `BatterySaver` choice is only trustworthy on the machine it was chosen
/// on, so it's tagged with a hardware fingerprint (`fingerprint::compute`)
/// and `store::load` resets it back to `Adaptive` if this launch's
/// fingerprint doesn't match — see `store.rs`'s module doc for the full
/// reconciliation rules (same machine / different machine / missing
/// fingerprint migration / corrupted file).
pub fn init(app: &AppHandle) -> PerformanceState {
    let profile = profile::detect();
    let current_fingerprint = fingerprint::compute(&profile);
    let loaded = store::load(app, &current_fingerprint);
    log::info!(
        "hardware profile: {} logical cores, {}MB RAM, gpu={:?}, ssd={:?}",
        profile.cpu_logical_cores,
        profile.total_ram_mb,
        profile.gpu_name,
        profile.storage_is_ssd
    );
    if loaded.hardware_changed {
        log::info!(
            "hardware fingerprint changed since performance.json was last written — \
             this looks like a different physical machine; resetting performance mode \
             to Adaptive rather than reusing an override chosen on other hardware"
        );
    }
    let mut manager = PerformanceManager::new(profile, loaded.mode);
    manager.set_hardware_refreshed(loaded.hardware_changed);
    log::info!("performance mode: {:?}, detected tier: {:?}", manager.mode(), manager.detected_tier());
    PerformanceState {
        manager: Mutex::new(manager),
        cpu_sampler: CpuUsageSampler::start(),
        telemetry_history: telemetry::TurnHistory::new(),
    }
}

/// Convenience for spawn-time call sites (Milestones 4b/5) that just need
/// the current effective config without holding onto the manager. Does NOT
/// consult the Milestone 6 memory-pressure tracker — see
/// `effective_config_checked` for the checkpoint-driven variant that does.
pub fn effective_config(app: &AppHandle) -> manager::PerformanceConfig {
    let state = app.state::<PerformanceState>();
    let manager = state.manager.lock().unwrap();
    manager.effective_config()
}

/// Milestone 6 checkpoint (extended to also cover CPU pressure, not just
/// RAM): feeds a freshly-measured available-RAM reading, plus the CPU
/// sampler's latest published reading, into the sustained-pressure tracker
/// and returns the (possibly clamped) config to use, plus a log-worthy
/// reason if the pressure state changed on this call. Callers should
/// measure RAM immediately before calling this (see
/// `profile::available_ram_mb`) — the natural checkpoints are STT sidecar
/// spawn and RAG document-upload/indexing start, not a timer. The CPU
/// reading is never measured inline here (that would block this checkpoint
/// for ~200ms — see `profile::CpuUsageSampler`'s doc); it's whatever the
/// background sampler thread most recently published, read without
/// blocking.
pub fn effective_config_checked(app: &AppHandle) -> manager::PerformanceConfig {
    let available = profile::available_ram_mb();
    let state = app.state::<PerformanceState>();
    let cpu_percent = state.cpu_sampler.latest_percent();
    let mut manager = state.manager.lock().unwrap();
    let (config, reason) = manager.effective_config_checked(available, cpu_percent);
    if let Some(reason) = reason {
        log::warn!("performance: {reason}");
    }
    config
}

/// Milestone 7: reads the current performance-correlation context (tier,
/// mode, pressure state, effective config) without observing a new RAM
/// sample — safe to call from any pipeline-stage instrumentation point.
pub fn perf_context(app: &AppHandle) -> telemetry::PerfContext {
    let state = app.state::<PerformanceState>();
    let manager = state.manager.lock().unwrap();
    manager.perf_context()
}

/// Records one turn's telemetry snapshot into the dashboard's in-memory
/// history — a `Mutex` lock plus a bounded `VecDeque` push, no I/O. Safe to
/// call from the voice-turn-completion path (`veronica::ask_veronica`)
/// without adding any meaningful latency there, matching the same cost
/// class as the other state locks already taken on that path.
pub fn record_turn_telemetry(app: &AppHandle, snapshot: telemetry::TurnSnapshot) {
    let state = app.state::<PerformanceState>();
    state.telemetry_history.push(snapshot);
}

/// STT/RAG scheduling coordination (Phase B): whether active STT should
/// currently throttle RAG background indexing on this machine/mode. See
/// `manager::PerformanceManager::should_throttle_background_work` for the
/// exact per-mode rule and `stt_rag_coordination` for how this gets acted
/// on.
pub fn should_throttle_background_work(app: &AppHandle) -> bool {
    let state = app.state::<PerformanceState>();
    let manager = state.manager.lock().unwrap();
    manager.should_throttle_background_work()
}

// Re-exported for the Milestone 6 runtime-adaptation checkpoint, which
// needs a fresh available-RAM reading independent of the cached profile.
pub use profile::available_ram_mb;
pub use manager::PerformanceConfig;
