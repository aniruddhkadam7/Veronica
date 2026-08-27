//! On-disk persistence for the user's performance mode preference, plus the
//! hardware fingerprint it was chosen under (see `fingerprint.rs`). The
//! hardware profile's raw numbers are never persisted (re-detected fresh
//! every launch, see `profile::detect`) — but WHICH machine an explicit
//! `mode` override was chosen on must be, so a stale `MaximumPerformance`/
//! `BatterySaver` choice made on one PC never silently forces its config
//! onto a different, later PC that happens to load the same
//! `performance.json` (e.g. a roamed/imaged AppData folder). Follows the
//! same load/save-to-AppData-JSON pattern as `notes_mode::store`.

use std::fs;
use std::path::{Path, PathBuf};

use tauri::{AppHandle, Manager};

use super::manager::PerformanceMode;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct StoredPerformanceSettings {
    #[serde(default)]
    mode: PerformanceMode,
    /// `None` for a pre-fingerprint `performance.json` (migration case) or a
    /// freshly-defaulted one — see `resolve`.
    #[serde(default)]
    hardware_fingerprint: Option<String>,
}

/// Result of loading (and reconciling) the persisted settings against the
/// current machine's hardware fingerprint.
pub struct LoadedPerformanceSettings {
    pub mode: PerformanceMode,
    /// `true` only when a persisted fingerprint was present and did NOT
    /// match the current machine — i.e. an explicit mode override was
    /// invalidated because this is a different physical PC than the one it
    /// was chosen on. `false` for a fresh install, a migrated pre-
    /// fingerprint file, a corrupted file, or a matching fingerprint.
    pub hardware_changed: bool,
}

fn settings_file_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("failed to resolve app data directory: {e}"))?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create app data directory: {e}"))?;
    Ok(dir.join("performance.json"))
}

/// Reads and parses `performance.json`. `None` covers every failure case
/// uniformly (missing file, unreadable file, invalid/partially-written
/// JSON) — callers must treat `None` exactly like "no prior settings",
/// never distinguish "corrupted" from "missing" beyond that, so a damaged
/// file always degrades gracefully to safe defaults instead of erroring.
fn load_stored(path: &Path) -> Option<StoredPerformanceSettings> {
    let contents = fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

/// Pure reconciliation logic (no file I/O — see `load` for the wrapper that
/// adds it), which is what makes this directly unit-testable:
///
/// - No prior settings at all (fresh install, corrupted/partially-written
///   file): `Adaptive`, not a hardware change — there is nothing to compare
///   against yet.
/// - Prior settings with no fingerprint (a `performance.json` written
///   before this feature existed): migrate by trusting the persisted
///   `mode` once — it was necessarily chosen on whatever machine wrote that
///   file, since fingerprinting didn't exist yet — and adopt a fingerprint
///   going forward. Not reported as a hardware change.
/// - Prior settings whose fingerprint matches the current machine: reuse
///   `mode` as-is. This is what makes an explicit `MaximumPerformance`/
///   `BatterySaver` choice durable across ordinary restarts.
/// - Prior settings whose fingerprint does NOT match: this is a different
///   physical machine (or the same machine after enough of a hardware
///   change to move its fingerprint — see `fingerprint.rs`). An explicit
///   override chosen on other hardware must never silently carry over, so
///   this resets to `Adaptive` and reports the change so the caller can
///   surface it and re-persist a fresh fingerprint.
fn resolve(stored: Option<StoredPerformanceSettings>, current_fingerprint: &str) -> (PerformanceMode, bool) {
    let Some(stored) = stored else {
        return (PerformanceMode::Adaptive, false);
    };
    match stored.hardware_fingerprint {
        None => (stored.mode, false),
        Some(saved) if saved == current_fingerprint => (stored.mode, false),
        Some(_) => (PerformanceMode::Adaptive, true),
    }
}

fn write(path: &Path, mode: PerformanceMode, fingerprint: &str) -> Result<(), String> {
    let json = serde_json::to_string_pretty(&StoredPerformanceSettings {
        mode,
        hardware_fingerprint: Some(fingerprint.to_string()),
    })
    .map_err(|e| e.to_string())?;
    fs::write(path, json).map_err(|e| e.to_string())
}

/// Loads the persisted mode, reconciles it against `current_fingerprint`
/// (see `resolve`), and always re-persists afterward — self-healing a
/// missing/corrupted/pre-fingerprint/stale-machine file into a clean,
/// current one in the same call, so every subsequent launch starts from a
/// known-good state. Never panics: every failure path (unresolvable app
/// data dir, unreadable/corrupted file, failed write) degrades to a safe
/// default and a logged warning instead.
pub fn load(app: &AppHandle, current_fingerprint: &str) -> LoadedPerformanceSettings {
    let path = match settings_file_path(app) {
        Ok(p) => p,
        Err(err) => {
            log::warn!("performance store: {err}");
            return LoadedPerformanceSettings { mode: PerformanceMode::Adaptive, hardware_changed: false };
        }
    };

    let stored = load_stored(&path);
    let (mode, hardware_changed) = resolve(stored, current_fingerprint);

    if let Err(err) = write(&path, mode, current_fingerprint) {
        log::warn!("performance store: failed to persist settings: {err}");
    }

    LoadedPerformanceSettings { mode, hardware_changed }
}

/// Persists an explicit mode change (`set_performance_mode`), tagged with
/// the current machine's fingerprint so a later launch on different
/// hardware knows not to trust it.
pub fn save_mode(app: &AppHandle, mode: PerformanceMode, fingerprint: &str) -> Result<(), String> {
    let path = settings_file_path(app)?;
    write(&path, mode, fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("smallbird-perf-store-test-{name}-{}.json", std::process::id()))
    }

    // -- resolve(): the pure reconciliation decision --------------------

    #[test]
    fn no_prior_settings_defaults_to_adaptive_and_is_not_a_hardware_change() {
        let (mode, changed) = resolve(None, "fp-a");
        assert_eq!(mode, PerformanceMode::Adaptive);
        assert!(!changed);
    }

    #[test]
    fn matching_fingerprint_reuses_the_persisted_mode() {
        let stored = Some(StoredPerformanceSettings {
            mode: PerformanceMode::MaximumPerformance,
            hardware_fingerprint: Some("fp-a".into()),
        });
        let (mode, changed) = resolve(stored, "fp-a");
        assert_eq!(mode, PerformanceMode::MaximumPerformance);
        assert!(!changed, "an explicit override on the SAME machine must remain respected");
    }

    #[test]
    fn different_fingerprint_invalidates_the_explicit_mode() {
        let stored = Some(StoredPerformanceSettings {
            mode: PerformanceMode::MaximumPerformance,
            hardware_fingerprint: Some("fp-old-high-end-pc".into()),
        });
        let (mode, changed) = resolve(stored, "fp-new-i3-laptop");
        assert_eq!(
            mode,
            PerformanceMode::Adaptive,
            "an explicit override chosen on a different physical machine must never silently carry over"
        );
        assert!(changed);
    }

    #[test]
    fn missing_fingerprint_field_migrates_by_trusting_the_existing_mode_once() {
        // Pre-fingerprint schema: the mode was necessarily set on this very
        // machine (the feature didn't exist yet to have come from anywhere
        // else), so it's trusted once and a fingerprint is adopted from here.
        let stored = Some(StoredPerformanceSettings { mode: PerformanceMode::BatterySaver, hardware_fingerprint: None });
        let (mode, changed) = resolve(stored, "fp-a");
        assert_eq!(mode, PerformanceMode::BatterySaver);
        assert!(!changed);
    }

    #[test]
    fn adaptive_mode_is_unaffected_either_way_since_it_is_never_hardware_specific() {
        let same = resolve(
            Some(StoredPerformanceSettings { mode: PerformanceMode::Adaptive, hardware_fingerprint: Some("fp-a".into()) }),
            "fp-a",
        );
        let different = resolve(
            Some(StoredPerformanceSettings { mode: PerformanceMode::Adaptive, hardware_fingerprint: Some("fp-old".into()) }),
            "fp-new",
        );
        assert_eq!(same.0, PerformanceMode::Adaptive);
        assert_eq!(different.0, PerformanceMode::Adaptive);
    }

    // -- load_stored(): real file I/O, missing/corrupted/partial-write ----

    #[test]
    fn missing_file_loads_as_none() {
        let path = temp_path("missing");
        let _ = fs::remove_file(&path);
        assert!(load_stored(&path).is_none());
    }

    #[test]
    fn corrupted_json_loads_as_none_rather_than_panicking() {
        let path = temp_path("corrupt");
        fs::write(&path, b"{ this is not valid json ").unwrap();
        assert!(load_stored(&path).is_none());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_partially_written_file_is_treated_the_same_as_corrupted() {
        // Simulates a write that was interrupted mid-flush (crash, power
        // loss) — valid UTF-8, invalid JSON.
        let path = temp_path("partial");
        fs::write(&path, br#"{"mode":"maximumPerformance","hardware_f"#).unwrap();
        assert!(load_stored(&path).is_none());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_empty_file_loads_as_none() {
        let path = temp_path("empty");
        fs::write(&path, b"").unwrap();
        assert!(load_stored(&path).is_none());
        let _ = fs::remove_file(&path);
    }

    // -- write() / load_stored() round trip --------------------------------

    #[test]
    fn write_then_load_round_trips_mode_and_fingerprint() {
        let path = temp_path("roundtrip");
        write(&path, PerformanceMode::BatterySaver, "fp-x").unwrap();
        let stored = load_stored(&path).expect("just-written file must parse");
        assert_eq!(stored.mode, PerformanceMode::BatterySaver);
        assert_eq!(stored.hardware_fingerprint.as_deref(), Some("fp-x"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn same_machine_across_launches_keeps_the_explicit_mode_stable() {
        // End-to-end version of test case A/F from the brief: write what a
        // first launch (or an explicit `set_performance_mode`) would
        // produce, then simulate a second launch reading it back with the
        // identical fingerprint.
        let path = temp_path("same-machine");
        write(&path, PerformanceMode::MaximumPerformance, "fp-same").unwrap();
        let (mode, changed) = resolve(load_stored(&path), "fp-same");
        assert_eq!(mode, PerformanceMode::MaximumPerformance);
        assert!(!changed);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_corrupted_file_on_disk_self_heals_after_one_resolve_and_write_cycle() {
        let path = temp_path("self-heal");
        fs::write(&path, b"not json at all").unwrap();
        let (mode, changed) = resolve(load_stored(&path), "fp-a");
        assert_eq!(mode, PerformanceMode::Adaptive);
        assert!(!changed);
        write(&path, mode, "fp-a").unwrap();
        // Next launch on the same machine now finds a clean, matching file.
        let (mode2, changed2) = resolve(load_stored(&path), "fp-a");
        assert_eq!(mode2, PerformanceMode::Adaptive);
        assert!(!changed2);
        let _ = fs::remove_file(&path);
    }
}
