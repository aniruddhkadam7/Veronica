//! Stable hardware-identity fingerprint, used to detect "this is a different
//! physical machine than the one `performance.json` was saved on" so a
//! stale, explicit performance-mode override never silently carries over
//! from one PC to another (see `store.rs`).
//!
//! Built ONLY from characteristics that matter for performance
//! classification (CPU identity + logical core count, RAM, GPU) — never a
//! machine-unique identifier such as the Windows username, a device serial
//! number, or a MAC address. Two different machines with coincidentally
//! identical CPU/RAM/GPU would collide onto the same fingerprint; that is
//! accepted, since this drives a performance-tuning decision, not
//! identity/security.

use super::profile::HardwareProfile;

/// Rounds `total_ram_mb` to the nearest GB before including it — the OS
/// never reports the full nominal RAM size (firmware/integrated-GPU
/// reservation typically costs a few hundred MB, see `tier.rs`'s module
/// doc), and the exact reserved amount can drift by a handful of MB between
/// boots as drivers update. Without rounding, the fingerprint for the same
/// physical machine could flicker from one launch to the next, which would
/// incorrectly look like "hardware changed" and reset the user's mode.
pub fn compute(profile: &HardwareProfile) -> String {
    let ram_gb = (profile.total_ram_mb + 512) / 1024;
    let gpu = profile.gpu_name.as_deref().unwrap_or("none");
    format!(
        "cpu:{}|cores:{}|ram_gb:{ram_gb}|gpu:{gpu}",
        profile.cpu_brand.trim(),
        profile.cpu_logical_cores,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(cpu: &str, cores: usize, ram_mb: u64, gpu: Option<&str>) -> HardwareProfile {
        HardwareProfile {
            cpu_physical_cores: (cores / 2).max(1),
            cpu_logical_cores: cores,
            cpu_brand: cpu.to_string(),
            total_ram_mb: ram_mb,
            available_ram_mb: ram_mb / 2,
            gpu_vendor: gpu.map(|_| "Test".to_string()),
            gpu_name: gpu.map(|g| g.to_string()),
            gpu_vram_mb: None,
            storage_is_ssd: None,
            windows_version: "Windows 11".to_string(),
        }
    }

    #[test]
    fn identical_profiles_produce_identical_fingerprints() {
        let a = profile("Intel i5-13400", 16, 16_000, Some("RTX 3050"));
        let b = profile("Intel i5-13400", 16, 16_000, Some("RTX 3050"));
        assert_eq!(compute(&a), compute(&b));
    }

    #[test]
    fn different_cpu_changes_the_fingerprint() {
        let a = profile("Intel i5-13400", 16, 16_000, None);
        let b = profile("Intel i3-1315U", 8, 16_000, None);
        assert_ne!(compute(&a), compute(&b));
    }

    #[test]
    fn different_core_count_on_the_same_cpu_name_changes_the_fingerprint() {
        let a = profile("Intel i3-1315U", 8, 8_000, None);
        let b = profile("Intel i3-1315U", 4, 8_000, None);
        assert_ne!(compute(&a), compute(&b));
    }

    #[test]
    fn different_ram_changes_the_fingerprint() {
        let a = profile("Intel i3-1315U", 8, 16_000, None);
        let b = profile("Intel i3-1315U", 8, 8_000, None);
        assert_ne!(compute(&a), compute(&b));
    }

    #[test]
    fn different_gpu_changes_the_fingerprint() {
        let a = profile("Intel i5-13400", 16, 16_000, Some("RTX 3050"));
        let b = profile("Intel i5-13400", 16, 16_000, Some("RTX 4060"));
        assert_ne!(compute(&a), compute(&b));
    }

    #[test]
    fn no_gpu_detected_is_a_stable_distinct_value_not_a_wildcard() {
        let a = profile("Intel i3-1315U", 8, 8_000, None);
        let b = profile("Intel i3-1315U", 8, 8_000, Some("RTX 3050"));
        assert_ne!(compute(&a), compute(&b));
    }

    #[test]
    fn small_ram_reporting_variance_across_boots_does_not_change_the_fingerprint() {
        // Real 8GB hardware reports slightly under 8192MB to the OS (see
        // tier.rs) and that reservation can drift a few MB between boots —
        // both must round to the same GB bucket.
        let a = profile("Intel i3-1315U", 8, 7873, None);
        let b = profile("Intel i3-1315U", 8, 7900, None);
        assert_eq!(compute(&a), compute(&b));
    }

    #[test]
    fn fingerprint_is_built_only_from_the_documented_fields() {
        // Guards against someone later appending a sensitive identifier
        // (username, serial number, MAC address) to the format string
        // without updating this test.
        let p = profile("Intel i3-1315U", 8, 7873, None);
        let fp = compute(&p);
        assert!(fp.starts_with("cpu:"));
        assert_eq!(fp.matches('|').count(), 3, "must be exactly cpu|cores|ram_gb|gpu, four fields");
    }
}
