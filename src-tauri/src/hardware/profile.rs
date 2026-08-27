//! Point-in-time snapshot of the machine Smallbird is running on. Detected fresh
//! at every app launch (see `manager::PerformanceManager::detect_and_load`)
//! rather than cached across launches, since hardware/available-RAM can
//! change between runs (e.g. a laptop plugged into a dock, a VM resize).

use sysinfo::System;

use super::gpu;
use super::storage;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareProfile {
    pub cpu_physical_cores: usize,
    pub cpu_logical_cores: usize,
    pub cpu_brand: String,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub gpu_vendor: Option<String>,
    pub gpu_name: Option<String>,
    pub gpu_vram_mb: Option<u64>,
    /// `None` means undetermined (the seek-penalty query failed or the OS
    /// denied access) — treated as "unknown", never penalized in scoring.
    pub storage_is_ssd: Option<bool>,
    pub windows_version: String,
}

/// Runs CPU/RAM/GPU/storage detection. Each sub-check is independently
/// best-effort — a GPU or storage detection failure never prevents the rest
/// of the profile from being produced, since this data only ever feeds a
/// tuning decision, not a hard requirement to start the app.
pub fn detect() -> HardwareProfile {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_logical_cores = sys.cpus().len().max(1);
    let cpu_physical_cores = sys.physical_core_count().unwrap_or(cpu_logical_cores).max(1);
    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".to_string());

    // sysinfo reports memory in bytes.
    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let available_ram_mb = sys.available_memory() / (1024 * 1024);

    let windows_version = System::long_os_version().unwrap_or_else(|| "Windows".to_string());

    let gpu = gpu::detect_primary_adapter();
    let storage_is_ssd = storage::detect_system_drive_is_ssd();

    HardwareProfile {
        cpu_physical_cores,
        cpu_logical_cores,
        cpu_brand,
        total_ram_mb,
        available_ram_mb,
        gpu_vendor: gpu.as_ref().map(|g| g.vendor.clone()),
        gpu_name: gpu.as_ref().map(|g| g.name.clone()),
        gpu_vram_mb: gpu.as_ref().and_then(|g| g.vram_mb),
        storage_is_ssd,
        windows_version,
    }
}

/// Snapshot of just the number that matters for the Milestone 6 runtime
/// checkpoint (spawn-time RAM check) — cheaper than a full `detect()` since
/// it skips GPU/storage/CPU-brand queries that don't change between a
/// session's checkpoints.
pub fn available_ram_mb() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory() / (1024 * 1024)
}

/// Background, non-blocking system-wide CPU utilization sampler. `sysinfo`
/// requires two readings spaced `MINIMUM_CPU_UPDATE_INTERVAL` (~200ms on
/// Windows) apart to compute a meaningful usage delta — far too slow to
/// take inline at an STT-sidecar-spawn or RAG-retrieval checkpoint (would
/// directly cost real-time responsiveness, the system's #2 priority right
/// after stability). Instead this runs one lightweight background thread,
/// started once at app launch, that keeps sampling on that same interval
/// and publishes the latest reading to a shared cell; checkpoint callers
/// just read the last published value, never block on a fresh sample.
pub struct CpuUsageSampler {
    latest: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

/// Sentinel published in `CpuUsageSampler::latest` until the first real
/// sample completes (or forever, if the sampler thread failed to start) —
/// distinguishes "no CPU reading available yet" from a genuine 0% idle
/// reading, which a real machine can legitimately report. NaN never arises
/// from `sysinfo`'s own usage calculation, so it's an unambiguous marker.
const CPU_SAMPLE_PENDING: u32 = 0x7fc0_0000; // f32::NAN.to_bits(), inlined so `AtomicU32::new` stays const

impl CpuUsageSampler {
    /// Spawns the sampler thread and returns a cheap, cloneable handle to
    /// its latest reading. Never fails — if thread spawn itself fails (an
    /// extremely constrained system already under severe pressure), the
    /// handle still works and simply reports "no reading" forever, which
    /// `pressure` treats as "no CPU signal" rather than a false "fully
    /// idle" claim that could mask real saturation (see
    /// `pressure::PressureTracker`).
    pub fn start() -> Self {
        let latest = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(CPU_SAMPLE_PENDING));
        let published = latest.clone();
        let spawned = std::thread::Builder::new()
            .name("cpu-pressure-sampler".into())
            .spawn(move || {
                let mut sys = System::new();
                sys.refresh_cpu_usage();
                loop {
                    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
                    sys.refresh_cpu_usage();
                    let pct = sys.global_cpu_usage();
                    published.store(pct.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
            });
        if spawned.is_err() {
            log::warn!("failed to start CPU pressure sampler thread; CPU pressure protection will be inactive");
        }
        Self { latest }
    }

    /// Latest published reading, or `None` if no sample has completed yet
    /// (briefly true right after `start()`, or permanently true if the
    /// sampler thread failed to spawn). Never blocks.
    pub fn latest_percent(&self) -> Option<f32> {
        let bits = self.latest.load(std::sync::atomic::Ordering::Relaxed);
        if bits == CPU_SAMPLE_PENDING {
            None
        } else {
            Some(f32::from_bits(bits))
        }
    }
}
