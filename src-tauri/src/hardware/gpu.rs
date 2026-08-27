//! GPU presence/VRAM detection via DXGI adapter enumeration.
//!
//! This is informational only — for hardware-tier scoring and the
//! Performance panel's display ("NVIDIA GeForce RTX 3050, 8GB detected").
//! It does NOT indicate whether a working CUDA/DirectML execution provider
//! is available for STT or RAG; GPU execution is out of scope for this pass
//! (see docs/stt-benchmark.md — GPU was tested and explicitly rejected for
//! STT, and RAG's torch build is CPU-only). DXGI adapter enumeration works
//! without any vendor driver runtime beyond the basic display driver, which
//! is why it was chosen over `nvml-wrapper` (NVIDIA-only, needs `nvml.dll`
//! present) or `wgpu` (pulls in a full graphics stack for a single query).

use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};

pub struct GpuAdapter {
    pub vendor: String,
    pub name: String,
    pub vram_mb: Option<u64>,
}

const VENDOR_NVIDIA: u32 = 0x10DE;
const VENDOR_AMD: u32 = 0x1002;
const VENDOR_INTEL: u32 = 0x8086;

fn vendor_name(vendor_id: u32) -> String {
    match vendor_id {
        VENDOR_NVIDIA => "NVIDIA".to_string(),
        VENDOR_AMD => "AMD".to_string(),
        VENDOR_INTEL => "Intel".to_string(),
        other => format!("Unknown (0x{other:04X})"),
    }
}

/// Returns the adapter with the most dedicated video memory (a reasonable
/// proxy for "the GPU that matters" on a multi-GPU laptop with integrated +
/// discrete graphics). `None` on any enumeration failure or if no adapter
/// reports dedicated VRAM — never panics.
pub fn detect_primary_adapter() -> Option<GpuAdapter> {
    // SAFETY: CreateDXGIFactory1/EnumAdapters1/GetDesc1 are standard,
    // well-documented Win32 COM calls used exactly as Microsoft's own docs
    // show; all `windows` crate FFI bindings are `unsafe` by construction.
    // Every fallible step here returns `None` on error rather than
    // unwrapping, so a driver/enumeration hiccup degrades to "no GPU
    // detected" instead of crashing the app.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;

        let mut best: Option<GpuAdapter> = None;
        let mut best_vram: u64 = 0;

        for i in 0.. {
            let adapter = match factory.EnumAdapters1(i) {
                Ok(a) => a,
                Err(_) => break,
            };
            let Ok(desc) = adapter.GetDesc1() else {
                continue;
            };

            let vram_mb = (desc.DedicatedVideoMemory as u64) / (1024 * 1024);
            // Skip the Microsoft Basic Render Driver / software adapters
            // (zero dedicated VRAM), which aren't useful for tiering.
            if vram_mb == 0 {
                continue;
            }
            if vram_mb > best_vram {
                let name = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_string();
                best_vram = vram_mb;
                best = Some(GpuAdapter {
                    vendor: vendor_name(desc.VendorId),
                    name,
                    vram_mb: Some(vram_mb),
                });
            }
        }

        best
    }
}
