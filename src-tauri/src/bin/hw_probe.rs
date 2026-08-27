//! Prints the real, freshly-detected hardware profile and the tier/score
//! breakdown it produces on THIS machine — no Tauri app, no AppHandle, no
//! `performance.json` involved. Exists so hardware classification can be
//! verified against ground truth on real hardware (see
//! `docs/performance-tuning.md`'s STT-start-reliability section) instead of
//! guessed at from a synthetic test profile.
//!
//!     cargo run --bin hw_probe

fn main() {
    let profile = desktop_lib::hardware::profile::detect();
    let fingerprint = desktop_lib::hardware::fingerprint::compute(&profile);
    let score = desktop_lib::hardware::tier::score(&profile);
    let tier = desktop_lib::hardware::tier::classify(&profile);

    println!("=== hardware profile (real detection) ===");
    println!("cpu_brand           : {}", profile.cpu_brand);
    println!("cpu_physical_cores   : {}", profile.cpu_physical_cores);
    println!("cpu_logical_cores    : {}", profile.cpu_logical_cores);
    println!("total_ram_mb         : {}", profile.total_ram_mb);
    println!("available_ram_mb     : {}", profile.available_ram_mb);
    println!("gpu_vendor           : {:?}", profile.gpu_vendor);
    println!("gpu_name             : {:?}", profile.gpu_name);
    println!("gpu_vram_mb          : {:?}", profile.gpu_vram_mb);
    println!("storage_is_ssd       : {:?}", profile.storage_is_ssd);
    println!("windows_version      : {}", profile.windows_version);
    println!();
    println!("hardware_fingerprint : {fingerprint}");
    println!();
    println!("=== tier scoring ===");
    println!("total score          : {score} / 80");
    println!("classified tier      : {tier:?}");

    let manager = desktop_lib::hardware::manager::PerformanceManager::new(
        profile,
        desktop_lib::hardware::manager::PerformanceMode::Adaptive,
    );
    let cfg = manager.effective_config();
    println!();
    println!("=== effective config (Adaptive mode) ===");
    println!("stt_num_threads         : {}", cfg.stt_num_threads);
    println!("rag_top_k               : {}", cfg.rag_top_k);
    println!("rag_max_context_chars   : {}", cfg.rag_max_context_chars);
    println!("rag_similarity_threshold: {}", cfg.rag_similarity_threshold);
    println!("rag_retrieval_timeout_ms: {}", cfg.rag_retrieval_timeout_ms);
    println!("rag_embed_batch_size    : {}", cfg.rag_embed_batch_size);
    println!("rag_torch_threads       : {}", cfg.rag_torch_threads);
}
