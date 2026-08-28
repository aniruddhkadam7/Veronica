//! System volume control via Windows Core Audio (`IAudioEndpointVolume`) —
//! the native API tier for this, same reasoning as `native.rs`'s doc: no
//! shelling out to a volume-control CLI.

use windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume;
use windows::Win32::Media::Audio::{eConsole, eRender, IMMDeviceEnumerator, MMDeviceEnumerator};
use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_APARTMENTTHREADED};

/// Opens the default audio render endpoint's volume interface and runs `f`
/// against it. COM is initialized per-call (apartment-threaded) — cheap,
/// and safe to call redundantly if some other part of the process already
/// initialized COM on this thread (`CoInitializeEx` returning
/// `RPC_E_CHANGED_MODE`/`S_FALSE` here is not treated as fatal).
fn with_endpoint_volume<T>(f: impl FnOnce(&IAudioEndpointVolume) -> Result<T, String>) -> Result<T, String> {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| format!("failed to reach the audio system: {e}"))?;
        let device = enumerator
            .GetDefaultAudioEndpoint(eRender, eConsole)
            .map_err(|e| format!("no default audio output device: {e}"))?;
        let endpoint_volume: IAudioEndpointVolume =
            device.Activate(CLSCTX_ALL, None).map_err(|e| format!("failed to open volume control: {e}"))?;
        f(&endpoint_volume)
    }
}

fn get_percent(ev: &IAudioEndpointVolume) -> Result<u8, String> {
    let level = unsafe { ev.GetMasterVolumeLevelScalar() }.map_err(|e| e.to_string())?;
    Ok((level * 100.0).round().clamp(0.0, 100.0) as u8)
}

fn set_percent(ev: &IAudioEndpointVolume, pct: u8) -> Result<(), String> {
    unsafe { ev.SetMasterVolumeLevelScalar(pct as f32 / 100.0, std::ptr::null()) }.map_err(|e| e.to_string())
}

pub fn set_volume_percent(pct: u8) -> Result<String, String> {
    let pct = pct.min(100);
    with_endpoint_volume(|ev| {
        set_percent(ev, pct)?;
        Ok(format!("Volume set to {pct} percent."))
    })
}

pub fn adjust_volume(delta_pct: i32) -> Result<String, String> {
    with_endpoint_volume(|ev| {
        let current = get_percent(ev)? as i32;
        let new = (current + delta_pct).clamp(0, 100) as u8;
        set_percent(ev, new)?;
        Ok(if delta_pct >= 0 { "Volume up.".to_string() } else { "Volume down.".to_string() })
    })
}

pub fn set_mute(mute: bool) -> Result<String, String> {
    with_endpoint_volume(|ev| {
        unsafe { ev.SetMute(mute, std::ptr::null()) }.map_err(|e| e.to_string())?;
        Ok(if mute { "Muted.".to_string() } else { "Unmuted.".to_string() })
    })
}

pub fn get_volume_percent() -> Result<String, String> {
    with_endpoint_volume(|ev| {
        let pct = get_percent(ev)?;
        let muted = unsafe { ev.GetMute() }.map_err(|e| e.to_string())?.as_bool();
        Ok(if muted { format!("Volume is {pct} percent, but muted.") } else { format!("Volume is {pct} percent.") })
    })
}
