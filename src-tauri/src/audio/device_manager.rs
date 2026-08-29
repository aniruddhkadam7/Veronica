use wasapi::{DeviceCollection, Direction};

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub direction: String,
}

/// Enumerates Windows audio endpoints (render devices for system-audio loopback,
/// capture devices for microphone input) so Settings can show device pickers.
pub struct AudioDeviceManager;

impl AudioDeviceManager {
    pub fn list_output_devices() -> Result<Vec<AudioDeviceInfo>, String> {
        Self::list_devices(Direction::Render)
    }

    pub fn list_input_devices() -> Result<Vec<AudioDeviceInfo>, String> {
        Self::list_devices(Direction::Capture)
    }

    /// Runs on a dedicated, throwaway OS thread — unlike `mic_capture`/
    /// `system_capture`'s dedicated capture threads (which always start
    /// fresh and never touched COM before), this is called from a Tauri
    /// command handler, and the WebView2 host thread that runs on already
    /// has COM initialized apartment-threaded (STA) by the WebView2 runtime
    /// itself. `wasapi::initialize_mta()` can't change an already-set COM
    /// mode on a thread (`RPC_E_CHANGED_MODE`, 0x80010106) — it can only
    /// initialize a thread that has no COM mode yet — so this must run on a
    /// fresh thread that has never called `CoInitializeEx`, exactly like the
    /// capture threads already do.
    fn list_devices(direction: Direction) -> Result<Vec<AudioDeviceInfo>, String> {
        std::thread::Builder::new()
            .name("audio-device-enum".into())
            .spawn(move || Self::list_devices_on_this_thread(direction))
            .map_err(|e| e.to_string())?
            .join()
            .map_err(|_| "audio device enumeration thread panicked".to_string())?
    }

    fn list_devices_on_this_thread(direction: Direction) -> Result<Vec<AudioDeviceInfo>, String> {
        wasapi::initialize_mta().ok().map_err(|e| e.to_string())?;

        let enumerator = wasapi::DeviceEnumerator::new().map_err(|e| e.to_string())?;
        let default = enumerator
            .get_default_device(&direction)
            .ok()
            .and_then(|d| d.get_id().ok());

        let collection: DeviceCollection = enumerator
            .get_device_collection(&direction)
            .map_err(|e| e.to_string())?;
        let count = collection.get_nbr_devices().map_err(|e| e.to_string())?;

        let mut devices = Vec::with_capacity(count as usize);
        for i in 0..count {
            let device = collection
                .get_device_at_index(i)
                .map_err(|e| e.to_string())?;
            let id = device.get_id().map_err(|e| e.to_string())?;
            let name = device.get_friendlyname().unwrap_or_else(|_| id.clone());
            let is_default = default.as_deref() == Some(id.as_str());
            devices.push(AudioDeviceInfo {
                id,
                name,
                is_default,
                direction: match direction {
                    Direction::Render => "output".to_string(),
                    Direction::Capture => "input".to_string(),
                },
            });
        }
        Ok(devices)
    }
}
