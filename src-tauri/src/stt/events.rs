use crate::audio::AudioSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SttEventKind {
    Partial,
    Final,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SttEvent {
    pub kind: SttEventKind,
    pub text: String,
    pub source: AudioSource,
    /// Present only for `Final` events; process-relative monotonic seconds.
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
}

/// Raw JSON shape emitted by the Python sidecar, one object per stdout line.
/// The sidecar's own decode is used only for endpoint timing — see
/// `stt/sidecar.rs`'s reader thread and `streaming_asr_sidecar/sidecar.py`'s
/// module doc — so `Partial.text`/`.source` are read (required to
/// deserialize a real "partial" line at all) but never used past that;
/// `#[allow(dead_code)]` documents that this is deliberate, not an oversight.
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(super) enum SidecarLine {
    Ready,
    #[allow(dead_code)]
    Partial {
        text: String,
        source: AudioSource,
    },
    Final {
        text: String,
        source: AudioSource,
        start_time: f64,
        end_time: f64,
    },
    Error {
        message: String,
    },
}
