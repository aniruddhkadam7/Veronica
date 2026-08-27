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
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(super) enum SidecarLine {
    Ready,
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
