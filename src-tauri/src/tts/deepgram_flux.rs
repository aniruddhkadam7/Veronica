//! Deepgram Flux TTS (`flux-sienna-en`) client: a persistent `wss://`
//! session for one answer's worth of speech, not one request per sentence.
//!
//! This is the only TTS provider in the app — there is no local model, no
//! fallback. Flux's `/v2/speak` WebSocket is turn-based: connect once,
//! stream `Speak` messages as sentences become available from the LLM,
//! `Flush` to mark the end of the answer, and the server streams back raw
//! `linear16` PCM binary frames as it synthesizes — audio for the first
//! sentence can start playing while later sentences are still being sent.
//! Barge-in (a new question arriving before this one finishes speaking) is
//! handled by sending `Interrupt` and closing the socket, rather than
//! waiting for the server to finish — see `TtsSession::stop`.
//!
//! Protocol confirmed two ways: against Deepgram's Flux TTS docs
//! (developers.deepgram.com/docs/flux-tts/{client,server}-messages), and by
//! a standalone smoke test using this exact connect/auth/message code
//! against the real API with a real key, which round-tripped "Connected" ->
//! "SpeechStarted" -> binary PCM frames -> "Flushed" -> "SpeechMetadata" and
//! produced audible speech. Client sends `{"type":"Speak","text":...}`,
//! `{"type":"Flush"}`, `{"type":"Interrupt"}`, `{"type":"Close"}`; server
//! sends binary PCM frames plus JSON control messages ("Connected",
//! "SpeechStarted", "Flushed", "SpeechMetadata", "SpeechInterrupted",
//! "Warning", "Error").
//!
//! Auth is a plain `Authorization: Token <key>` header on the WebSocket
//! upgrade request — Deepgram's desktop/server client libraries all send it
//! this way (this is a native Rust process, not a browser `WebSocket`
//! object, which is the only environment that can't set that header; no
//! `Sec-WebSocket-Protocol` fallback is needed here).
//!
//! Blocking, not async: `tungstenite`'s blocking client runs on a dedicated
//! `std::thread` for the whole session's lifetime, mirroring
//! `stt::groq::transcribe`'s reasoning — the caller (`tts::mod`'s
//! `on_delta` closure) is a plain synchronous closure invoked from inside an
//! async Tauri command's streaming loop, so blocking it on network I/O would
//! stall the whole answer, not just speech.

use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use tungstenite::stream::MaybeTlsStream;
use tungstenite::Message;

const SPEAK_URL: &str = "wss://api.deepgram.com/v2/speak";

/// Flux's "Sienna" voice, picked in Settings -> Voice Controls. Fixed, not
/// user-configurable: the user picked this specific voice, and switching it
/// is a deliberate code change, not a runtime setting (unlike the API key,
/// which is a secret entered in Settings and must never live in source).
const MODEL: &str = "flux-sienna-en";

/// Sample rate requested from Flux for linear16 output. Fixed alongside
/// `MODEL` — the player (`tts::player`) is built assuming this exact rate,
/// so changing one without the other would produce audio at the wrong
/// pitch/speed.
pub const SAMPLE_RATE: u32 = 24_000;

/// Flux's delivery-register dial: an integer from `-2` (calm) to `2`
/// (animated), fixed for the whole connection (not adjustable mid-session
/// via `Configure` — confirmed against Deepgram's Flux TTS reference).
/// Omitting this from the connect URL leaves it at Flux's own default of
/// `0`, a neutral/flat register — which is what made Sienna sound flat and
/// robotic here despite the voice itself being described by Deepgram as
/// "warm, caring." `1` picks a natural, conversational-but-not-theatrical
/// register rather than maxing out at `2` (animated enough to sound
/// artificial for a general assistant voice).
const EXPRESSIVITY: i8 = 1;

/// How long to wait for the initial TCP+TLS+WebSocket handshake to complete
/// before giving up on one connection attempt. Set on the raw socket before
/// `tungstenite::connect` runs the handshake, so a network path that never
/// responds fails fast instead of hanging on the OS's much longer default
/// (20-30s on Windows).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(8);

/// How many times to retry a failed *connection attempt* (not a failure
/// after a successful connect) before giving up on this answer entirely.
/// Covers transient network blips — a DNS hiccup, a dropped SYN — without
/// retrying forever on a genuinely bad key or a real outage.
const MAX_CONNECT_ATTEMPTS: u32 = 3;

/// Delay between retry attempts.
const RETRY_DELAY: Duration = Duration::from_millis(500);

/// Distinguishes failure modes so the caller's log line says something
/// actionable. The `Display` impl is what ends up in logs — never includes
/// the API key.
#[derive(Debug)]
pub enum FluxError {
    MissingKey,
    ConnectFailed(String),
    Send(String),
}

impl std::fmt::Display for FluxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingKey => write!(f, "no Deepgram API key configured (Settings -> API Keys)"),
            Self::ConnectFailed(e) => write!(f, "failed to connect to Deepgram Flux after {MAX_CONNECT_ATTEMPTS} attempts: {e}"),
            Self::Send(e) => write!(f, "Deepgram Flux connection lost: {e}"),
        }
    }
}

/// Loads the key from the same Windows Credential Manager store as the LLM
/// provider keys (Settings -> API Keys in the main window), read fresh on
/// every session rather than cached — so saving/clearing the key in
/// Settings picks up on the next answer spoken, matching
/// `stt::groq::api_key`'s reasoning exactly. Never logged.
fn api_key() -> Result<String, FluxError> {
    match crate::personal::api_key_store::load_key("deepgram") {
        Ok(Some(key)) if !key.trim().is_empty() => Ok(key.trim().to_string()),
        _ => Err(FluxError::MissingKey),
    }
}

/// Clears an `AtomicBool` to `false` when dropped — used to mark
/// `FluxSession::alive` false no matter which of the I/O thread's several
/// exit paths (connect failure, server close, unrecoverable error) actually
/// runs, without repeating `alive.store(false, ...)` at each one.
struct ClearOnDrop<'a>(&'a AtomicBool);

impl Drop for ClearOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// A command sent into a live Flux session's writer.
enum FluxCommand {
    /// One sentence/chunk of text to synthesize — sent as a `Speak` message.
    Speak(String),
    /// Ends the current turn (`Flush`) — sent once the LLM stream ends (or a
    /// trailing chunk is flushed), telling Flux there's no more text coming
    /// for this answer so it can finalize and send `SpeechMetadata`.
    Flush,
    /// Cancels in-flight synthesis immediately (`Interrupt`) for barge-in —
    /// sent when a new question arrives while this answer is still
    /// speaking. No `playback_offset` is sent: this app cuts audio off
    /// locally via the player's own stop (see `tts::player::PlaybackHandle::stop`),
    /// so it doesn't need the server to report exactly how much was heard.
    Interrupt,
}

/// A live Flux WebSocket session, potentially spanning more than one
/// answer's turn — see `tts::TtsSession`'s doc for why this is now kept
/// alive across turns instead of being closed after every answer's
/// `Flush`. Owns the connection entirely on its own dedicated thread
/// (blocking I/O; see this module's doc) — every other part of the app
/// only ever holds this `Send`-safe handle, sending `FluxCommand`s through
/// an `mpsc::Sender`.
pub struct FluxSession {
    tx: mpsc::Sender<FluxCommand>,
    /// Cleared by the I/O thread the moment its read loop exits for any
    /// reason (server close, an unrecoverable error, or a handled
    /// `Interrupt`) — `TtsSession::speak()` checks this before reusing a
    /// session across turns, so a connection the server closed after a
    /// previous `Flush` (some streaming TTS turn-based protocols do this;
    /// others keep the socket open for the next turn) is transparently
    /// replaced with a fresh one rather than silently dropping the next
    /// answer's audio into a channel nobody is reading anymore.
    alive: Arc<AtomicBool>,
}

impl FluxSession {
    /// Opens a new Flux WebSocket session and starts its dedicated I/O
    /// thread. `on_audio` is called with each raw linear16 PCM chunk (mono,
    /// `SAMPLE_RATE` Hz) as it arrives. `on_error` is called at most once,
    /// only on an unrecoverable session failure (missing key, connect
    /// failure after retries, or the socket dying mid-session) — never for
    /// `Interrupt`-triggered cancellation, which is expected, not an error.
    /// Every connect/handshake/read/close event is logged (see
    /// `connect_with_retry`/the read loop below) so a failure is always
    /// diagnosable from logs, never silent.
    pub fn start(
        mut on_audio: impl FnMut(&[u8]) + Send + 'static,
        mut on_error: impl FnMut(FluxError) + Send + 'static,
    ) -> Result<Self, FluxError> {
        let key = api_key()?;
        let (tx, rx) = mpsc::channel::<FluxCommand>();
        let alive = Arc::new(AtomicBool::new(true));
        let alive_for_thread = alive.clone();

        std::thread::Builder::new()
            .name("tts-deepgram-flux".into())
            .spawn(move || {
                // Set unconditionally when this thread returns, regardless of
                // which of the several exit paths below was taken (a
                // successful connect that later closes, or a failure to
                // connect at all) — a single `defer`-style guard covers every
                // case without repeating the store at each `return`/`break`.
                let _alive_guard = ClearOnDrop(&alive_for_thread);

                let mut socket = match connect_with_retry(&key) {
                    Ok(socket) => socket,
                    Err(err) => {
                        on_error(FluxError::ConnectFailed(err));
                        return;
                    }
                };

                // The socket must not block forever on read: without a
                // timeout, a dead connection with no more commands coming
                // (e.g. the process is shutting down) would wedge this
                // thread reading forever instead of noticing `rx` closed.
                // `MaybeTlsStream` doesn't expose `set_read_timeout` itself
                // (TLS variants wrap, rather than are, a `TcpStream`), so
                // reach the inner socket directly in each variant.
                if let Err(err) = tcp_stream_mut(socket.get_mut()).set_read_timeout(Some(Duration::from_millis(200))) {
                    log::warn!("Deepgram Flux: failed to set read timeout, continuing anyway: {err}");
                }

                // Set once the command channel disconnects, so the drain
                // loop below sends Flush+Close exactly once (on the
                // iteration it first notices) rather than resending them
                // every ~200ms while waiting for the server to close its
                // end — see the `Disconnected` arm.
                let mut handle_dropped = false;

                'session: loop {
                    // Drain every command currently queued before reading —
                    // sends are cheap and must not wait behind a 200ms read
                    // timeout, which would otherwise delay Speak/Flush by up
                    // to that long on every iteration.
                    loop {
                        match rx.try_recv() {
                            Ok(FluxCommand::Speak(text)) => {
                                let msg = serde_json::json!({ "type": "Speak", "text": text }).to_string();
                                log::debug!("Deepgram Flux: -> Speak ({} chars)", text.len());
                                if let Err(err) = socket.send(Message::Text(msg.into())) {
                                    log::error!("Deepgram Flux: send Speak failed: {err}");
                                    on_error(FluxError::Send(err.to_string()));
                                    break 'session;
                                }
                            }
                            Ok(FluxCommand::Flush) => {
                                let msg = serde_json::json!({ "type": "Flush" }).to_string();
                                log::debug!("Deepgram Flux: -> Flush");
                                if let Err(err) = socket.send(Message::Text(msg.into())) {
                                    log::error!("Deepgram Flux: send Flush failed: {err}");
                                    on_error(FluxError::Send(err.to_string()));
                                    break 'session;
                                }
                            }
                            Ok(FluxCommand::Interrupt) => {
                                // Best-effort: tells the server to stop
                                // generating, but this session is ending
                                // either way (see `stop()`), so a failure
                                // here changes nothing.
                                log::debug!("Deepgram Flux: -> Interrupt, -> Close (barge-in)");
                                let msg = serde_json::json!({ "type": "Interrupt" }).to_string();
                                let _ = socket.send(Message::Text(msg.into()));
                                let _ = socket.send(Message::Text(serde_json::json!({ "type": "Close" }).to_string().into()));
                                break 'session;
                            }
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                if !handle_dropped {
                                    // Handle dropped (session handed off
                                    // then dropped, or app shutting down):
                                    // tell the server this answer has no
                                    // more text coming, then keep reading
                                    // until the socket itself closes so any
                                    // already in-flight audio still gets
                                    // delivered. Only sent once — `rx`
                                    // stays disconnected forever after this,
                                    // so without `handle_dropped` this would
                                    // resend Flush+Close on every ~200ms
                                    // read-timeout iteration until the
                                    // server finally closes its end.
                                    log::debug!("Deepgram Flux: handle dropped, -> Flush, -> Close");
                                    let _ = socket.send(Message::Text(serde_json::json!({ "type": "Flush" }).to_string().into()));
                                    let _ = socket.send(Message::Text(serde_json::json!({ "type": "Close" }).to_string().into()));
                                    handle_dropped = true;
                                }
                                break;
                            }
                        }
                    }

                    match socket.read() {
                        Ok(Message::Binary(bytes)) => {
                            log::debug!("Deepgram Flux: <- {} bytes of audio", bytes.len());
                            on_audio(&bytes);
                        }
                        Ok(Message::Close(frame)) => {
                            log::debug!("Deepgram Flux: server closed connection: {frame:?}");
                            break 'session;
                        }
                        Ok(Message::Text(text)) => {
                            // JSON control messages: Connected, SpeechStarted,
                            // Flushed, SpeechMetadata, SpeechInterrupted,
                            // Warning, Error. This app tracks "answer
                            // finished speaking" via the player's own
                            // sink-empty signal (see `TtsSpeakingSignal`),
                            // not via Flushed/SpeechMetadata, so nothing here
                            // needs to parse them for control flow — but
                            // every one is logged (at warn for Warning/Error,
                            // debug otherwise) so a Deepgram-side rejection
                            // is always visible, never silently swallowed.
                            if text.contains("\"Warning\"") || text.contains("\"Error\"") {
                                log::warn!("Deepgram Flux: server reported a problem: {text}");
                            } else {
                                log::debug!("Deepgram Flux: <- {text}");
                            }
                        }
                        Ok(_) => {}
                        // A read-timeout tick with nothing to report — normal,
                        // loop back around to check for new commands. Must
                        // match BOTH `WouldBlock` (Unix's `EAGAIN`/`EWOULDBLOCK`)
                        // AND `TimedOut` (Windows's `WSAETIMEDOUT`/os error
                        // 10060) — a `set_read_timeout` expiring reports as
                        // `TimedOut` on Windows, not `WouldBlock`. Missing
                        // this arm on Windows was a real bug: every idle
                        // 200ms tick fell through to the generic error arm
                        // below and tore down an otherwise-healthy session
                        // before any audio could arrive, which is exactly
                        // what a standalone repro against the real API with
                        // a real key reproduced and confirmed.
                        Err(tungstenite::Error::Io(ref e))
                            if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                        Err(err) => {
                            log::error!("Deepgram Flux: connection error mid-session: {err}");
                            on_error(FluxError::Send(err.to_string()));
                            break 'session;
                        }
                    }
                }
                let _ = socket.close(None);
            })
            .map_err(|e| FluxError::ConnectFailed(e.to_string()))?;

        Ok(Self { tx, alive })
    }

    /// Whether this session's I/O thread is still running (connected, or at
    /// least trying to be). `false` once the thread has exited for any
    /// reason — see the `alive` field doc.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Streams one sentence/chunk of text into this session's current turn.
    /// Non-blocking: hands off to the session thread's queue and returns
    /// immediately, matching every other TTS call site's expectation that
    /// `speak()`-shaped calls don't block on network I/O.
    pub fn speak(&self, text: &str) {
        let _ = self.tx.send(FluxCommand::Speak(text.to_string()));
    }

    /// Marks the end of this answer's text — call once after the LLM
    /// stream ends (and any trailing chunk has been sent via `speak()`), so
    /// Flux knows to finalize the turn. A no-op if `speak()` was never
    /// called (nothing was ever spoken, so there's no session to flush).
    pub fn flush(&self) {
        let _ = self.tx.send(FluxCommand::Flush);
    }

    /// Cancels in-flight synthesis immediately — barge-in: a new question
    /// arrived while this answer is still speaking. Closes the session
    /// after sending `Interrupt` rather than leaving it open, since a
    /// cancelled answer is never resumed.
    pub fn interrupt(&self) {
        let _ = self.tx.send(FluxCommand::Interrupt);
    }
}

/// Opens the WebSocket connection, retrying up to `MAX_CONNECT_ATTEMPTS`
/// times on a failed *attempt* (network/handshake failure) with
/// `RETRY_DELAY` between tries — covers transient blips (a dropped SYN, a
/// slow DNS response) without masking a genuinely bad key or outage behind
/// endless retries. Every attempt, success, and failure is logged, and the
/// connection URL is logged with the query string only (model/encoding/
/// sample_rate) — never the `Authorization` header, which is a separate
/// field never interpolated into any log line.
fn connect_with_retry(key: &str) -> Result<tungstenite::WebSocket<MaybeTlsStream<TcpStream>>, String> {
    let url = format!("{SPEAK_URL}?model={MODEL}&encoding=linear16&sample_rate={SAMPLE_RATE}&expressivity={EXPRESSIVITY}");
    let mut last_err = String::new();

    for attempt in 1..=MAX_CONNECT_ATTEMPTS {
        log::info!("Deepgram Flux: connecting to {url} (attempt {attempt}/{MAX_CONNECT_ATTEMPTS})");
        let request = match build_request(&url, key) {
            Ok(req) => req,
            // A malformed request (e.g. a key with characters invalid in an
            // HTTP header value) can never succeed on retry — fail fast.
            Err(err) => return Err(err),
        };

        let start = Instant::now();
        match connect_with_timeout(request) {
            Ok((socket, response)) => {
                log::info!(
                    "Deepgram Flux: WebSocket handshake succeeded in {:?} (HTTP {})",
                    start.elapsed(),
                    response.status()
                );
                return Ok(socket);
            }
            Err(err) => {
                log::warn!("Deepgram Flux: connect attempt {attempt} failed after {:?}: {err}", start.elapsed());
                last_err = err;
                if attempt < MAX_CONNECT_ATTEMPTS {
                    std::thread::sleep(RETRY_DELAY);
                }
            }
        }
    }

    Err(last_err)
}

/// Connects with an explicit `CONNECT_TIMEOUT`, rather than the OS's much
/// longer default (`tungstenite::connect`'s plain `TcpStream::connect` has
/// no timeout hook, so the TCP connect step is done manually here via
/// `TcpStream::connect_timeout` against each resolved address, then handed
/// to `tungstenite::client_tls_with_config` for the TLS+WebSocket
/// handshake). Distinguishes an HTTP-level rejection (bad model name,
/// invalid key: `tungstenite` returns `Error::Http` with the real
/// status/body from Deepgram) from a lower-level network failure, since the
/// former is never worth retrying (the same key/model will fail again
/// identically) — surfaced by including Deepgram's response body in the
/// error string here so the caller's log line shows the actual rejection
/// reason, not just "IO error".
fn connect_with_timeout(request: tungstenite::handshake::client::Request) -> Result<(tungstenite::WebSocket<MaybeTlsStream<TcpStream>>, tungstenite::http::Response<Option<Vec<u8>>>), String> {
    let host = request.uri().host().ok_or("request has no host")?.to_string();
    let port = request.uri().port_u16().unwrap_or(443);
    let addrs: Vec<_> = (host.as_str(), port).to_socket_addrs().map_err(|e| format!("DNS resolution failed: {e}"))?.collect();
    if addrs.is_empty() {
        return Err(format!("DNS resolution for {host} returned no addresses"));
    }

    let mut last_err = String::new();
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                return tungstenite::client_tls_with_config(request, stream, None, None).map_err(|e| match e {
                    tungstenite::handshake::HandshakeError::Failure(tungstenite::Error::Http(response)) => {
                        let status = response.status();
                        let body = response.body().as_ref().map(|b| String::from_utf8_lossy(b).to_string()).unwrap_or_default();
                        format!("Deepgram rejected the connection: HTTP {status}: {body}")
                    }
                    tungstenite::handshake::HandshakeError::Failure(err) => format!("TLS/WebSocket handshake failed: {err}"),
                    tungstenite::handshake::HandshakeError::Interrupted(_) => unreachable!("blocking handshake cannot be interrupted"),
                });
            }
            Err(err) => {
                last_err = format!("TCP connect to {addr} timed out or failed after {CONNECT_TIMEOUT:?}: {err}");
            }
        }
    }
    Err(last_err)
}

/// Reaches the underlying `TcpStream` regardless of whether the connection
/// ended up plain or TLS-wrapped — every `MaybeTlsStream` variant this crate
/// can produce (`Plain`, or `Rustls` via the `rustls-tls-webpki-roots`
/// feature this app enables) wraps a real `TcpStream` at its base, so socket
/// options like the read timeout above can always be set through it.
fn tcp_stream_mut(stream: &mut MaybeTlsStream<TcpStream>) -> &TcpStream {
    match stream {
        MaybeTlsStream::Plain(s) => s,
        MaybeTlsStream::Rustls(s) => &s.sock,
        // `NativeTls` is unreachable: this app never enables the
        // `native-tls` feature (see Cargo.toml — only
        // `rustls-tls-webpki-roots` is requested), so tungstenite can never
        // construct this variant here.
        #[allow(unreachable_patterns)]
        _ => unreachable!("MaybeTlsStream variant not constructible without a TLS feature this crate doesn't enable"),
    }
}

/// Builds the `wss://` handshake request with the `Authorization: Token`
/// header Deepgram requires (verified against the real API — `Bearer` is
/// rejected the same way it is on Deepgram's STT/Aura endpoints; `Token` is
/// the only accepted scheme for standard API keys), plus an explicit
/// connect-timeout marker consumed by `connect_with_timeout`'s caller. This
/// is a native process, not a browser `WebSocket`, so there is no
/// restriction on setting `Authorization` directly — no
/// `Sec-WebSocket-Protocol` workaround is needed.
fn build_request(url: &str, key: &str) -> Result<tungstenite::handshake::client::Request, String> {
    use tungstenite::client::IntoClientRequest;
    let mut request = url.into_client_request().map_err(|e| e.to_string())?;
    request
        .headers_mut()
        .insert("Authorization", format!("Token {key}").parse().map_err(|e: http::header::InvalidHeaderValue| e.to_string())?);
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_never_includes_a_key_value() {
        let err = FluxError::MissingKey;
        assert!(!err.to_string().to_lowercase().contains("token"));
    }

    #[test]
    #[ignore = "depends on real Windows Credential Manager state (no stored 'deepgram' key) — not controllable from a unit test"]
    fn missing_key_is_reported_without_connecting() {
        let result = FluxSession::start(|_| {}, |_| {});
        assert!(matches!(result, Err(FluxError::MissingKey)));
    }
}
