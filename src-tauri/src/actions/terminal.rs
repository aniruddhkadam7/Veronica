//! Terminal command execution — `cmd /C <command>`, run on a dedicated
//! `std::thread` (matching this crate's convention for blocking work, e.g.
//! `stt::groq`, since `tokio` here only has the `"time"` feature, not
//! `"rt"`) rather than blocking the async executor. Risk is decided
//! entirely by `registry::classify_command_risk` before this module is ever
//! reached — `run_command` itself is unconditional execution once called,
//! same as every other tool.

use std::io::Read;
use std::process::Stdio;
use std::sync::mpsc;
use std::time::Duration;

use crate::process_util::hidden_command;
use crate::state::CancelToken;

/// Output is truncated to this many characters before being returned, so a
/// runaway or very verbose command can't blow out the tool-result payload
/// handed back to the model.
const OUTPUT_MAX_CHARS: usize = 4000;

/// How often the polling loop checks the cancel token / child exit status.
const POLL_INTERVAL: Duration = Duration::from_millis(150);

pub async fn run_command(command: &str, working_dir: Option<&str>, cancel: &CancelToken) -> Result<String, String> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err("no command given".into());
    }

    let mut cmd = hidden_command("cmd");
    cmd.args(["/C", trimmed]).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(dir) = working_dir {
        if !dir.trim().is_empty() {
            cmd.current_dir(dir.trim());
        }
    }

    let mut child = cmd.spawn().map_err(|e| format!("couldn't start command: {e}"))?;
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();

    // Poll on a dedicated thread so this async fn doesn't block the
    // executor, and so the cancel token can be checked periodically —
    // killing the child if a newer turn supersedes this one, rather than
    // leaving it running unattended.
    let (tx, rx) = mpsc::channel();
    let cancel = cancel.clone();
    std::thread::spawn(move || {
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let mut out = String::new();
                    let mut err = String::new();
                    if let Some(mut s) = stdout.take() {
                        let _ = s.read_to_string(&mut out);
                    }
                    if let Some(mut s) = stderr.take() {
                        let _ = s.read_to_string(&mut err);
                    }
                    let _ = tx.send(Ok((status.code().unwrap_or(-1), out, err)));
                    return;
                }
                Ok(None) => {
                    if cancel.is_cancelled() {
                        let _ = child.kill();
                        let _ = tx.send(Err("cancelled".to_string()));
                        return;
                    }
                    std::thread::sleep(POLL_INTERVAL);
                }
                Err(e) => {
                    let _ = tx.send(Err(format!("error waiting for command: {e}")));
                    return;
                }
            }
        }
    });

    let (exit_code, stdout_text, stderr_text) = match rx.recv() {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("command execution thread ended unexpectedly".to_string()),
    };

    let mut combined = String::new();
    if !stdout_text.trim().is_empty() {
        combined.push_str(stdout_text.trim());
    }
    if !stderr_text.trim().is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(stderr_text.trim());
    }
    if combined.len() > OUTPUT_MAX_CHARS {
        combined.truncate(OUTPUT_MAX_CHARS);
        combined.push_str("\n...(truncated)");
    }

    if exit_code == 0 {
        Ok(if combined.is_empty() { "Command completed with no output.".to_string() } else { combined })
    } else {
        Err(format!("Command exited with code {exit_code}.{}", if combined.is_empty() { String::new() } else { format!(" Output: {combined}") }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_command_captures_stdout_and_exit_code() {
        tauri::async_runtime::block_on(async {
            let cancel = CancelToken::new();
            let result = run_command("echo hello", None, &cancel).await.unwrap();
            assert!(result.contains("hello"));
        });
    }

    #[test]
    fn run_command_nonzero_exit_reports_failure_text_not_panic() {
        tauri::async_runtime::block_on(async {
            let cancel = CancelToken::new();
            let result = run_command("exit 3", None, &cancel).await;
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("3"));
        });
    }

    #[test]
    fn empty_command_is_rejected() {
        tauri::async_runtime::block_on(async {
            let cancel = CancelToken::new();
            assert!(run_command("", None, &cancel).await.is_err());
        });
    }
}
