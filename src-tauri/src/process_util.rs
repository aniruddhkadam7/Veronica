use std::ffi::OsStr;
use std::process::Command;

/// Windows `CREATE_NO_WINDOW` process creation flag. Without it, spawning a
/// console subprocess (Python, `py -3 --version`, etc.) from this GUI app
/// briefly flashes a visible CMD/PowerShell-style console window on screen —
/// redirecting stdio (`Stdio::null()`/`Stdio::piped()`) does NOT suppress
/// that window, only this flag does.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// `Command::new` with the console window suppressed on Windows. Every
/// subprocess this app spawns (STT sidecar, RAG service, Python version
/// probes, ...) should be built from this instead of `Command::new` directly.
pub fn hidden_command(program: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}
