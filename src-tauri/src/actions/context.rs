//! Derives `WorkingState` context updates (current app/window/file/folder)
//! from a just-executed `Capability` — one small match, called from both
//! dispatch sites (`veronica.rs`'s fast-router arm and the agent loop's
//! tool-call loop) right after `WorkingState::record_action`, so "open it"/
//! "delete that file" resolve against whatever was actually just touched
//! regardless of which path executed it.

use std::path::Path;

use super::capability::{Capability, FileOp, StorageOp, StorageQuery, WindowOp, WindowQueryOp};

/// What to write into `WorkingState` after a capability ran — every field is
/// `Option` and `WorkingState::note_context` only overwrites the ones that
/// are `Some`, so an action that says nothing about (e.g.) the current file
/// leaves it untouched rather than clearing it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ContextUpdate {
    pub app: Option<String>,
    pub window: Option<String>,
    pub file: Option<String>,
    pub folder: Option<String>,
}

fn parent_folder(path: &str) -> Option<String> {
    Path::new(path).parent().filter(|p| !p.as_os_str().is_empty()).map(|p| p.display().to_string())
}

pub fn derive_context_updates(capability: &Capability) -> ContextUpdate {
    match capability {
        Capability::LaunchOrFocusApp(name) => ContextUpdate { app: Some(name.clone()), ..Default::default() },
        Capability::LaunchAppWithArg { app, arg } => {
            // `arg` is usually a folder/project path (e.g. "open VS Code and
            // open my security project") rather than a single file — set
            // both app and folder so a later "read the auth code in there"
            // has a folder to search under.
            ContextUpdate { app: Some(app.clone()), folder: Some(arg.clone()), ..Default::default() }
        }
        Capability::WindowOp { target: Some(target), op } if *op == WindowOp::Focus => ContextUpdate { window: Some(target.clone()), ..Default::default() },
        Capability::WindowQuery(WindowQueryOp::GetActive) => ContextUpdate::default(), // the result text names the window, but we don't parse it back out here — GetActive is a read, not a navigation
        Capability::FileOp(FileOp::CreateFile { path, .. }) | Capability::FileOp(FileOp::WriteFile { path, .. }) | Capability::FileOp(FileOp::ReadFile { path }) => {
            ContextUpdate { file: Some(path.clone()), folder: parent_folder(path), ..Default::default() }
        }
        Capability::FileOp(FileOp::CreateFolder { path }) | Capability::StorageQuery(StorageQuery::ListFolder { path }) => {
            ContextUpdate { folder: Some(path.clone()), ..Default::default() }
        }
        Capability::StorageOp(StorageOp::MoveOrRename { to, .. }) => ContextUpdate { file: Some(to.clone()), folder: parent_folder(to), ..Default::default() },
        _ => ContextUpdate::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launching_an_app_sets_current_app() {
        let update = derive_context_updates(&Capability::LaunchOrFocusApp("VS Code".to_string()));
        assert_eq!(update.app.as_deref(), Some("VS Code"));
    }

    #[test]
    fn creating_a_file_sets_current_file_and_folder() {
        let update = derive_context_updates(&Capability::FileOp(FileOp::CreateFile { path: r"C:\proj\notes.txt".to_string(), content: None }));
        assert_eq!(update.file.as_deref(), Some(r"C:\proj\notes.txt"));
        assert_eq!(update.folder.as_deref(), Some(r"C:\proj"));
    }

    #[test]
    fn launch_app_with_arg_sets_both_app_and_folder() {
        let update = derive_context_updates(&Capability::LaunchAppWithArg { app: "VS Code".to_string(), arg: r"C:\proj\security".to_string() });
        assert_eq!(update.app.as_deref(), Some("VS Code"));
        assert_eq!(update.folder.as_deref(), Some(r"C:\proj\security"));
    }

    #[test]
    fn unrelated_capability_produces_an_empty_update() {
        let update = derive_context_updates(&Capability::SystemInfo(crate::actions::capability::SystemInfoKind::Cpu));
        assert_eq!(update, ContextUpdate::default());
    }
}
