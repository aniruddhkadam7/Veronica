mod actions;
mod analyzer;
// Public so the `audio_probe` binary (src/bin/audio_probe.rs) can drive the
// real capture path when measuring the pipeline, rather than a copy of it that
// could drift from what actually ships.
pub mod audio;
mod backend;
mod commands;
// Public so the RAG/STT spawn call sites and the `rag-bench`/`stt-bench`
// verification steps (Milestone 4a) can reference `PerformanceConfig`
// without a re-export chain.
pub mod hardware;
mod history;
mod interview_mode;
mod main_window;
mod meeting_mode;
mod notes_mode;
mod overlay_window;
mod personal;
mod process_util;
mod rag;
mod state;
mod tls_init;
mod veronica;
mod voice_command;
// Public alongside `audio` so `src/bin/pipeline_test.rs` can drive the real
// recording pipeline headlessly instead of a reimplementation of it.
pub mod stt;
pub mod transcript;
mod windows_capture_protection;

use tauri::Manager;

use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();
    tls_init::ensure_installed();

    tauri::Builder::default()
        // Must be the very first plugin registered (Tauri requirement). A
        // second launch of the app (double-clicking the exe/shortcut again,
        // including while the first launch is still mid-startup and its
        // window isn't visible yet) is redirected here instead of spawning a
        // second, fully separate process — the new invocation exits
        // immediately and this callback just brings the existing app to the
        // front instead.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            let focus_target = app
                .get_webview_window(interview_mode::OVERLAY_WINDOW_LABEL)
                .filter(|w| w.is_visible().unwrap_or(false))
                .or_else(|| app.get_webview_window(main_window::MAIN_WINDOW_LABEL));
            if let Some(window) = focus_target {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    // Only react on key-down, not key-up, so holding the
                    // combination doesn't repeatedly toggle the overlay.
                    if event.state() != tauri_plugin_global_shortcut::ShortcutState::Pressed {
                        return;
                    }
                    if shortcut.matches(
                        tauri_plugin_global_shortcut::Modifiers::CONTROL
                            | tauri_plugin_global_shortcut::Modifiers::SHIFT,
                        tauri_plugin_global_shortcut::Code::KeyV,
                    ) {
                        // Veronica's shortcut only makes sense while the
                        // Interview Mode overlay is open (mic assistant state
                        // lives there) — the frontend does the actual
                        // start/stop_mic_assistant call so its React state
                        // (micAssistantActive) stays in sync with what the
                        // mic button shows; this just forwards the key press
                        // as an event.
                        use tauri::Emitter;
                        let _ = app.emit("veronica:toggle-shortcut", ());
                        return;
                    }
                    if let Err(err) = interview_mode::toggle_overlay_window(app) {
                        log::warn!("failed to toggle Interview Mode overlay via hotkey: {err}");
                    }
                })
                .build(),
        )
        .manage(AppState::default())
        .manage(history::HistoryStore::default())
        .manage(meeting_mode::history::MeetingHistoryStore::default())
        .manage(hardware::stt_rag_coordination::SttRagCoordination::default())
        .manage(voice_command::MicAssistantSession::default())
        .setup(|app| {
            // Hardware detection + performance mode load first, synchronously
            // — cheap (a few sysinfo/DXGI/IOCTL calls, no network/process
            // spawn), and the RAG service spawn below reads the resulting
            // tier config for its embedding batch size / torch thread count.
            let performance_state = hardware::init(&app.handle().clone());
            let initial_embed_config = {
                let cfg = performance_state.0.lock().unwrap().effective_config();
                rag::EmbedProcessConfig {
                    embed_batch_size: cfg.rag_embed_batch_size,
                    torch_threads: cfg.rag_torch_threads,
                }
            };
            app.manage(performance_state);

            main_window::apply_light_titlebar(&app.handle().clone());
            main_window::position_top_center(&app.handle().clone());

            use tauri_plugin_global_shortcut::GlobalShortcutExt;
            // Ctrl+Shift+Space: show/hide the Interview Mode overlay (spec
            // section 19/21). Registration failure (e.g. the combination is
            // already claimed by another application) is logged, not fatal —
            // the overlay remains reachable from the main window's button.
            if let Err(err) = app.global_shortcut().register("Ctrl+Shift+Space") {
                log::warn!("failed to register Ctrl+Shift+Space global hotkey: {err}");
            }
            // Ctrl+Shift+V: toggle Veronica (mic assistant) listening without
            // touching the overlay — see the handler above and
            // InterviewOverlay.tsx's "veronica:toggle-shortcut" listener.
            if let Err(err) = app.global_shortcut().register("Ctrl+Shift+V") {
                log::warn!("failed to register Ctrl+Shift+V global hotkey: {err}");
            }
            // Spawn the local RAG service in the background; the app remains
            // usable for recording/transcription immediately even while this
            // is still starting up or if it's unavailable entirely (see
            // rag::process::RagServiceHandle::spawn).
            let handle = app.handle().clone();
            let handle_for_spawn = handle.clone();
            std::thread::spawn(move || match rag::RagServiceHandle::spawn(initial_embed_config, Some(&handle_for_spawn)) {
                Ok(service) => {
                    if service.is_some() {
                        rag::wait_until_healthy_default();
                    }
                    let state = handle.state::<AppState>();
                    let lock_result = state.rag_service.lock();
                    if let Ok(mut slot) = lock_result {
                        *slot = service;
                    }
                }
                Err(err) => {
                    log::error!("failed to start RAG service: {err}");
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_output_devices,
            commands::list_input_devices,
            commands::start_system_audio_capture,
            commands::pause_recording,
            commands::resume_recording,
            commands::stop_audio_capture,
            commands::get_current_session,
            commands::get_recording_state,
            commands::start_new_interview,
            commands::clear_transcript,
            commands::export_transcript_txt,
            commands::export_transcript_json,
            commands::check_backend_connection,
            commands::analyze_interview,
            commands::check_rag_connection,
            commands::analyze_setup_context,
            commands::upload_document,
            commands::get_document_text,
            commands::list_documents,
            commands::delete_document,
            commands::clear_knowledge_base,
            commands::knowledge_base_status,
            commands::search_knowledge_base,
            interview_mode::commands::show_interview_overlay,
            interview_mode::commands::hide_interview_overlay,
            interview_mode::commands::toggle_interview_overlay,
            interview_mode::commands::set_overlay_always_on_top,
            interview_mode::commands::resize_interview_overlay,
            interview_mode::commands::start_backend_session,
            interview_mode::commands::end_backend_session,
            veronica::ask_veronica,
            veronica::set_mode,
            veronica::get_mode,
            actions::run_veronica_action,
            hardware::commands::get_stt_mode_info,
            history::list_interview_history,
            history::archive_interview_session,
            history::delete_interview_history_entry,
            meeting_mode::commands::track_meeting_item,
            meeting_mode::commands::clear_meeting_session,
            meeting_mode::commands::end_meeting,
            meeting_mode::history::list_meeting_history,
            meeting_mode::history::archive_meeting,
            meeting_mode::history::delete_meeting_history_entry,
            notes_mode::store::list_notes,
            notes_mode::store::get_note,
            notes_mode::store::create_note,
            notes_mode::store::update_note,
            notes_mode::store::delete_note,
            notes_mode::store::search_notes,
            notes_mode::dictation::start_note_dictation,
            notes_mode::dictation::stop_note_dictation,
            notes_mode::ai::summarize_note,
            notes_mode::ai::ask_about_notes,
            hardware::commands::get_hardware_profile,
            hardware::commands::get_performance_mode,
            hardware::commands::set_performance_mode,
            main_window::set_popover_content_height,
            personal::commands::personal_get_api_key,
            personal::commands::personal_set_api_key,
            personal::commands::personal_clear_api_key,
            voice_command::start_mic_assistant,
            voice_command::stop_mic_assistant,
            voice_command::launch_app,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
