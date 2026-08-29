// pub(crate) (not private) so `personal::agent`'s tool loop can reach
// `Capability`/`execute_tool`/`fast_router` without a re-export chain —
// still invisible outside this crate (the `bin/*.rs` diagnostic binaries
// depend on `desktop_lib` as an external crate and have no need for it).
pub(crate) mod actions;
mod analyzer;
// Public so the `audio_probe` binary (src/bin/audio_probe.rs) can drive the
// real capture path when measuring the pipeline, rather than a copy of it that
// could drift from what actually ships.
pub mod audio;
mod backend;
mod commands;
mod conversation;
mod http_client;
mod interrupt;
mod language;
// Public so the RAG/STT spawn call sites and the `rag-bench`/`stt-bench`
// verification steps (Milestone 4a) can reference `PerformanceConfig`
// without a re-export chain.
pub mod hardware;
mod main_window;
mod notes_mode;
mod overlay_window;
mod personal;
mod process_util;
mod rag;
mod state;
mod tls_init;
mod tray;
mod veronica;
mod veronica_widget;
mod veronica_window;
mod voice_command;
mod working_state;
// Public alongside `audio` so `src/bin/pipeline_test.rs` can drive the real
// recording pipeline headlessly instead of a reimplementation of it.
pub mod stt;
pub mod transcript;
// Public alongside `audio`/`stt` so headless test binaries
// (src/bin/pipeline_test*.rs) can construct a `tts::TtsSpeakingSignal` to
// pass into `audio::run_stt_pipeline`'s signature, matching that function's
// existing PauseSignal parameter — those binaries have no real TTS session,
// only a `TtsSpeakingSignal::default()` that never becomes true.
pub mod tts;
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
                .get_webview_window(veronica_window::OVERLAY_WINDOW_LABEL)
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
                    if shortcut.matches(
                        tauri_plugin_global_shortcut::Modifiers::CONTROL
                            | tauri_plugin_global_shortcut::Modifiers::SHIFT,
                        tauri_plugin_global_shortcut::Code::Space,
                    ) && tray::app_was_fully_hidden(app)
                    {
                        // The app was closed to tray (no window visible) —
                        // this keypress is "open Veronica from nothing", so
                        // she should also greet, not just silently appear.
                        // See veronica_window::wake_veronica.
                        veronica_window::wake_veronica(app);
                        return;
                    }
                    if let Err(err) = veronica_window::toggle_overlay_window_sync(app) {
                        log::warn!("failed to toggle Veronica overlay via hotkey: {err}");
                    }
                })
                .build(),
        )
        .manage(AppState::default())
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

            if let Err(err) = tray::setup(&app.handle().clone()) {
                log::warn!("failed to set up tray icon: {err}");
            }

            // Closing the main window (the titlebar X, or Alt+F4) hides it
            // to the tray instead of quitting the whole app — the global
            // hotkeys (and the ability to bring Veronica back at all) only
            // work while the process keeps running, so "closing" the window
            // must not kill the process. The tray menu's "Quit" is the only
            // way to actually exit (see tray.rs).
            if let Some(main_window) = app.get_webview_window(main_window::MAIN_WINDOW_LABEL) {
                let window_to_hide = main_window.clone();
                main_window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = window_to_hide.hide();
                    }
                });
            }

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
            commands::get_selected_devices,
            commands::set_input_device,
            commands::set_output_device,
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
            veronica_window::show_interview_overlay,
            veronica_window::hide_interview_overlay,
            veronica_window::toggle_interview_overlay,
            veronica_window::set_overlay_always_on_top,
            veronica_window::resize_interview_overlay,
            veronica_window::start_backend_session,
            veronica_window::end_backend_session,
            veronica::ask_veronica,
            veronica::speak_greeting,
            veronica::stop_speaking,
            veronica::try_interrupt,
            veronica::get_conversation_history,
            veronica::reset_conversation,
            veronica_widget::show_veronica_widget,
            veronica_widget::hide_veronica_widget,
            veronica_widget::resize_veronica_widget,
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
            voice_command::set_mic_muted,
            voice_command::get_mic_muted,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
