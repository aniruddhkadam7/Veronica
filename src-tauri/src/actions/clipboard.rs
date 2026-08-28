//! Clipboard read/write via raw Win32 (`OpenClipboard`/`SetClipboardData`/
//! `GetClipboardData` with `CF_UNICODETEXT`) rather than adding a clipboard
//! crate — matches `native.rs`'s "native Win32 calls only" convention for
//! this module.

use windows::Win32::Foundation::HANDLE;
use windows::Win32::System::DataExchange::{CloseClipboard, EmptyClipboard, GetClipboardData, OpenClipboard, SetClipboardData};
use windows::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE};

const CF_UNICODETEXT: u32 = 13;

pub fn write_text(text: &str) -> Result<String, String> {
    unsafe {
        OpenClipboard(None).map_err(|e| format!("couldn't access the clipboard: {e}"))?;
        let result = write_text_inner(text);
        let _ = CloseClipboard();
        result
    }
}

unsafe fn write_text_inner(text: &str) -> Result<String, String> {
    let _ = EmptyClipboard();
    let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
    let byte_len = wide.len() * std::mem::size_of::<u16>();

    let hmem = GlobalAlloc(GMEM_MOVEABLE, byte_len).map_err(|e| format!("failed to allocate clipboard memory: {e}"))?;
    let ptr = GlobalLock(hmem);
    if ptr.is_null() {
        return Err("failed to lock clipboard memory".to_string());
    }
    std::ptr::copy_nonoverlapping(wide.as_ptr(), ptr as *mut u16, wide.len());
    let _ = GlobalUnlock(hmem);

    SetClipboardData(CF_UNICODETEXT, Some(HANDLE(hmem.0))).map_err(|e| format!("failed to set clipboard data: {e}"))?;
    Ok("Copied.".to_string())
}

pub fn read_text() -> Result<String, String> {
    unsafe {
        OpenClipboard(None).map_err(|e| format!("couldn't access the clipboard: {e}"))?;
        let result = read_text_inner();
        let _ = CloseClipboard();
        result
    }
}

unsafe fn read_text_inner() -> Result<String, String> {
    let handle = GetClipboardData(CF_UNICODETEXT).map_err(|_| "the clipboard doesn't have text on it".to_string())?;
    let ptr = GlobalLock(windows::Win32::Foundation::HGLOBAL(handle.0)) as *const u16;
    if ptr.is_null() {
        return Err("the clipboard doesn't have text on it".to_string());
    }
    let mut len = 0usize;
    while *ptr.add(len) != 0 {
        len += 1;
    }
    let text = String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len));
    let _ = GlobalUnlock(windows::Win32::Foundation::HGLOBAL(handle.0));
    Ok(text)
}
