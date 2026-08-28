//! Primary-monitor screen capture via GDI `BitBlt`, encoded to PNG — backs
//! the `CaptureScreen` agent-loop tool (see `capability.rs`'s doc: this is
//! always paired with an LLM call to reason over the image, so it's never a
//! fast-router target on its own).

use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits, ReleaseDC, SelectObject,
    BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, SRCCOPY,
};

use super::native::screen_width_height;

/// Captures the primary monitor and returns PNG-encoded bytes. Synchronous
/// GDI work (typically tens of milliseconds for a full-HD/QHD screen) — no
/// network, no async needed.
pub fn capture_primary_screen_png() -> Result<Vec<u8>, String> {
    let (width, height) = screen_width_height();
    if width <= 0 || height <= 0 {
        return Err("couldn't read the screen size".to_string());
    }

    let mut buffer = vec![0u8; (width as usize) * (height as usize) * 4];

    unsafe {
        let screen_dc = GetDC(None);
        if screen_dc.is_invalid() {
            return Err("couldn't get the screen device context".to_string());
        }
        let mem_dc = CreateCompatibleDC(Some(screen_dc));
        let bitmap = CreateCompatibleBitmap(screen_dc, width, height);
        let previous = SelectObject(mem_dc, bitmap.into());

        let blit_ok = BitBlt(mem_dc, 0, 0, width, height, Some(screen_dc), 0, 0, SRCCOPY);

        let mut bmi = BITMAPINFO::default();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = -height; // negative = top-down DIB, matching how we read it below
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB.0 as u32;

        let scan_result = GetDIBits(mem_dc, bitmap, 0, height as u32, Some(buffer.as_mut_ptr() as *mut _), &mut bmi, DIB_RGB_COLORS);

        let _ = SelectObject(mem_dc, previous);
        let _ = DeleteObject(bitmap.into());
        let _ = DeleteDC(mem_dc);
        ReleaseDC(None, screen_dc);

        if blit_ok.is_err() || scan_result == 0 {
            return Err("failed to capture the screen".to_string());
        }
    }

    // BGRA (what GetDIBits produced) -> RGBA (what `image` expects).
    for chunk in buffer.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    let img = image::RgbaImage::from_raw(width as u32, height as u32, buffer)
        .ok_or_else(|| "failed to build the captured image".to_string())?;
    let mut png_bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .map_err(|e| format!("failed to encode screenshot as PNG: {e}"))?;
    Ok(png_bytes)
}
