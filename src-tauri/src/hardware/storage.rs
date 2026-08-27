//! Best-effort SSD vs HDD detection for the system drive, via the same
//! "seek penalty" query Windows itself uses internally (Disk Defragmenter,
//! Task Manager's storage type column). Weighted lightly in tier scoring —
//! STT/RAG are CPU/RAM-bound in steady state, not disk-I/O-bound; storage
//! type only affects one-time model/document load latency.

use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::ptr;

use windows::Win32::Foundation::{CloseHandle, GENERIC_READ, HANDLE};
use windows::Win32::Storage::FileSystem::{
    CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
};
use windows::Win32::System::Ioctl::{
    StorageDeviceSeekPenaltyProperty, DEVICE_SEEK_PENALTY_DESCRIPTOR, IOCTL_STORAGE_QUERY_PROPERTY,
    PropertyStandardQuery, STORAGE_PROPERTY_QUERY,
};
use windows::Win32::System::IO::DeviceIoControl;

/// Queries the physical drive backing `C:\` (Smallbird's app data / model /
/// working files all live under the system drive in practice). Returns
/// `None` on any failure — permission denial, an unusual storage stack
/// (e.g. some virtualized/cloud-mounted drives), or an unsupported query —
/// so callers must treat `None` as "unknown", not "HDD".
pub fn detect_system_drive_is_ssd() -> Option<bool> {
    // SAFETY: standard CreateFileW + DeviceIoControl sequence for
    // IOCTL_STORAGE_QUERY_PROPERTY, matching Microsoft's documented usage.
    // No admin privileges are required for this specific query (unlike raw
    // sector access). Every failure path returns `None` rather than
    // unwrapping or panicking.
    unsafe {
        let path: Vec<u16> = std::ffi::OsStr::new("\\\\.\\PhysicalDrive0")
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let handle: HANDLE = CreateFileW(
            windows::core::PCWSTR(path.as_ptr()),
            GENERIC_READ.0,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            windows::Win32::Storage::FileSystem::FILE_FLAGS_AND_ATTRIBUTES(0),
            None,
        )
        .ok()?;

        let query = STORAGE_PROPERTY_QUERY {
            PropertyId: StorageDeviceSeekPenaltyProperty,
            QueryType: PropertyStandardQuery,
            AdditionalParameters: [0],
        };

        let mut descriptor = DEVICE_SEEK_PENALTY_DESCRIPTOR::default();
        let mut bytes_returned: u32 = 0;

        let ok = DeviceIoControl(
            handle,
            IOCTL_STORAGE_QUERY_PROPERTY,
            Some(&query as *const _ as *const c_void),
            std::mem::size_of::<STORAGE_PROPERTY_QUERY>() as u32,
            Some(&mut descriptor as *mut _ as *mut c_void),
            std::mem::size_of::<DEVICE_SEEK_PENALTY_DESCRIPTOR>() as u32,
            Some(&mut bytes_returned),
            Some(ptr::null_mut()),
        );

        let _ = CloseHandle(handle);

        match ok {
            Ok(()) => Some(!descriptor.IncursSeekPenalty),
            Err(_) => None,
        }
    }
}
