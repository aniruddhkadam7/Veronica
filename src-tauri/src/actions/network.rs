//! Network queries: adapter status via `sysinfo` (already a dependency —
//! avoids a new Win32 feature for the common case), a host ping via the
//! Win32 ICMP API (`IcmpSendEcho`, not a shelled-out `ping.exe`), and
//! listening TCP ports via `GetExtendedTcpTable` (`IPHLPAPI`) — useful for
//! diagnosing "why is Docker unhealthy" (are its expected ports actually
//! listening?).

use std::net::Ipv4Addr;

use windows::Win32::NetworkManagement::IpHelper::{
    GetExtendedTcpTable, IcmpCloseHandle, IcmpCreateFile, IcmpSendEcho, ICMP_ECHO_REPLY, MIB_TCPROW_OWNER_PID, MIB_TCPTABLE_OWNER_PID, TCP_TABLE_OWNER_PID_ALL,
};
use windows::Win32::Networking::WinSock::AF_INET;

pub fn network_status() -> Result<String, String> {
    let networks = sysinfo::Networks::new_with_refreshed_list();
    if networks.is_empty() {
        return Ok("No network adapters found.".to_string());
    }
    let mut lines = Vec::new();
    for (name, data) in networks.iter() {
        let up = data.total_received() > 0 || data.total_transmitted() > 0;
        lines.push(format!("{name}: {}", if up { "active" } else { "idle" }));
    }
    Ok(format!("Network adapters — {}.", lines.join(", ")))
}

/// ICMP echo (ping) via the Win32 API — resolves `host` to an IPv4 address
/// first (a bare `std::net::ToSocketAddrs` DNS lookup, not a shell call).
pub fn ping_host(host: &str) -> Result<String, String> {
    let trimmed = host.trim();
    if trimmed.is_empty() {
        return Err("no host given".into());
    }
    let addr: Ipv4Addr = resolve_ipv4(trimmed).ok_or_else(|| format!("couldn't resolve \"{trimmed}\""))?;

    unsafe {
        let handle = IcmpCreateFile().map_err(|e| format!("couldn't start ping: {e}"))?;
        let send_data = b"veronica-ping";
        let mut reply_buffer = vec![0u8; std::mem::size_of::<ICMP_ECHO_REPLY>() + send_data.len() + 8];
        let dest: u32 = u32::from_ne_bytes(addr.octets());

        let result = IcmpSendEcho(handle, dest, send_data.as_ptr() as *const _, send_data.len() as u16, None, reply_buffer.as_mut_ptr() as *mut _, reply_buffer.len() as u32, 4000);
        let _ = IcmpCloseHandle(handle);

        if result == 0 {
            return Err(format!("\"{trimmed}\" didn't respond to ping."));
        }
        let reply = &*(reply_buffer.as_ptr() as *const ICMP_ECHO_REPLY);
        Ok(format!("\"{trimmed}\" responded in {} ms.", reply.RoundTripTime))
    }
}

fn resolve_ipv4(host: &str) -> Option<Ipv4Addr> {
    use std::net::ToSocketAddrs;
    (host, 0)
        .to_socket_addrs()
        .ok()?
        .filter_map(|addr| match addr {
            std::net::SocketAddr::V4(v4) => Some(*v4.ip()),
            _ => None,
        })
        .next()
}

/// Lists locally listening TCP ports (state `LISTEN`) via
/// `GetExtendedTcpTable` — useful to check whether a service (e.g. Docker's
/// expected port) is actually bound.
pub fn listening_ports() -> Result<String, String> {
    unsafe {
        let mut size: u32 = 0;
        let _ = GetExtendedTcpTable(None, &mut size, false, AF_INET.0 as u32, TCP_TABLE_OWNER_PID_ALL, 0);
        if size == 0 {
            return Ok("No listening ports found.".to_string());
        }
        let mut buffer = vec![0u8; size as usize];
        let result = GetExtendedTcpTable(Some(buffer.as_mut_ptr() as *mut _), &mut size, false, AF_INET.0 as u32, TCP_TABLE_OWNER_PID_ALL, 0);
        if result != 0 {
            return Err(format!("couldn't read the TCP table (error code {result})"));
        }

        let table = &*(buffer.as_ptr() as *const MIB_TCPTABLE_OWNER_PID);
        let count = table.dwNumEntries as usize;
        let rows_ptr = table.table.as_ptr();
        let rows = std::slice::from_raw_parts(rows_ptr, count);

        // MIB_TCP_STATE_LISTEN == 2
        const MIB_TCP_STATE_LISTEN: u32 = 2;
        let listening: Vec<String> = rows
            .iter()
            .filter(|row: &&MIB_TCPROW_OWNER_PID| row.dwState == MIB_TCP_STATE_LISTEN)
            .map(|row| {
                let port = u16::from_be(row.dwLocalPort as u16);
                format!("port {port} (pid {})", row.dwOwningPid)
            })
            .collect();

        if listening.is_empty() {
            Ok("No listening ports found.".to_string())
        } else {
            Ok(format!("Listening ports: {}.", listening.join(", ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_status_does_not_panic_and_returns_text() {
        // Environment-dependent (adapter count varies by machine), so just
        // assert it completes without panicking and returns something.
        let result = network_status();
        assert!(result.is_ok());
    }

    #[test]
    fn listening_ports_does_not_panic() {
        let result = listening_ports();
        assert!(result.is_ok());
    }

    #[test]
    fn ping_host_with_empty_host_is_an_error() {
        assert!(ping_host("").is_err());
    }
}
