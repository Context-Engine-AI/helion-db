//! Lightweight lock/permit instrumentation for production wedge diagnosis.
//!
//! Each `log` call emits a single `tracing::info!` line under the
//! `helix_diag` target with consistent fields:
//!   tid   – thread id (so we can correlate across log lines from one thread)
//!   op    – wait | got | release | enter | exit | reject | timeout
//!   site  – name of the lock/permit/scope (e.g. `write_txn_gate`,
//!           `WRITE_BLOCKING`, `with_write_txn`, `worker_loop.dequeue`)
//!   ctx   – optional free-form context (collection name, request path, etc.)
//!
//! Disabled by default. Set `HELIX_DIAG_LOG=1` to re-enable for a debugging
//! window — the call sites stay wired so you only flip the env var. Leaving
//! this on emits 5+ INFO lines per gateway request, which costs measurable
//! latency on the read path under load (see 2026-05-05 incident: lmdb-fair
//! build's per-admission diag::log was the dominant cost on read p99).
//!
//! Filtering when enabled: `RUST_LOG=helix_diag=info,info` or grep
//! `helix_diag` in the plain log.

use std::sync::OnceLock;

fn diag_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HELIX_DIAG_LOG")
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

#[inline]
pub fn log(op: &str, site: &str, ctx: &str) {
    if !diag_enabled() {
        return;
    }
    tracing::info!(
        target: "helix_diag",
        tid = ?std::thread::current().id(),
        op = op,
        site = site,
        ctx = ctx,
        "helix_diag"
    );
}

/// Convenience: log on entry, drop a guard that logs on exit.
pub struct Span {
    site: &'static str,
    ctx: String,
}

impl Span {
    #[inline]
    pub fn enter(site: &'static str, ctx: impl Into<String>) -> Self {
        let ctx = ctx.into();
        log("enter", site, &ctx);
        Self { site, ctx }
    }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        log("exit", self.site, &self.ctx);
    }
}
