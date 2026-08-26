use std::sync::OnceLock;

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "full" | "debug" | "all"
            )
        })
        .unwrap_or(false)
}

/// Opt-in gate for high-cardinality or per-request diagnostics.
///
/// Core health and backpressure metrics stay enabled. This gate is for
/// expensive debugging families emitted from search, scan, upsert, and LMDB
/// write hot paths.
pub fn hot_path_metrics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_truthy("HELIX_HOT_PATH_METRICS")
            || std::env::var("HELIX_METRICS_LEVEL")
                .ok()
                .map(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "full" | "debug" | "all"
                    )
                })
                .unwrap_or(false)
    })
}
