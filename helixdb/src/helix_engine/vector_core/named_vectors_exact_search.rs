use super::*;

/// Unix seconds of the last warning about a failed broad prefilter scan.
static BROAD_SCAN_FAILURE_WARNED_AT: AtomicU64 = AtomicU64::new(0);

/// The same for a failed sidecar scan of a set of ids.
static EXACT_SCAN_FAILURE_WARNED_AT: AtomicU64 = AtomicU64::new(0);

/// True when the last warning is a minute old; the caller that gets `true`
/// has taken this minute's warning.
fn broad_scan_warning_due(warned_at: &AtomicU64, now_secs: u64) -> bool {
    let last = warned_at.load(AtomicOrdering::Relaxed);
    now_secs.saturating_sub(last) >= 60
        && warned_at
            .compare_exchange(
                last,
                now_secs,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
}

/// A set of this many ids or fewer is never handed to the sidecar scan,
/// whatever the space holds: its ids are scored where their rows are, from
/// one batched read of the rows a segment
/// (`VectorCore::exact_hits_from_rows`) or, where that declines, by a read
/// of each id. The client's own id lists are sets of that size (its
/// gate's, 500 ids at most as it is set, and its symbol lookup's 512).
const EXACT_SCAN_MIN_IDS: usize = 512;

#[cfg(test)]
thread_local! {
    static EXACT_SCAN_ROWS_PER_ID_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Test-only: run `f` with `exact_scan_rows_per_id` forced on the calling
/// thread.
#[cfg(test)]
pub(crate) fn with_exact_scan_rows_per_id_for_test<R>(rows: usize, f: impl FnOnce() -> R) -> R {
    let previous = EXACT_SCAN_ROWS_PER_ID_OVERRIDE.with(|cell| cell.replace(Some(rows)));
    let out = f();
    EXACT_SCAN_ROWS_PER_ID_OVERRIDE.with(|cell| cell.set(previous));
    out
}

#[cfg(test)]
thread_local! {
    static CLOSED_BEFORE_EXACT_SCAN: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Test-only: the next set the rule hands to a scan on the calling thread
/// finds `segment`'s sidecar file closed, as when the file goes between the
/// rule's look at the segments and the scan.
#[cfg(test)]
pub(crate) fn close_before_exact_scan_for_test(segment: &str) {
    CLOSED_BEFORE_EXACT_SCAN.with(|cell| cell.replace(Some(segment.to_string())));
}

/// `HELIX_DENSE_EXACT_SCAN_ROWS_PER_ID` as set: 64 unset or unreadable.
fn exact_scan_rows_per_id_of(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(64)
}

/// How many sidecar rows a scan may walk for each id of a set before reading
/// the ids one by one is the cheaper way to score it
/// (`HELIX_DENSE_EXACT_SCAN_ROWS_PER_ID`; 0: a set is never scanned). The
/// figures are of before the ids' rows were read in one batch a segment
/// (`VectorCore::exact_hits_from_rows`): a read of one id was two point
/// reads, 102 to 128 µs warm on a reader; a scan walks every ordinal row of
/// the space, 1.27 µs each, so the two cost the same at 77 to 96 rows an id.
/// 64 is the planner's own ratio of rows scanned to a point read
/// (`REST_ROWS_PER_NODE_READ`), and left the scan 15 to 32% the cheaper at
/// the bound. The table that settles the 64 is measured with the batch in.
fn exact_scan_rows_per_id() -> usize {
    #[cfg(test)]
    if let Some(forced) = EXACT_SCAN_ROWS_PER_ID_OVERRIDE.with(|cell| cell.get()) {
        return forced;
    }
    static ROWS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        exact_scan_rows_per_id_of(
            std::env::var("HELIX_DENSE_EXACT_SCAN_ROWS_PER_ID")
                .ok()
                .as_deref(),
        )
    });
    *ROWS
}

/// Whether a sidecar scan of `rows` rows is the cheaper way to score a set
/// of `ids` ids.
fn scan_is_cheaper(ids: usize, rows: u64, rows_per_id: usize) -> bool {
    ids > EXACT_SCAN_MIN_IDS
        && rows_per_id > 0
        && rows <= (ids as u64).saturating_mul(rows_per_id as u64)
}

impl NamedVectorManager {
    /// Why a set of `ids` ids is scored by a read of each id, or `None`
    /// where one sidecar scan of `segments` is the cheaper way to the same
    /// rows. The reasons are the `way` labels of
    /// `helix_dense_exact_scorer_total`.
    ///
    /// A scan stands in only on a reader replica's read that is pinned to a
    /// snapshot. A writer walks its cached ordinal map, which an insert
    /// fills before its batch commits, so a scan there can score a row a
    /// read of that id does not find yet. A replica's read that is not
    /// pinned (`HELIX_LSM_READER_SNAPSHOT_READS` off, or a snapshot that
    /// did not open) may see a manifest refresh between the walk of the
    /// ordinal rows and the row reads that follow it.
    fn read_per_id_because(
        &self,
        r: &AnyRead<'_>,
        segments: &[(String, DenseVectorSegmentRole)],
        ids: usize,
    ) -> Result<Option<&'static str>, VectorError> {
        if ids <= EXACT_SCAN_MIN_IDS {
            return Ok(Some("per_id_small"));
        }
        let rows_per_id = exact_scan_rows_per_id();
        if rows_per_id == 0 {
            return Ok(Some("per_id_off"));
        }
        if !matches!(r, AnyRead::LsmReader(Some(_))) {
            return Ok(Some("per_id_handle"));
        }
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut rows = 0u64;
        for (segment, _) in segments {
            // A segment with no core open is passed over by both ways.
            let Some(core) = cores.get(segment) else {
                continue;
            };
            let Some(slots) = core.exact_scan_slots(r)? else {
                return Ok(Some("per_id_segment"));
            };
            rows = rows.saturating_add(slots);
        }
        Ok((!scan_is_cheaper(ids, rows, rows_per_id)).then_some("per_id_rows"))
    }

    /// The nearest `k` of `candidate_ids` in dense space `name`, each id
    /// scored where its newest copy is.
    ///
    /// The ids are scored by the cheaper of two ways that return the same
    /// rows in the same order: id by id, segment by segment, or one sidecar
    /// scan (`broad_prefilter_scan_of`) when the set is large for the rows it
    /// lies in (`read_per_id_because`). Id by id, a segment's rows are read
    /// in one batch and scored from the read
    /// (`VectorCore::exact_hits_from_rows`); a segment that declines it has
    /// each id read, as every segment had. A scan that declines or fails is
    /// followed by the ids one by one; one whose reads were cancelled with
    /// the request is not. `helix_dense_exact_scorer_total` counts which way
    /// each set went.
    pub fn dense_search_candidate_ids_exact_be<I>(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        query: &[f32],
        k: usize,
        candidate_ids: I,
        selectivity_hint: Option<f32>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        I: IntoIterator<Item = u128>,
    {
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut candidate_ids = unique_candidate_ids(candidate_ids.into_iter().collect());

        let (newer, segment_infos, retired) = self.dense_segment_infos_at(r, name)?;
        let r = newer.as_ref().unwrap_or(r);
        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }
        let segment_infos = self.searchable_dense_segments_be(r, name, segment_infos)?;

        let way = match self.read_per_id_because(r, &segment_infos, candidate_ids.len())? {
            Some(why) => why,
            None => {
                #[cfg(test)]
                if let Some(segment) = CLOSED_BEFORE_EXACT_SCAN.with(|cell| cell.take()) {
                    self.cores.read().unwrap()[&segment].close_mmap_sidecar_for_test();
                }
                let set: HashSet<u128> = candidate_ids.iter().copied().collect();
                match self.broad_prefilter_scan_of::<fn(u128) -> bool>(
                    r,
                    name,
                    &segment_infos,
                    &retired,
                    query,
                    k,
                    &set,
                    None,
                    selectivity_hint,
                ) {
                    Ok(Some(found)) => {
                        metrics::counter!("helix_dense_exact_scorer_total", "way" => "scan")
                            .increment(1);
                        return Ok(found);
                    }
                    Ok(None) => "scan_declined",
                    Err(error) if error.to_string().contains("LSM read request cancelled") => {
                        metrics::counter!(
                            "helix_dense_exact_scorer_total",
                            "way" => "scan_cancelled"
                        )
                        .increment(1);
                        return Err(error);
                    }
                    Err(error) => {
                        let now_secs = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_or(0, |since| since.as_secs());
                        if broad_scan_warning_due(&EXACT_SCAN_FAILURE_WARNED_AT, now_secs) {
                            tracing::warn!(
                                collection = %self.collection_name,
                                vector = %name,
                                %error,
                                "sidecar scan of a set of ids failed; the ids are read one by one"
                            );
                        }
                        "scan_failed"
                    }
                }
            }
        };
        metrics::counter!("helix_dense_exact_scorer_total", "way" => way).increment(1);

        let recency_ranks = segment_recency_ranks(name, &segment_infos);
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut combined: Vec<(u32, HVector)> = Vec::with_capacity(k * segment_infos.len());
        metrics::histogram!(
            "helix_named_vector_exact_search_items",
            "collection" => self.collection_name.clone(),
            "vector" => name.to_string(),
            "measurement" => "candidate_ids",
            "segment_index" => "all",
            "segment_count" => segment_infos.len().to_string(),
        )
        .record(candidate_ids.len() as f64);
        metrics::histogram!(
            "helix_named_vector_exact_search_items",
            "collection" => self.collection_name.clone(),
            "vector" => name.to_string(),
            "measurement" => "segment_count",
            "segment_index" => "all",
            "segment_count" => segment_infos.len().to_string(),
        )
        .record(segment_infos.len() as f64);

        for (segment_index, (segment_name, _)) in dense_segments_by_recency(name, &segment_infos)
            .into_iter()
            .enumerate()
        {
            if candidate_ids.is_empty() {
                break;
            }
            let Some(core) = cores.get(segment_name) else {
                self.note_unreachable_segment(name, segment_name, &retired, "exact_id_search");
                continue;
            };
            let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
            let candidate_start = Instant::now();
            // One batched read of the rows says which ids this segment holds
            // and scores them; where it declines, each id is read as before.
            let from_rows =
                core.exact_hits_from_rows(r, query, &candidate_ids, k, selectivity_hint)?;
            let (segment_candidate_ids, row_hits, filter_time) = match from_rows {
                Some((present, hits, read)) => {
                    let score_time = candidate_start.elapsed().saturating_sub(read);
                    // Every current copy, whether or not it is among the
                    // segment's nearest (`take_candidate_ids_present_in_segment`).
                    let held: HashSet<u128> = present.iter().copied().collect();
                    candidate_ids.retain(|id| !held.contains(id));
                    (present, Some((hits, score_time)), read)
                }
                None => {
                    let routed =
                        take_candidate_ids_present_in_segment(core, r, &mut candidate_ids)?;
                    (routed, None, candidate_start.elapsed())
                }
            };
            metrics::histogram!(
                "helix_named_vector_exact_search_stage_ms",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "stage" => "segment_candidate_filter",
                "segment_index" => segment_index.to_string(),
                "segment_count" => segment_infos.len().to_string(),
            )
            .record(filter_time.as_secs_f64() * 1000.0);
            metrics::histogram!(
                "helix_named_vector_exact_search_items",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "measurement" => "segment_candidate_ids",
                "segment_index" => segment_index.to_string(),
                "segment_count" => segment_infos.len().to_string(),
            )
            .record(segment_candidate_ids.len() as f64);
            if segment_candidate_ids.is_empty() {
                continue;
            }
            let (hits, score_time) = match row_hits {
                Some(scored) => scored,
                None => {
                    let search_start = Instant::now();
                    let hits = core.search_exact_ids_with_selectivity(
                        r,
                        query,
                        segment_candidate_ids.iter().copied(),
                        k,
                        selectivity_hint,
                    )?;
                    (hits, search_start.elapsed())
                }
            };
            metrics::histogram!(
                "helix_named_vector_exact_search_stage_ms",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "stage" => "segment_exact_score",
                "segment_index" => segment_index.to_string(),
                "segment_count" => segment_infos.len().to_string(),
            )
            .record(score_time.as_secs_f64() * 1000.0);
            metrics::histogram!(
                "helix_named_vector_exact_search_items",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "measurement" => "segment_hits",
                "segment_index" => segment_index.to_string(),
                "segment_count" => segment_infos.len().to_string(),
            )
            .record(hits.len() as f64);
            combined.extend(hits.into_iter().map(|hit| (rank, hit)));
        }

        let dedup_start = Instant::now();
        let results = dedup_keep_newest(combined, k);
        metrics::histogram!(
            "helix_named_vector_exact_search_stage_ms",
            "collection" => self.collection_name.clone(),
            "vector" => name.to_string(),
            "stage" => "dedup_keep_newest",
            "segment_index" => "all",
            "segment_count" => segment_infos.len().to_string(),
        )
        .record(dedup_start.elapsed().as_secs_f64() * 1000.0);
        metrics::histogram!(
            "helix_named_vector_exact_search_items",
            "collection" => self.collection_name.clone(),
            "vector" => name.to_string(),
            "measurement" => "results_returned",
            "segment_index" => "all",
            "segment_count" => segment_infos.len().to_string(),
        )
        .record(results.len() as f64);
        Ok(results)
    }

    /// Broad-filter optimized dense search: scores prefilter members via the
    /// mmap sidecar (sequential ordinal scan + cheap `HashSet` check), then
    /// applies the expensive `recheck` best-first until `k` members pass.
    ///
    /// Returns the same result shape as `dense_search_with_id_filter_ef_be`;
    /// the caller must fall back to that method when this returns `Err` or
    /// when the inner per-segment scan returns `None` (no sidecar available).
    ///
    /// Because every caller falls back, a scan that fails costs its request
    /// nothing but time and would show nowhere: `helix_broad_scan_total`
    /// counts how each scan ended (`served`, `declined` for a segment without
    /// a sidecar, `cancelled` when the caller left during it, `failed`), and
    /// a failure is logged at most once a minute.
    pub fn dense_search_broad_prefilter_scan_be<F>(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        query: &[f32],
        k: usize,
        prefilter: &HashSet<u128>,
        recheck: Option<&[F]>,
        selectivity_hint: Option<f32>,
    ) -> Result<Option<Vec<HVector>>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let scan =
            self.broad_prefilter_scan_be(r, name, query, k, prefilter, recheck, selectivity_hint);
        let outcome = match &scan {
            Ok(Some(_)) => "served",
            Ok(None) => "declined",
            // Its reads were cancelled with the request: nothing is wrong
            // with the scan, and nobody is left to fall back for.
            Err(error) if error.to_string().contains("LSM read request cancelled") => "cancelled",
            Err(_) => "failed",
        };
        metrics::counter!("helix_broad_scan_total", "outcome" => outcome).increment(1);
        if let (Err(error), "failed") = (&scan, outcome) {
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| since.as_secs());
            if broad_scan_warning_due(&BROAD_SCAN_FAILURE_WARNED_AT, now_secs) {
                tracing::warn!(
                    collection = %self.collection_name,
                    vector = %name,
                    %error,
                    "broad prefilter scan failed; the search falls back to the graph walk"
                );
            }
        }
        scan
    }

    fn broad_prefilter_scan_be<F>(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        query: &[f32],
        k: usize,
        prefilter: &HashSet<u128>,
        recheck: Option<&[F]>,
        selectivity_hint: Option<f32>,
    ) -> Result<Option<Vec<HVector>>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        if k == 0 || prefilter.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let (newer_read, segment_infos, retired) = self.dense_segment_infos_at(r, name)?;
        let r = newer_read.as_ref().unwrap_or(r);
        let segment_infos = self.searchable_dense_segments_be(r, name, segment_infos)?;
        self.broad_prefilter_scan_of(
            r,
            name,
            &segment_infos,
            &retired,
            query,
            k,
            prefilter,
            recheck,
            selectivity_hint,
        )
    }

    /// The scan over `segment_infos`, the segments a read at `r` searches.
    #[allow(clippy::too_many_arguments)]
    fn broad_prefilter_scan_of<F>(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        segment_infos: &[(String, DenseVectorSegmentRole)],
        retired: &[String],
        query: &[f32],
        k: usize,
        prefilter: &HashSet<u128>,
        recheck: Option<&[F]>,
        selectivity_hint: Option<f32>,
    ) -> Result<Option<Vec<HVector>>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        // Newest first, the order the read per id asks them in
        // (`dense_search_candidate_ids_exact_be`): a segment's place in it is
        // its recency rank, and each segment is told what the newer ones
        // hold. A segment cuts its members to its nearest `k`, so an id
        // stored twice must be its newest holder's alone before that cut:
        // keeping the newest of the copies that came back afterwards returned
        // the older copy whenever the newer fell outside its segment's cut.
        let by_recency = dense_segments_by_recency(name, segment_infos);
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let mut combined: Vec<(u32, HVector)> = Vec::with_capacity(k * by_recency.len());
        let mut newer = NewerCopies::default();

        for (rank, (segment_name, _role)) in by_recency.iter().enumerate() {
            let Some(core) = cores.get(segment_name) else {
                self.note_unreachable_segment(name, segment_name, retired, "broad_prefilter_scan");
                continue;
            };
            let hits = core.search_broad_prefilter_scan_after(
                r,
                query,
                k,
                prefilter,
                recheck,
                selectivity_hint,
                &mut newer,
            )?;
            match hits {
                Some(results) => {
                    combined.extend(results.into_iter().map(|hit| (rank as u32, hit)));
                }
                None => {
                    // At least one segment lacks a sidecar; caller must fall
                    // back to the regular HNSW path for correctness.
                    return Ok(None);
                }
            }
        }
        // A hit scored from a listing an older segment's read then found to
        // have no row is no row of the space; the older copy answers.
        if newer.has_void() {
            combined.retain(|(rank, hit)| !newer.is_void(&by_recency[*rank as usize].0, hit.id));
        }

        Ok(Some(dedup_keep_newest(combined, k)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_broad_scan_is_warned_about_once_a_minute() {
        let warned_at = AtomicU64::new(0);
        assert!(broad_scan_warning_due(&warned_at, 1_000));
        assert!(!broad_scan_warning_due(&warned_at, 1_059));
        assert!(broad_scan_warning_due(&warned_at, 1_060));
        assert!(!broad_scan_warning_due(&warned_at, 1_060));
    }

    #[test]
    fn a_set_is_scanned_above_512_ids_and_within_its_rows_an_id() {
        // The client's own id lists, whatever the space holds.
        for ids in [1, 200, 512] {
            assert!(!scan_is_cheaper(ids, 0, 64), "{ids} ids");
            assert!(!scan_is_cheaper(ids, 1, usize::MAX), "{ids} ids");
        }
        // 64 rows an id: 513 ids in 32,832 rows, and not in one row more.
        assert!(scan_is_cheaper(513, 32_832, 64));
        assert!(!scan_is_cheaper(513, 32_833, 64));
        // The collection the figures were measured on, 99,927 rows.
        assert!(!scan_is_cheaper(1_561, 99_927, 64));
        assert!(scan_is_cheaper(1_562, 99_927, 64));
        // The setting off its default: never, one row an id, no bound.
        assert!(!scan_is_cheaper(100_000, 0, 0));
        assert!(scan_is_cheaper(513, 513, 1));
        assert!(!scan_is_cheaper(513, 514, 1));
        assert!(scan_is_cheaper(513, u64::MAX, usize::MAX));
    }

    #[test]
    fn the_rows_an_id_setting_is_read_as_a_count() {
        assert_eq!(exact_scan_rows_per_id_of(None), 64);
        assert_eq!(exact_scan_rows_per_id_of(Some("0")), 0);
        assert_eq!(exact_scan_rows_per_id_of(Some(" 1 ")), 1);
        assert_eq!(exact_scan_rows_per_id_of(Some("100000")), 100_000);
        for unreadable in ["", "many", "-1", "6.4"] {
            assert_eq!(
                exact_scan_rows_per_id_of(Some(unreadable)),
                64,
                "{unreadable:?}"
            );
        }
    }
}
