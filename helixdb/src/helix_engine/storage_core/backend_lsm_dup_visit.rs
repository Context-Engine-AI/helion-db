use super::*;

impl LsmBackend {
    pub(super) fn for_each_dup_rows(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        key: &[u8],
        visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.for_each_dup_rows_from(txn, ns, key, &[], visit)
    }

    /// [`Self::for_each_dup_rows`] from the first value that is not below
    /// `from`. The scan starts there, so no row before it is read.
    pub(crate) fn for_each_dup_rows_from(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        key: &[u8],
        from: &[u8],
        mut visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        #[cfg(test)]
        let mut visit = |value: &[u8]| {
            DUP_SCAN_ROWS_FOR_TEST.with(|count| count.set(count.get() + 1));
            visit(value)
        };
        let kp = dup_key_prefix(ns, key);
        if txn.pending.is_empty() {
            block_on_lsm_read_mapped(
                &self.rt,
                async {
                    let mut iter = txn
                        .snapshot
                        .scan_prefix_with_options(kp.clone(), from.., &lsm_scan_options())
                        .await
                        .map_err(io)?;
                    while let Some(kv) = iter.next().await.map_err(io)? {
                        let k = kv.key.as_ref();
                        let value: &[u8] = if k.len() >= kp.len() {
                            &k[kp.len()..]
                        } else {
                            &[]
                        };
                        if !visit(value) {
                            break;
                        }
                    }
                    Ok::<(), BackendError>(())
                },
                |e| e,
            )?;
            return Ok(());
        }

        let mut pending_rows: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for (full_key, pending) in &txn.pending {
            if pending.is_none() {
                continue;
            }
            if !full_key.starts_with(&kp) {
                continue;
            }
            let value = &full_key[kp.len()..];
            if value < from {
                continue;
            }
            pending_rows.insert(full_key.clone(), value.to_vec());
        }

        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let mut pending_iter = pending_rows.into_iter().peekable();
                let mut iter = txn
                    .snapshot
                    .scan_prefix_with_options(kp.clone(), from.., &lsm_scan_options())
                    .await
                    .map_err(io)?;
                while let Some(kv) = iter.next().await.map_err(io)? {
                    let k = kv.key.as_ref();
                    while let Some((pending_key, pending_value)) = pending_iter.peek() {
                        if pending_key.as_slice() >= k {
                            break;
                        }
                        if !visit(pending_value) {
                            return Ok(());
                        }
                        pending_iter.next();
                    }
                    if txn.pending.contains_key(k) {
                        continue;
                    }
                    let value: &[u8] = if k.len() >= kp.len() {
                        &k[kp.len()..]
                    } else {
                        &[]
                    };
                    if !visit(value) {
                        return Ok(());
                    }
                }
                for (_full_key, value) in pending_iter {
                    if !visit(&value) {
                        break;
                    }
                }
                Ok::<(), BackendError>(())
            },
            |e| e,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS: Namespace<'static> = Namespace::OutEdges;

    /// The values under `n1` from `from` on, and the rows read for them.
    fn read_from(be: &LsmBackend, r: &LsmRead, from: &[u8]) -> (Vec<Vec<u8>>, usize) {
        let before = dup_scan_rows_for_test();
        let mut values = Vec::new();
        be.for_each_dup_rows_from(r, NS, b"n1", from, |value| {
            values.push(value.to_vec());
            true
        })
        .unwrap();
        (values, dup_scan_rows_for_test() - before)
    }

    fn values(listed: &[&[u8]]) -> Vec<Vec<u8>> {
        listed.iter().map(|value| value.to_vec()).collect()
    }

    #[test]
    fn a_dup_read_from_a_value_reads_no_row_before_it() {
        let be = LsmBackend::open_in_memory("/dup-from-test").unwrap();
        let mut w = be.begin_write().unwrap();
        for value in [b"a", b"c", b"e", b"g"] {
            be.put_dup(&mut w, NS, b"n1", value).unwrap();
        }
        // Keys on either side of `n1`, and one it starts with.
        be.put_dup(&mut w, NS, b"n", b"z").unwrap();
        be.put_dup(&mut w, NS, b"n0", b"z").unwrap();
        be.put_dup(&mut w, NS, b"n2", b"a").unwrap();
        be.commit(w).unwrap();

        // From a value that is there, from between two, from before the
        // first and from past the last.
        let r = be.begin_read().unwrap();
        assert_eq!(read_from(&be, &r, b"c"), (values(&[b"c", b"e", b"g"]), 3));
        assert_eq!(read_from(&be, &r, b"d"), (values(&[b"e", b"g"]), 2));
        assert_eq!(
            read_from(&be, &r, b""),
            (values(&[b"a", b"c", b"e", b"g"]), 4)
        );
        assert_eq!(read_from(&be, &r, b"h"), (values(&[]), 0));
        // It ends when it is told to.
        let mut first = Vec::new();
        be.for_each_dup_rows_from(&r, NS, b"n1", b"b", |value| {
            first.push(value.to_vec());
            false
        })
        .unwrap();
        assert_eq!(first, values(&[b"c"]));

        // A batch's own rows are read with the committed ones, and none of
        // them before `from` either.
        let mut w = be.begin_write().unwrap();
        for value in [b"b", b"f", b"h"] {
            be.put_dup(&mut w, NS, b"n1", value).unwrap();
        }
        be.delete_dup(&mut w, NS, b"n1", b"e").unwrap();
        let overlay = be.begin_read_with_pending(w.pending().clone()).unwrap();
        assert_eq!(
            read_from(&be, &overlay, b"c").0,
            values(&[b"c", b"f", b"g", b"h"])
        );
        assert_eq!(
            read_from(&be, &overlay, b"").0,
            values(&[b"a", b"b", b"c", b"f", b"g", b"h"])
        );
        assert_eq!(read_from(&be, &overlay, b"g").0, values(&[b"g", b"h"]));
    }
}
