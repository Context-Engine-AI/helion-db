use crate::helix_engine::types::GraphError;
use heed3::{types::Bytes, Database, RwTxn};

pub(crate) struct SecondaryIndexSnapshot {
    pub(crate) db: Database<Bytes, Bytes>,
    pub(crate) key: Vec<u8>,
    previous: Option<Vec<u8>>,
}

pub(crate) fn snapshot_secondary_key(
    txn: &mut RwTxn<'_>,
    db: Database<Bytes, Bytes>,
    key: Vec<u8>,
) -> Result<SecondaryIndexSnapshot, GraphError> {
    let previous = db.get(txn, key.as_slice())?.map(|bytes| bytes.to_vec());
    Ok(SecondaryIndexSnapshot { db, key, previous })
}

pub(crate) fn restore_secondary_snapshots(
    txn: &mut RwTxn<'_>,
    snapshots: &[SecondaryIndexSnapshot],
) -> Result<(), GraphError> {
    for snapshot in snapshots.iter().rev() {
        match &snapshot.previous {
            Some(previous) => snapshot
                .db
                .put(txn, snapshot.key.as_slice(), previous.as_slice())?,
            None => {
                snapshot.db.delete(txn, snapshot.key.as_slice())?;
            }
        }
    }
    Ok(())
}

pub(crate) fn secondary_key_points_to_node(
    txn: &mut RwTxn<'_>,
    db: Database<Bytes, Bytes>,
    key: &[u8],
    node_id: u128,
) -> Result<bool, GraphError> {
    let id_bytes = node_id.to_be_bytes();
    Ok(db
        .get(txn, key)?
        .is_some_and(|existing| existing == id_bytes.as_slice()))
}
