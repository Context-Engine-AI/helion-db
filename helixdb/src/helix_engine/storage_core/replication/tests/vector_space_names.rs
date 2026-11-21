//! A dense vector space's name is the stem of its segments' sidecar files
//! under the collection's directory, joined as it comes. So it has to be one
//! path segment, as a collection's name has to: `../x` names files beside
//! the directory, and an absolute name files anywhere. And it has to be
//! without a dot: a sidecar's file is its stem `with_extension`, which cuts
//! the stem at its last dot.

use super::*;
use crate::helix_gateway::async_gateway::graph_error_status;

/// Names that cannot be a sidecar's stem.
const UNUSABLE: [&str; 7] = ["../escaped", "/tmp/absolute", "a/b", "..", ".", "", "v1.5"];
/// Names that can.
const USABLE: [&str; 3] = ["dense", "bge-base-en-v1_5", "_edge"];

/// Why a dot is refused: every segment of such a space gets one file name.
#[test]
fn the_sidecars_of_two_segments_of_a_dotted_name_are_one_file() {
    let sidecar = |segment: &str| std::path::Path::new(segment).with_extension("hvec");
    assert_eq!(sidecar("v1.5__seg_000001"), sidecar("v1.5__seg_000002"));
    assert_ne!(sidecar("v1_5__seg_000001"), sidecar("v1_5__seg_000002"));
}

fn dense(name: &str) -> HashMap<String, NamedVectorConfig> {
    HashMap::from([(
        name.to_string(),
        NamedVectorConfig {
            size: 4,
            distance: DistanceMetric::Cosine,
            spindle: Default::default(),
        },
    )])
}

fn create(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    vectors: HashMap<String, NamedVectorConfig>,
) -> Result<(), GraphError> {
    apply_mutation(
        collections,
        config,
        &ReplicatedMutation::CreateCollection {
            name: name.into(),
            vectors,
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
}

fn update(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    vectors: HashMap<String, NamedVectorConfig>,
) -> Result<(), GraphError> {
    apply_mutation(
        collections,
        config,
        &ReplicatedMutation::UpdateCollection {
            name: name.into(),
            vectors,
            sparse_vectors: HashMap::new(),
        },
    )
}

/// The names in `dir`: what the manager keeps there is one directory a
/// collection, so a sidecar stem that left a collection's directory shows.
fn entries(dir: &std::path::Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    found
}

/// What a refused name is answered with: the caller's mistake, as a
/// collection's name that is not one path segment is.
fn assert_refused(result: Result<(), GraphError>, vector: &str, case: &str) {
    let error = result.expect_err(&format!("{case}: vector name {vector:?} was taken"));
    assert_eq!(
        graph_error_status(&error),
        400,
        "{case} {vector:?}: {error}"
    );
    assert!(
        error.to_string().contains(&format!("{vector:?}")),
        "{error}"
    );
}

fn refuses_and_takes(config: Config, backend: &str) {
    let temp_dir = TempDir::new().unwrap();
    let data = temp_dir.path().join("data");
    let collections = Arc::new(CollectionManager::new(data.clone(), config.clone()).unwrap());

    // A create is refused before it makes anything: no collection, no file.
    let kept = data.join("collections");
    let before = entries(&kept);
    for (index, vector) in UNUSABLE.iter().enumerate() {
        let name = format!("refused_{index}");
        let case = format!("{backend} create");
        assert_refused(
            create(&collections, &config, &name, dense(vector)),
            vector,
            &case,
        );
        assert!(
            !collections.collection_exists(&name).unwrap(),
            "{case} {vector:?}"
        );
    }
    assert_eq!(
        entries(&kept),
        before,
        "{backend}: a refused create left something"
    );

    // One unusable name refuses the whole request, the usable one with it.
    let mut mixed = dense("dense");
    mixed.extend(dense("../escaped"));
    assert_refused(
        create(&collections, &config, "refused_mixed", mixed),
        "../escaped",
        backend,
    );
    assert!(!collections.collection_exists("refused_mixed").unwrap());

    create(&collections, &config, "repo", HashMap::new()).unwrap();
    let before = entries(&kept);
    for vector in UNUSABLE {
        let case = format!("{backend} update");
        assert_refused(
            update(&collections, &config, "repo", dense(vector)),
            vector,
            &case,
        );
    }
    assert_eq!(
        entries(&kept),
        before,
        "{backend}: a refused update left something"
    );

    for (index, vector) in USABLE.iter().enumerate() {
        create(
            &collections,
            &config,
            &format!("taken_{index}"),
            dense(vector),
        )
        .unwrap_or_else(|error| panic!("{backend} create {vector:?}: {error}"));
        update(&collections, &config, "repo", dense(vector))
            .unwrap_or_else(|error| panic!("{backend} update {vector:?}: {error}"));
    }
}

#[test]
fn a_vector_name_that_is_not_one_path_segment_names_no_vector_space_lsm() {
    refuses_and_takes(Config::new(8, 32, 64, 1).with_lsm_in_memory(), "lsm");
}

#[test]
fn a_vector_name_that_is_not_one_path_segment_names_no_vector_space_lmdb() {
    refuses_and_takes(Config::new(8, 32, 64, 1), "lmdb");
}
