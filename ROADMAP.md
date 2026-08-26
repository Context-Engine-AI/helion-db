# Helion Roadmap

Helion's roadmap focuses on a reliable, self-hosted graph and vector database
with predictable resource use and stable compatibility surfaces.

## Current priorities

### Storage reliability

- Strengthen crash recovery, snapshot validation, and WAL lifecycle tests.
- Keep LMDB map growth and collection-cache behavior bounded under churn.
- Complete LSM catalog recovery from object storage without local state.
- Make writer fencing and reader freshness observable and testable.

### Vector search

- Improve indexed-segment convergence and background merge scheduling.
- Expand payload-index coverage for filtered dense and sparse search.
- Continue quantized sidecar validation without changing default accuracy.
- Add reproducible public benchmarks with documented datasets and hardware.

### Graph operations

- Expand traversal and graph-algorithm correctness coverage.
- Improve large-subgraph pagination and cancellation behavior.
- Preserve stable native graph APIs while internal storage evolves.

### Distribution and operations

- Publish reproducible source tags and matching CLI/container artifacts.
- Maintain Docker Compose and Minikube examples for both storage backends.
- Document backup, restore, monitoring, and multi-node deployment boundaries.
- Keep configuration examples free of provider accounts, credentials, and
  organization-specific infrastructure.

### Developer experience

- Improve HelixQL diagnostics and local project workflows.
- Keep the Qdrant-compatible API documented and regression-tested.
- Add contribution, security, and release documentation as the community grows.

## Compatibility policy

The public product name is Helion. Existing `helix`, `helixdb`, and
`.helix/repo/helix-db` identifiers remain until a separately documented
migration can preserve existing projects and storage layouts.

## Licensing boundary

Helion is developed from the GPLv3 source lineage documented in
[PROVENANCE.md](PROVENANCE.md). AGPL versions of upstream HelixDB are not source
donors. New contributions to this repository are accepted under
`GPL-3.0-only`.
