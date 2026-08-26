# Helion

Helion is an open-source graph and vector database written in Rust. It combines
graph traversal, dense and sparse vector search, Qdrant-compatible HTTP APIs,
and disk-backed storage in one self-hosted engine.

Helion is maintained by Context Engine AI and distributed under
`GPL-3.0-only`. Compatibility-sensitive crate names and runtime paths still use
`helix` and `helixdb`.

## Why we built Helion

Context Engine started with two databases: Qdrant for vector search and
Memgraph for graph relationships. That combination helped us get the product
off the ground, but it also meant operating two storage systems, two scaling
models, two backup paths, and an application layer responsible for joining the
answers.

By April 2026, our startup runway was shrinking. We could no longer afford the
financial and operational cost of running both databases, yet graph and vector
retrieval were both fundamental to the product. We needed one engine that we
could understand, operate, and improve ourselves.

HelixDB already brought graph and vector primitives together in Rust. We began
from its last unambiguous GPLv3 source point, commit
[`71ebdb9`](https://github.com/HelixDB/helix-db/commit/71ebdb9896ff1a2448eb55ad26014ec8428c8086),
and built an independent line around the constraints we actually had: local
disk, object storage, bounded memory, compatibility with existing clients, and
the ability to debug every layer when production went wrong. We did not copy
source from the later AGPL line; the detailed boundary is recorded in
[PROVENANCE.md](PROVENANCE.md).

HelixDB has since moved its current project to
[Apache-2.0](https://github.com/HelixDB/helix-db/commit/461a0de42348f9806f52b4274150961885ff5145).
We appreciate that decision and the upstream team's renewed investment in
permissive open source. Helion remains a separate GPLv3 project because that is
the lineage from which our work began and the license under which we are
releasing our combined work.

Helion is what came out of that moment: not a theoretical database rewrite,
but a startup learning to own its infrastructure because it had to.

## Features

- Dense HNSW and sparse-vector search with payload filtering
- Native graph storage and traversal APIs
- Qdrant-compatible collection, point, search, scroll, facet, and snapshot routes
- LMDB storage with WAL, snapshots, and automatic map growth
- Optional SlateDB-backed LSM storage on S3-compatible object storage
- Single-writer fencing and read-replica support for the LSM backend
- HelixQL compiler and local CLI workflow
- Raft-capable replication primitives

## Quick start

### Docker

```bash
docker build -t helion-db .
docker run --rm \
  -p 6969:6969 \
  -v helion-data:/data \
  helion-db
```

The service listens on `http://localhost:6969`. Check it with:

```bash
curl http://localhost:6969/health
```

### Build from source

Requirements:

- Rust 1.94 or later
- `clang`, `pkg-config`, OpenSSL development headers, and Protocol Buffers

```bash
cargo build --release -p helix-container
HELIX_DATA_DIR=/tmp/helion-data \
  ./target/release/helix-container
```

### CLI

```bash
cargo install --path helix-cli
helix init --path ./example-project
helix check --path ./example-project
helix deploy --path ./example-project --local
```

## Storage backends

Helion supports two storage modes selected through `HELIX_STORAGE_BACKEND`:

- `lmdb`: local disk-backed storage and the default for a single-node deployment
- `lsm`: SlateDB-backed storage using S3 or an S3-compatible object store

Local LSM examples are available under `deploy/lsm-cloud/` for Docker Compose,
MinIO, and Minikube. Kubernetes tuning examples are configuration templates,
not credentials or production manifests.

## Workspace

- `helixdb`: graph/vector engine, query compiler, storage, and protocol types
- `helix-container`: HTTP server binary
- `helix-cli`: local project and deployment CLI
- `hbuild`: build helper
- `get_routes`: handler-registration proc macro
- `vendor/slatedb`: vendored Apache-2.0 SlateDB fork with separate provenance

## Documentation

- [Architecture](docs/architecture.md)
- [API reference](docs/api-reference.md)
- [Operations](docs/operations.md)
- [HelixQL grammar](docs/grammar.md)
- [PostgreSQL ingestion tests](docs/postgres-tests.md)
- [Roadmap](ROADMAP.md)
- [Provenance](PROVENANCE.md)

## Compatibility

The public project is named Helion and its repository is
`Context-Engine-AI/helion-db`. Existing `helix`, `helixdb`, and
`.helix/repo/helix-db` identifiers remain where changing them would break CLI,
crate, query, or storage compatibility.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). By contributing, you agree to license
your contribution under `GPL-3.0-only`.

## Maintainers

Helion is built and maintained at
[Context Engine AI](https://github.com/Context-Engine-AI) by:

- [@m1rl0k](https://github.com/m1rl0k)
- [@gigs-eng](https://github.com/gigs-eng)
- [@mikahoy045](https://github.com/mikahoy045)

## License

Helion is licensed under the GNU General Public License version 3 only. See
[LICENSE](LICENSE), [COPYRIGHT](COPYRIGHT), and [PROVENANCE.md](PROVENANCE.md).
