# Provenance

Helion is an independently maintained continuation of the GPL-licensed HelixDB
codebase.

## Upstream source boundary

The clean upstream GPL source anchor is:

```text
Repository: https://github.com/HelixDB/helix-db
Commit:    71ebdb9896ff1a2448eb55ad26014ec8428c8086
Date:      2025-05-11
License:   GNU General Public License v3.0
```

At that commit, the root `LICENSE`, upstream README, and crate metadata all
declared GPLv3.

The next upstream commit, `25ac2e55d7f65aabce48c27ef2424e4227a36307`,
changed only `README.md`, including its license wording. It did not change the
`helixdb` source tree. Helion therefore uses `71ebdb...` as its unambiguous GPL
source anchor rather than relying on that transitional documentation commit.

No source from the later AGPL-licensed upstream line was copied into Helion.
Later upstream versions may be consulted only to describe desired outcomes for
clean-room, independently written work.

## Context Engine AI modifications

Context Engine AI began modifying the GPL source in April 2026. Those
modifications include storage, vector, graph, compatibility API, operational,
and documentation work. Context Engine AI licenses its contributions under
`GPL-3.0-only` as part of the combined Helion work.

## Third-party components

Third-party code retains its own copyright and attribution notices. In
particular, `vendor/slatedb/` remains subject to its included Apache-2.0 license
and provenance documentation. The combined Helion distribution is conveyed
under GPL-3.0-only where the GPL applies.
