# Repository Instructions

Helion is the GPL-3.0-only public fork and independent continuation documented
in `PROVENANCE.md`.

## Licensing and provenance

- Keep the root `LICENSE`, `COPYRIGHT`, and `PROVENANCE.md` consistent.
- New first-party crates must declare `license = "GPL-3.0-only"`.
- Preserve third-party license and attribution files, including the vendored
  SlateDB license and provenance.
- Do not copy source from AGPL versions of upstream HelixDB. External versions
  may inform clean-room requirements, tests, and independently written designs.

## Compatibility

- Use Helion for public branding and `Context-Engine-AI/helion-db` for active
  repository links.
- Preserve compatibility-sensitive names such as `helix`, `helixdb`, and
  `.helix/repo/helix-db` unless a migration is explicitly in scope.
- Audit `helix-cli`, `helix-container`, and `hbuild` together before changing
  install or runtime paths.

## Public-repository hygiene

- Never commit credentials, provider account identifiers, live bucket names,
  private checkout paths, or organization-specific production manifests.
- Keep deployment examples generic and parameterized.
- Update user-facing documentation when behavior or workflows change.

## Verification

- Prefer targeted tests for changed behavior.
- Run `cargo fmt --all -- --check` and relevant workspace tests before release.
- Run a secret scan over every ref intended for publication.
- Release binaries and containers only from a public source tag containing the
  exact corresponding source and build scripts.
