# Contributing to Helion

Thank you for contributing to Helion.

## License of contributions

Helion is licensed under `GPL-3.0-only`. By submitting a contribution, you
represent that you have the right to submit it and agree that it may be
distributed under `GPL-3.0-only` as part of Helion.

Do not submit code copied from AGPL versions of upstream HelixDB or from any
source whose license is incompatible with GPLv3. Preserve third-party notices
when incorporating compatible material.

## Development workflow

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Run formatting and the relevant tests.
4. Update documentation when interfaces or workflows change.
5. Open a pull request describing the motivation, validation, and compatibility
   impact.

Common checks:

```bash
cargo fmt --all -- --check
cargo test --workspace --locked
```

For changes to storage backends, also run the applicable Docker Compose or
Minikube verification under `deploy/`.

## Reporting issues

Use the repository issue templates for reproducible bugs and feature requests.
Report security vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
