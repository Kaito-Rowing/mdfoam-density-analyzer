# Security Policy

## Supported Versions

This project is early-stage. Security fixes are handled on the default branch unless a maintainer documents a release support policy later.

## Reporting Security Issues

Please do not open a public GitHub Issue for sensitive security reports involving:

- SSH/SFTP connection handling;
- credential storage;
- private key handling;
- password or passphrase handling;
- remote file access;
- local cache behavior for remote results.

If you need to report a sensitive issue, contact the maintainer through the maintainer's GitHub profile rather than posting secrets or private details in an Issue.

## What Not to Share Publicly

Do not paste or upload any of the following into GitHub Issues, pull requests, screenshots, or logs:

- private keys;
- passwords or passphrases;
- SSH host details that should remain private;
- access tokens;
- unpublished experimental data;
- full simulation cases containing research data.

When possible, reproduce the issue using synthetic data or a minimal text fragment that does not contain secrets or private research content.

## Scope

Security reports related to this project are most likely to involve local desktop behavior, SSH/SFTP access, credential storage, and handling of remote case files. General OpenFOAM or mdFOAM security issues should be reported to their respective projects.
