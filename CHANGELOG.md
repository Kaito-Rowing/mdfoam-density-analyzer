# Changelog

All notable changes to this project will be documented in this file.

This project follows a lightweight changelog format. Version numbers are intended to follow semantic versioning once release artifacts are published.

## [0.1.0] - 2026-06-04

### Added

- Initial public project positioning for mdFOAM/OpenFOAM-style droplet post-processing.
- Direct Python parsing of OpenFOAM ASCII density fields and `polyMesh` files.
- Batch analysis for local and SSH/SFTP-accessed case directories.
- Droplet volume, equivalent radius, contact angle, contact radius, and evaporation-time calculations.
- CSV, PNG, and GIF export paths from the desktop GUI.
- OSS community metadata, including MIT license, contributing guide, security policy, support notes, issue templates, and pull request template.
- Pytest-based core logic tests with a synthetic OpenFOAM-like fixture.
- GitHub Actions CI for Python 3.11 and 3.12.

### Notes

- No release artifact has been published yet.
- Large real mdFOAM/OpenFOAM case data is intentionally excluded from the repository.
