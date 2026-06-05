# Changelog

All notable changes to this project will be documented in this file.

This project follows a lightweight changelog format. Version numbers are intended to follow semantic versioning once release artifacts are published.

## [Unreleased]

### Added

- Versioned analysis-settings save/load through `mdfoam_project.json`.
- Reproducibility manifests containing application version, analysis settings, input file metadata, mesh statistics, and per-case result summaries.
- Automatic `analysis_manifest.json` inclusion in CSV export folders.

## [0.1.1] - 2026-06-04

### Added

- UI language selector with Japanese, English, Chinese, Spanish, and Hindi labels.
- GUI regression coverage for language switching without changing internal settings.
- README documentation for the supported UI languages.

### Changed

- Graph, source, theory, and visualization mode controls now keep stable internal values independent of the displayed language.

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

- This release is source-first and does not include packaged desktop binaries.
- Large real mdFOAM/OpenFOAM case data is intentionally excluded from the repository.
