# Contributing

Thank you for your interest in improving mdFOAM Density Analyzer.

This project is focused on reproducible quantitative post-processing for mdFOAM/OpenFOAM-style molecular simulation outputs. Contributions should preserve the current direction: direct Python parsing of OpenFOAM ASCII files without adding a ParaView or pvpython dependency.

## Development Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python app.py
```

Run tests:

```powershell
python -m unittest discover
```

## Case Data Policy

mdFOAM/OpenFOAM ASCII case data can become very large. Do not commit real simulation outputs, generated case directories, local cache directories, SSH/SFTP cache contents, exported PNG/GIF files, or CSV exports unless a maintainer explicitly asks for a tiny fixture.

For bug reports and tests, prefer:

- synthetic data generated inside a test;
- small hand-written OpenFOAM ASCII text fragments;
- minimal directory fixtures with only the files needed to reproduce the behavior.

If a real research case is needed to explain a problem, describe the structure and relevant field snippets instead of attaching the full case.

## Project Constraints

- Do not add a ParaView or pvpython dependency.
- Keep OpenFOAM ASCII parsing in Python.
- Keep the default density field as `rhoM_water` unless there is a clear compatibility reason to change it.
- Do not assume constant cell volume.
- Preserve the existing evaporation-time definition: the first time in the consecutive zero-volume sequence.
- Keep contact angle calculation aligned with the current contour-point and sphere-fit approach.

## Tests and Manual Checks

When changing any of the following areas, include tests or clearly describe manual verification in the pull request:

- OpenFOAM ASCII reader behavior;
- mesh geometry, cell volume, or cell center calculation;
- contact angle or contact radius calculation;
- SSH/SFTP discovery, sync, caching, or credential behavior;
- GUI behavior, export behavior, or visualization behavior.

For GUI-only changes, automated coverage may not always be practical. In that case, document the platform, command used to start the app, and the manual workflow you checked.

## Pull Requests

Before opening a pull request:

- keep changes scoped to one topic;
- avoid unrelated formatting churn;
- run relevant tests when possible;
- include a concise summary and verification notes;
- mention any limitations or intentionally untested areas.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
