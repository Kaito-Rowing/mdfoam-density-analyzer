# AGENTS.md

This repository contains a Python desktop application for analyzing mdFOAM/OpenFOAM-style molecular simulation results.

The core rule for contributors and coding agents is simple: do not introduce a ParaView or `pvpython` dependency. The analyzer should read OpenFOAM ASCII files directly in Python whenever practical.

## Project Goal

The application is intended for reproducible batch analysis of nanoscale droplet evaporation studies. A typical workflow may involve tens to roughly 100 related cases. For density regions above a selected threshold, the app should compute and display:

- droplet volume;
- equivalent radius, `R_eq = (3V / (4*pi))^(1/3)`;
- contact angle;
- contact radius;
- evaporation completion time;
- one summary table row per case;
- selected-case plots for volume, equivalent radius, contact angle, and contact radius;
- an all-case evaporation completion time plot;
- CSV and PNG exports.

## Repository Data Policy

Do not commit real simulation outputs, large case directories, generated exports, or local batch artifacts.

This repository intentionally does not include sample mdFOAM/OpenFOAM case data. Local validation folders such as `run001_x001/` may exist on a developer machine, but they must stay outside Git.

Previously removed sample or batch artifacts included names such as `initial/`, `initial_a/`, `main/`, `AllDisMpi_*`, `batch_*.out`, `batch_status.txt`, `case_info.txt`, and `log.mapFields`. Keep similar large or run-specific data out of the repository.

Tiny synthetic fixtures under `tests/fixtures/` are allowed when they are small, intentional, and useful for regression tests.

## Expected Input Structure

Multiple-case parent folder:

```text
parent/
  case001/
    main/
      constant/polyMesh/
      <time>/
        rhoM_water
        ...
  case002/
    main/
      ...
```

Single-case folder:

```text
case001/
  main/
    constant/polyMesh/
    <time>/
      rhoM_water
      ...
```

Case discovery should follow the current application behavior:

- if the selected folder itself contains `main/`, treat it as one case;
- if child folders under the selected folder contain `main/`, treat those children as multiple cases;
- prefer reconstructed time directories such as `main/<time>/<density field>`;
- if reconstructed fields are unavailable, read and aggregate `main/processor*/<time>/<density field>`;
- numeric time directories such as `main/0` should be analyzed only when the selected density field exists there.

## Data Format Assumptions

- The default density field is `rhoM_water`.
- Density fields are OpenFOAM ASCII `volScalarField` files.
- Density is treated as cell-centered scalar data.
- `internalField nonuniform List<scalar>` is expected to contain one scalar per cell.
- The GUI should allow selecting fields named like `rhoM_*` or `rhoN_*`.
- Contact angle and contact radius calculations use density-threshold contour points reconstructed from cell-center data.
- The droplet may cross periodic boundaries in x/y, so xy periodic unwrapping should remain enabled by default.

## Cell Volumes and Cell Centers

Most target cases do not provide a separate cell-volume field such as `V`.

Compute cell volumes and centers directly from `constant/polyMesh` whenever possible, using:

- `points`;
- `faces`;
- `owner`;
- `neighbour`.

Do not assume constant cell volume. Earlier validation data contained more than one cell volume.

Only fall back to user-entered cell volume or `dx`, `dy`, and `dz` when mesh-derived volume calculation is unavailable. In that fallback mode, cell centers are not available, so contact angle and contact radius should be left blank.

## Contact Angle and Contact Radius Method

The contact-angle implementation should stay close to the previous workflow of extracting contour points in ParaView and processing them with `Sesshoku6.py`, while keeping the actual implementation inside Python.

Expected behavior:

- reconstruct density-threshold contour points by linearly interpolating neighboring cell-center pairs that cross the threshold;
- use the contour point cloud as the droplet surface point set;
- estimate the substrate height as the mean z value of points at or below `z_min + 0.05 * z_height`;
- select fit points from `z_min + fit_lower * z_height` to `z_min + fit_upper * z_height`;
- keep the default fit range at `fit_lower=0.5`, `fit_upper=1.0`;
- fit a sphere using `x^2 + y^2 + z^2 + Dx + Ey + Fz + G = 0`;
- compute `theta = acos((z_base - zc) / R)` in degrees;
- compute `r_contact = R * sin(theta)`;
- return blank contact metrics when there are fewer than 4 fit points, `z_height == 0`, or `R < 1e-12`;
- keep xy periodic correction enabled by default, using mesh x/y bounds to estimate periodic lengths and unwrap around a circular mean center.

## Default Analysis Settings

- Density threshold: `500`
- Zero-volume tolerance: `0`
- Consecutive zero count: `3`
- Contact-angle fit lower bound: `0.5`
- Contact-angle fit upper bound: `1.0`
- Average contact-angle range: `100%`
- xy periodic correction: enabled

Evaporation completion time is the first time in the consecutive zero-volume sequence, not the last time.

Example:

```text
1.17e-08: non-zero
1.18e-08: non-zero
1.19e-08: zero
1.20e-08: zero
1.21e-08: zero
```

With a consecutive zero count of `3`, the evaporation completion time is `1.19e-08`.

## Historical Validation Reference

The original sample data has been removed from the repository. Earlier validation with `rhoM_water >= 500` used the following reference values:

- time count: `127`;
- maximum volume: approximately `3.97004541958126e-26`;
- final volume: `0`;
- evaporation completion time: `1.19e-08`;
- cell count: `3825`;
- more than one cell volume existed;
- contact angle near `1e-10`: approximately `81.10057426317968 deg`;
- contact radius near `1e-10`: approximately `2.8254863800372293e-09 m`;
- late time steps with too few droplet or fit points may have blank contact metrics.

These values are only regression references. Equivalent external case data is required to reproduce them today.

## Main Files

- `app.py`: application entry point.
- `run_app.bat`: Windows launcher that can point at a local Miniconda Python.
- `Sesshoku6.py`: original contact-angle script used as a methodological reference; the app should use the ported in-app implementation instead.
- `src/mdfoam_analyzer/openfoam.py`: OpenFOAM ASCII parsing and mesh-derived volume/center calculation.
- `src/mdfoam_analyzer/analysis.py`: case discovery, metrics calculation, evaporation time detection, and CSV export.
- `src/mdfoam_analyzer/gui.py`: PySide6 GUI, tables, plots, Run/Stop workflow, CSV/PNG/GIF export, and visualization.

## Running the App

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the app:

```powershell
python app.py
```

On Windows:

```powershell
.\run_app.bat
```

## GUI Expectations

- The GUI should support Japanese, English, Chinese, Spanish, and Hindi labels.
- Internal behavior must not depend on translated display strings; use stable internal values for combo boxes and modes.
- The UI should support selecting a parent folder, case list, density field, density threshold, zero tolerance, consecutive zero count, and fallback cell-volume or `dx/dy/dz` values.
- Contact-angle controls should include fit lower/upper bounds, average contact-angle range, and xy periodic correction.
- Analysis must run in the background and support Stop.
- The result table should include initial contact angle, final valid contact angle, average contact angle, initial contact radius, and final valid contact radius.
- Table headers should be clickable for column selection.
- `Ctrl+C` should copy selected table cells as TSV suitable for Excel.
- Volume, equivalent radius, contact angle, and contact radius plots should be point plots, not connected line plots.
- The all-case evaporation completion time view should be a bar chart.
- Time-series CSV output must include `contact_angle_deg`, `contact_radius`, and `contact_fit_point_count`.

## Development Rules

- Do not add ParaView or `pvpython` dependencies.
- Preserve the direct OpenFOAM ASCII parsing approach.
- Keep `rhoM_water` as the default density field.
- Do not change evaporation completion time to the end of the zero sequence.
- Do not assume constant cell volume.
- Keep contact-angle calculation based on reconstructed threshold contour points and the sphere-fit method described above.
- Keep xy periodic correction enabled by default.
- Keep graph output for volume, equivalent radius, contact angle, and contact radius as point plots.
- Avoid committing large simulation outputs, generated case directories, caches, or local exports.

## Useful Future Work

- Performance validation on parent folders containing roughly 100 cases.
- Analysis result caching.
- Case-name parameter extraction for tables and plot axes.
- Reloading exported time-series CSV files for plotting.
- Comparing multiple density thresholds.
- Unit switching for volume and radius.
- Filtering result tables to error cases.
- Overlaying multiple selected cases in a single graph.
- In-GUI cross-section views for contact-angle fitting.
