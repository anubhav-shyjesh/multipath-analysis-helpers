# Multipath Channel Simulation Library

***This entrie library has been vibe coded for a project related to multipath fading analysis. None of the code was handwritten, only specific design choices were given. The model used was Claude Sonnet 5.5, for anyone curious.***

Python tools for simulating indoor multipath in a 2D floor plan (room or hallway, with obstacles) and turning the result into complex channel responses for WiFi, 5G and upper mid-band carriers. This repo holds the **simulation and data-preparation modules only**; fading statistics, diversity and other analysis are built on top of the saved channel files.

## Pipeline

```
scene + Tx + Rx points
        |  paths2d.trace_paths          stage 1: geometry only, no frequency
        v
   paths .h5   (PathSet)
        |  channel2d.build_channels     stage 2: band, materials, polarization
        v
 channels .h5  (ChannelSet: H[receiver, frequency])
        |
        v
   your analysis code
```

Stage 1 is the slow step. Stage 2 is fast and repeatable, so one trace serves every band, material set and polarization.

## Contents

| File | Purpose |
|---|---|
| `paths2d.py` | Stage 1. Wall/obstacle scenes, image-source path finding (direct + specular bounces to a chosen order, with occlusion), receiver grids and arrays, HDF5 read/write. |
| `channel2d.py` | Stage 2. Band presets, ITU-R P.2040-3 materials, Fresnel reflection (TE/TM), free-space loss, H(f) synthesis, impulse response, delay spread, HDF5 read/write, and helpers for selecting rows and array placements and for local normalization. |
| `inspect_paths.py` | Quick looks at traced paths: side-by-side path plots, array view, per-receiver metrics, profiles along a row. |
| `test_paths2d.py`, `test_channel2d.py` | Verification suites (40 tests). |
| `docs/` | `paths2d_reference.pdf` and `channel2d_reference.pdf`: function-by-function references with executed examples. |

## Install

```bash
pip install numpy h5py scipy matplotlib pytest
```

`scipy` is only needed by `channel2d.local_normalize`; `matplotlib` only by plotting. Developed on Python 3.13 with numpy 2.x.

## Quick start

```python
import paths2d as p2
import channel2d as c2

# Stage 1: hallway with a box obstacle, a row of receivers 1 mm apart
scene = p2.add_box(p2.rectangle_room(20.0, 3.0), 8.0, 0.8, 9.0, 2.2)
rx, xs, _ = p2.make_rx_grid((4.0, 4.5), (1.5, 1.5), 0.001)
ps = p2.trace_paths(scene, (2.0, 1.5), rx, max_order=4)
p2.save_pathset("hallway_o4.h5", ps)

# Stage 2: any band, any material set, no re-tracing
band = c2.BANDS["wifi_5ghz"]
ch = c2.build_channels(ps, band, material_map={"wall": "concrete"}, pol="TE")
c2.save_channels("hallway_wifi5.h5", ch, source="hallway_o4.h5")

env = ch.envelope()                     # narrowband |H| per receiver
ok = ch.rx_valid & (ch.n_paths > 0)     # always mask before statistics
```

Command line for stage 2: `python channel2d.py paths.h5 wifi_5ghz out.h5 TE`

## Tests

```bash
pytest -v
```

The suites check the code against closed forms: the recursive image engine against the rectangle image lattice, reciprocity, the specular law, Friis free-space loss, comb-filter nulls, Fresnel limits and the Brewster angle, perfect-conductor rooms, and HDF5 round trips.

## Model and conventions

- Scalar field, quasi-static (no motion), isotropic antennas, 2D top-down geometry.
- Path gain `a = (lambda / 4 pi L) * product of Fresnel coefficients`, i.e. 1/L spreading, so amplitudes carry over to a 3D model.
- Frequency samples span the band with the carrier at index `N // 2`. Phases use the absolute frequency.
- TE (E field out of the plane, i.e. vertical antennas) is the default and is exact. TM is a scalar approximation; use it only as a sensitivity check.
- Rows of `H`: NaN means an invalid receiver (inside an obstacle or outside the room); zeros mean a valid receiver with no path.

## Limitations

- Specular reflection only: no diffraction, diffuse scattering or transmission, so shadows behind obstacles are hard-edged.
- 2D omits floor and ceiling bounces and polarization mixing.
- Materials and spreading are evaluated at the carrier only.
- No antenna patterns, transmit power or noise.
- The alias window `1/delta_f` is not checked against path lengths.

## Data files

HDF5, versioned (`format_version`). Path files store scene, Tx/Rx, and per-path length, order, wall ids, incidence angles, bounce points and angles of arrival/departure. Channel files store `H`, frequencies, receivers, band, material map and the stage 1 settings, so every file is self-describing. See the reference PDFs for the exact layouts.
