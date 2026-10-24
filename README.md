# Multipath Channel Simulation Library

***This entire library has been vibe coded for a project related to multipath fading analysis. None of the code was handwritten, only specific design choices were given. The model used was Claude Sonnet 5.5, for anyone curious.***

Python tools for simulating indoor multipath in a 2D floor plan or a 3D room (with obstacles) and turning the result into complex channel responses for WiFi, 5G and upper mid-band carriers. This repo holds the **simulation and data-preparation modules only**; fading statistics, diversity and other analysis are built on top of the saved channel files.

There are two independent chains:

- **2D**: a pure-numpy image-source tracer (`paths2d`) and a channel builder with Fresnel physics (`channel2d`).
- **3D**: a path tracer built on [Sionna RT](https://github.com/NVlabs/sionna-rt) (`trace3d`) and a channel builder with a choice of carrier-phase model and receive arrays (`channel3d`).

## Repository layout

```
2D Modules/
    paths2d.py          stage 1 (2D): scenes, image-source tracing, path files
    channel2d.py        stage 2 (2D): bands, materials, Fresnel, H(f), helpers
3D Modules/
    trace3d.py          stage 1 (3D): scene spec, Sionna RT tracer, 3D path files
    channel3d.py        stage 2 (3D): phase modes, receive arrays, channel files
docs/
    docs_paths2d_reference.pdf      docs_channel2d_reference.pdf
    docs_trace3d_reference.pdf      docs_channel3d_reference.pdf
tests/
    test_paths2d.py  test_channel2d.py  test_trace3d.py  test_channel3d.py
    inspect_paths2d.py              quick looks at traced 2D paths
README.md
```

`channel3d.py` imports `channel2d.py` (bands, the impulse response) and `trace3d.py` (the path-file format). Python must be able to find all four modules, see "Running from this layout" below.

## Pipelines

```
2D                                          3D
scene + Tx + Rx points                      SceneSpec + Tx + Rx points
   | paths2d.trace_paths                       | trace3d.trace_paths3d   (Sionna RT, one trace per carrier)
   v                                           v
paths .h5 (PathSet)                         paths .h5 (PathSet3D)
   | channel2d.build_channels                 | channel3d.build_channels3d
   v                                           v
channels .h5 (H[receiver, freq])            channels .h5 (H[receiver, element, freq])
   |                                           |
   +------------------> your analysis code <---+
```

2D stage 1 is frequency independent: one trace serves every band, material set and polarization. 3D stage 1 is **per carrier**, because Sionna evaluates reflection coefficients at the scene frequency; bandwidth, number of samples, phase mode, seed and arrays can all be changed in stage 2 without re-tracing.

## Contents

| File | Purpose |
|---|---|
| `paths2d.py` | 2D stage 1. Wall/obstacle scenes, image-source path finding (direct + specular bounces to a chosen order, with occlusion), receiver grids and arrays, HDF5 read/write. |
| `channel2d.py` | 2D stage 2. Band presets, ITU-R P.2040-3 materials, Fresnel reflection (TE/TM), free-space loss, H(f) synthesis, impulse response, delay spread, HDF5 read/write, and helpers for selecting rows and array placements and for local normalization. |
| `trace3d.py` | 3D stage 1. `SceneSpec` (rectangular room plus box obstacles with ITU materials), `trace_paths3d` (line of sight + specular reflections via Sionna RT, one isotropic vertically polarized antenna at each end), `make_rx_grid3d`, and the 3D path file (`PathSet3D`, `save_pathset3d`, `load_pathset3d`). The file-format part does not need Sionna. |
| `channel3d.py` | 3D stage 2. `build_channels3d` with two phase models (`geometric`, `random_carrier`), plane-wave receive arrays (`make_ula3d`), `ChannelSet3D`, HDF5 read/write, `make_channels3d`. Re-exports `BandConfig`, `BANDS` and `cir` from `channel2d`. |
| `tests/inspect_paths2d.py` | Quick looks at traced 2D paths: side-by-side path plots, array view, per-receiver metrics, profiles along a row. |
| `tests/test_*.py` | Verification suites (61 tests, see Tests). |
| `docs/` | Function-by-function PDF references. The 2D references include executed examples; `docs_channel3d_reference.pdf` has four executed trace3d + channel3d examples; `docs_trace3d_reference.pdf` is descriptions only. |

## Install

### 2D modules and analysis (light environment)

```bash
pip install numpy h5py scipy matplotlib pytest
```

`scipy` is only needed by `channel2d.local_normalize`; `matplotlib` only by plotting. `channel3d.py` and the file-format half of `trace3d.py` also work in this environment, because Sionna is imported lazily inside the tracer only. Developed on Python 3.13 with numpy 2.x.

### 3D tracing (Sionna RT environment)

Only the step that traces paths (`trace3d.trace_paths3d`) needs Sionna RT. A separate conda environment keeps its heavy dependencies away from the analysis environment:

```bash
conda create -n env-trace3d python=3.11      # or any version your Sionna/Mitsuba release supports
conda activate env-trace3d
pip install sionna-rt h5py numpy scipy matplotlib pytest
```

`sionna-rt` brings in Mitsuba 3 and Dr.Jit. The 3D code was developed and tested against `sionna-rt 2.2.0`, `mitsuba 3.9.1` and `drjit 1.5.0` (Python 3.13, CPU backend). Sionna RT has the same requirements as Mitsuba 3; consult the [Sionna RT repository](https://github.com/NVlabs/sionna-rt) and the [Sionna documentation](https://nvlabs.github.io/sionna/) for your platform. TensorFlow is not needed.

**GPU (NVIDIA).** Install a recent NVIDIA driver; OptiX ships with the driver, so CUDA and OptiX do not need to be installed separately. On a laptop with hybrid graphics, make sure Windows runs Python on the NVIDIA GPU rather than the integrated one. Check that the CUDA backend loads:

```bash
nvidia-smi
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print(mi.Float(1.0))"
```

**CPU fallback.** Without a usable NVIDIA GPU, Dr.Jit needs LLVM (for example `apt install llvm`, or the LLVM installer on Windows). Check with `mi.set_variant('llvm_ad_mono_polarized')`. Everything works on the CPU, only slower; the first call is also slow because of kernel compilation.

**Which backend is used.** `trace_paths3d(..., variant=None)` tries the CUDA variant first, checks that the GPU driver really loads, and falls back to the LLVM variant otherwise. The chosen variant is stored in the path file (`ps.meta["variant"]`). Pass `variant="cuda_ad_mono_polarized"` to force the GPU, so that a failure raises an error instead of falling back silently, or `variant="llvm_ad_mono_polarized"` to force the CPU.

**Check the install.** Run the 3D tracer tests; they skip automatically if `sionna-rt` is missing, and `TRACE3D_VARIANT` forces a Mitsuba variant:

```bash
pytest tests/test_trace3d.py -v
```

On a GPU, path counts and lengths should match the CPU results; amplitudes may differ at the single-precision level.

### Running from this layout

The code files live in two folders with a space in their names, and `channel3d.py` imports the others as plain modules. The simplest way is to put both folders on `PYTHONPATH`.

Windows (PowerShell):

```powershell
$env:PYTHONPATH = "$PWD\2D Modules;$PWD\3D Modules"
pytest tests -v
```

Linux or macOS:

```bash
export PYTHONPATH="$PWD/2D Modules:$PWD/3D Modules"
pytest tests -v
```

In VS Code, the same can be set once in a `.env` file with `PYTHONPATH=2D Modules;3D Modules` (use `:` on Linux/macOS), or by renaming the folders without spaces, for example `modules2d` and `modules3d`.

## Quick start

### 2D

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

Command line for 2D stage 2: `python channel2d.py paths.h5 wifi_5ghz out.h5 TE`

### 3D

```python
import trace3d as t3
import channel3d as c3

# Stage 1: room with a box obstacle; trace at the carrier of the band you will use
spec = t3.SceneSpec((12.0, 4.0, 3.0), boxes=((6.0, 1.5, 0.0, 7.0, 2.5, 1.2),))
band = c3.BANDS["wifi_5ghz"]
rx, xs, ys = t3.make_rx_grid3d((8.0, 11.0), (0.8, 3.2), 1.0, 0.15)
ps = t3.trace_paths3d(spec, tx=(1.0, 2.0, 1.5), rx_points=rx, fc=band.fc, max_order=3)
t3.save_pathset3d("room_5p5.h5", ps)

# Stage 2: phase model and receive array are chosen here, no re-tracing
elem = c3.make_ula3d(4, band.wavelength / 2, axis=(0, 1, 0))   # optional 4-element array
ch = c3.build_channels3d(ps, band, phase_mode="random_carrier", elem=elem, seed=0)
c3.save_channels3d("room_5p5_chan.h5", ch, source="room_5p5.h5")

h = ch.narrowband()                                  # (receivers, elements)
ok = ch.rx_valid & (ch.n_paths > 0)                  # always mask before statistics
```

Trace once per carrier. `build_channels3d` refuses a band whose carrier differs from the one the paths were traced at (`check_fc=False` overrides this for sensitivity checks).

## Tests

```bash
pytest tests -v        # with PYTHONPATH set as above
```

| Suite | Tests | Needs Sionna |
|---|---|---|
| `test_paths2d.py`, `test_channel2d.py` | 40 | no |
| `test_channel3d.py` (covers `channel3d` and the path-file part of `trace3d`) | 13 | no |
| `test_trace3d.py` | 8 (2 scene tests need no Sionna; 6 tracer tests skip without it) | yes |

The 2D suites check the code against closed forms: the recursive image engine against the rectangle image lattice, reciprocity, the specular law, Friis free-space loss, comb-filter nulls, Fresnel limits and the Brewster angle, perfect-conductor rooms, and HDF5 round trips.

The 3D suites check: agreement between channel2d and channel3d on the same physics (geometric mode), single-path carrier phase and gain convention, array steering against exact element distances, Rayleigh and Rician moments of the random-carrier mode, the sin(kd)/(kd) array correlation for isotropic arrivals, file round trips and error handling. The tracer tests check the Sionna output against the image-source lattice of an empty box (orders 1 to 3), the line-of-sight amplitude and phase convention, single reflections off all six surfaces against polarized Fresnel coefficients, shadowing by a box, and batch-size independence.

## Model and conventions

Common to both chains:

- Scalar-amplitude channel per path, quasi-static (no motion), isotropic antennas, no noise or transmit power.
- Path amplitude `a = (lambda / 4 pi L) * (reflection factors)`, 1/L spreading, referenced to the carrier.
- Frequency samples span the band with the carrier at index `N // 2`. Phases use the absolute frequency.
- Rows of `H`: NaN means an invalid receiver (inside an obstacle or outside the room); zeros mean a valid receiver with no path.

2D (`paths2d`, `channel2d`):

- 2D top-down geometry. TE (E field out of the plane, i.e. vertical antennas) is the default and is exact. TM is a scalar approximation; use it only as a sensitivity check.

3D (`trace3d`, `channel3d`):

- Rectangular room with axis-aligned box obstacles. One isotropic, vertically polarized antenna at the transmitter and at each receiver. Line of sight and specular reflections only; diffraction, diffuse scattering and refraction (transmission) are switched off.
- Polarization is carried through every bounce by Sionna, so floor and ceiling act mostly as TM and the side walls mostly as TE, with a small mix because rays are tilted.
- The path file stores the amplitude **without** the propagation phase (Sionna's own coefficient includes it; `trace_paths3d` removes it). Arrival and departure directions are unit vectors; `aoa` points toward where the ray comes from.
- Walls are slabs of ITU-R P.2040 material. `material_thickness=1.0` m (default) makes them behave as half-spaces, matching the channel2d Fresnel model; thinner slabs add slab interference.
- Two carrier-phase models in stage 2: `geometric` (exact path phase) and `random_carrier` (uniform random phase per receiver and path; delays, amplitudes and array steering stay geometric). In random mode neighbouring receiver points are uncorrelated by construction, so use the array option for anything spatial. Receive arrays use a plane-wave model about the traced point.

## Limitations

- Specular reflection only: no diffraction, diffuse scattering or transmission, so shadows behind obstacles are hard-edged and the multipath is sparser than in reality (more so at higher frequencies). State this alongside any result.
- Materials and spreading are evaluated at the carrier only.
- No antenna patterns (one fixed isotropic element), transmit power or noise.
- The alias window `1/delta_f` is not checked against path lengths.
- 2D omits floor and ceiling bounces and polarization mixing.
- 3D path discovery by Sionna is statistical (shoot-and-bounce). It matched the analytic image lattice exactly up to order 3 on an empty box, but at order 4 one of 129 paths was missed even with 10 million rays. Check completeness for your own scene and `samples_per_src` before relying on high orders.
- Sionna computes in single precision: the geometric carrier phase carries about 1e-3 rad of error for paths of a few hundred metres (irrelevant in random-carrier mode).
- Receive arrays assume plane waves about the traced point; `build_channels3d` warns when the array offsets exceed 10 percent of the shortest path.
- 3D scenes are limited to a rectangular room with box obstacles. Other geometry needs a different scene builder.

## Data files

HDF5, versioned (`format_version`), with a `kind` attribute on the 3D files so a wrong file type is rejected on load. 2D path files store scene, Tx/Rx, and per-path length, order, wall ids, incidence angles, bounce points and angles of arrival/departure. 3D path files store Tx/Rx, validity flags, per-path length, order, complex amplitude and arrival/departure unit vectors, the trace frequency and the scene and tracer settings. Channel files store `H` (`[receiver, frequency]` in 2D, `[receiver, element, frequency]` in 3D), frequencies, receivers, band, material or phase-mode information and the stage 1 settings, so every file is self-describing. See the reference PDFs for the exact layouts.