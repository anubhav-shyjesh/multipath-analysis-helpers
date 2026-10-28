"""
channel3d.py -- Stage 2 of the 3D simulator: 3D path lists (trace3d.PathSet3D) -> complex channels.

Unlike channel2d.py, material and Fresnel physics are NOT computed here: the 3D tracer already stored
one complex amplitude ``a`` per path (spreading + reflections + polarization, at the trace frequency).
This module only applies the band, chooses how the carrier phase of each path is set, and optionally
evaluates a receive array. Bands, the CIR and the selection helpers are reused from channel2d.

Phase model. Per receiver, element m, frequency f_n:

    H_m(f_n) = sum_k a_k exp(j theta_k) exp(-j 2 pi (f_n - fc) tau_k) exp(+j 2 pi f_n (d_m . u_k) / C0)

    tau_k = L_k / C0, d_m = element offset from the traced point, u_k = arrival unit vector (toward the source).

    phase_mode="geometric":      theta_k = -2 pi fc tau_k   (exact path phase; equals channel2d's model)
    phase_mode="random_carrier": theta_k ~ Uniform[0, 2 pi), independent per (receiver, path), common
                                 to all frequencies and array elements of that receiver

In both modes the delay phase across the band, the array steering and the amplitudes come from the
geometry. Only the carrier phase differs. Consequences to keep in mind:
  * Random mode gives every receiver point its own independent phases, so neighbouring receiver points
    are NOT spatially correlated. Use the array option (steering) for anything spatial.
  * Arrays use the plane-wave model about the traced point: keep element offsets small against the
    distance to the nearest reflection point (a warning is issued otherwise).
  * With many comparable paths, uniform random phases yield Rayleigh/Rician statistics by construction,
    so what the comparison tests is whether the geometry supplies enough comparable paths.

Sections: 1 array helper | 2 ChannelSet3D and HDF5 | 3 synthesis
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import h5py
import numpy as np

from sim_tools.modules_2d import channel2d as c2
from sim_tools.modules_2d.channel2d import BandConfig, BANDS, cir          # re-exported for convenience  # noqa: F401
from sim_tools.modules_3d.trace3d import PathSet3D, C0

FORMAT_VERSION = "1.0"
PHASE_MODES = ("geometric", "random_carrier")


# ---------------------------------------------------------------------------
# 1. Array helper
# ---------------------------------------------------------------------------

def make_ula3d(n_elem: int, spacing: float, axis=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Element offsets (n_elem, 3) of a uniform linear array centred on the origin.

    Parameters
    ----------
    n_elem : number of elements
    spacing : element spacing in metres
    axis : array direction (normalized internally)
    """
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    return ((np.arange(n_elem) - (n_elem - 1) / 2) * spacing)[:, None] * ax[None, :]


# ---------------------------------------------------------------------------
# 2. ChannelSet3D and HDF5
# ---------------------------------------------------------------------------

@dataclass
class ChannelSet3D:
    """Complex channels of one Tx, R receiver points, each with M array elements, one band.

    Attributes
    ----------
    H : (R, M, N) complex   NaN rows for invalid receivers, zeros for valid receivers with no path
    freqs : (N,)            absolute frequencies (Hz); index N//2 is the carrier
    rx : (R, 3)             receiver (array centre) positions
    elem : (M, 3)           element offsets from the receiver position (zeros for a single antenna)
    rx_valid : (R,) bool
    n_paths : (R,) int
    tx : (3,)
    band : BandConfig
    meta : dict             phase_mode, seed, source file, stage-1 settings
    """

    H: np.ndarray
    freqs: np.ndarray
    rx: np.ndarray
    elem: np.ndarray
    rx_valid: np.ndarray
    n_paths: np.ndarray
    tx: np.ndarray
    band: BandConfig
    meta: dict = field(default_factory=dict)

    def narrowband(self) -> np.ndarray:
        """(R, M) complex channel at the carrier."""
        return self.H[..., self.band.n_sc // 2]

    def envelope(self) -> np.ndarray:
        """(R, M) magnitude of the narrowband channel."""
        return np.abs(self.narrowband())

    def wideband_gain_db(self) -> np.ndarray:
        """(R, M) 10*log10 of the frequency-averaged |H|^2 (-inf for shadowed receivers)."""
        with np.errstate(divide="ignore"):
            return 10 * np.log10(np.mean(np.abs(self.H) ** 2, axis=-1))


def save_channels3d(filename: str, ch: ChannelSet3D, source: str | None = None) -> None:
    """Write a ChannelSet3D to HDF5 (gzip level 4). Layout: /H (R,M,N) complex, /freqs, /elem,
    /rx/{xyz,valid,n_paths}, /tx/xyz; attributes format_version, kind="channels3d", created_utc,
    band_json, meta_json, source."""
    with h5py.File(filename, "w") as f:
        f.attrs.update(format_version=FORMAT_VERSION, kind="channels3d",
                       created_utc=datetime.now(timezone.utc).isoformat(),
                       band_json=json.dumps(asdict(ch.band)), meta_json=json.dumps(ch.meta, default=float),
                       source=source or "")
        f.create_dataset("H", data=ch.H, compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("freqs", data=ch.freqs)
        f.create_dataset("elem", data=ch.elem)
        g = f.create_group("rx")
        g.create_dataset("xyz", data=ch.rx)
        g.create_dataset("valid", data=ch.rx_valid)
        g.create_dataset("n_paths", data=ch.n_paths)
        f.create_group("tx").create_dataset("xyz", data=ch.tx)


def load_channels3d(filename: str) -> ChannelSet3D:
    """Read a file written by ``save_channels3d``. Raises ValueError for another kind of file
    (e.g. a 2D channel file) or a different major format version."""
    with h5py.File(filename, "r") as f:
        if str(f.attrs.get("kind", "")) != "channels3d":
            raise ValueError(f"{filename} is not a 3D channel file")
        ver = str(f.attrs["format_version"])
        if ver.split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise ValueError(f"Unsupported format_version {ver} (expected {FORMAT_VERSION})")
        meta = json.loads(f.attrs["meta_json"])
        meta["source"] = str(f.attrs["source"])
        return ChannelSet3D(H=f["H"][()], freqs=f["freqs"][()], rx=f["rx/xyz"][()], elem=f["elem"][()],
                            rx_valid=f["rx/valid"][()].astype(bool), n_paths=f["rx/n_paths"][()],
                            tx=f["tx/xyz"][()], band=BandConfig(**json.loads(f.attrs["band_json"])),
                            meta=meta)


# ---------------------------------------------------------------------------
# 3. Synthesis
# ---------------------------------------------------------------------------

def build_channels3d(ps: PathSet3D, band: BandConfig, phase_mode: str = "random_carrier",
                     elem: np.ndarray | None = None, seed: int = 0, dtype=np.complex128,
                     check_fc: bool = True, progress: bool = False) -> ChannelSet3D:
    """Turn a PathSet3D into a ChannelSet3D for one band.

    Parameters
    ----------
    ps : PathSet3D from trace3d (traced at the band's carrier)
    band : BandConfig
    phase_mode : "random_carrier" (default) or "geometric"; see the module docstring
    elem : (M, 3) element offsets (see ``make_ula3d``); default one element at the traced point
    seed : seed of the random carrier phases (ignored in geometric mode). The phases are drawn for every
           path of the file at once, so a seed reproduces the same channel regardless of which
           receivers are valid. Use different seeds for independent realizations.
    dtype : complex128 (default) or complex64
    check_fc : raise if band.fc differs from the trace frequency (the stored amplitudes depend on it);
               False allows deliberate sensitivity checks
    progress : print coarse progress

    Returns
    -------
    ChannelSet3D with H of shape (R, M, N).
    """
    if phase_mode not in PHASE_MODES:
        raise ValueError(f"phase_mode must be one of {PHASE_MODES}")
    if check_fc and abs(band.fc - ps.fc_trace) > 1e-6 * ps.fc_trace:
        raise ValueError(f"Paths were traced at {ps.fc_trace / 1e9:g} GHz but the band carrier is "
                         f"{band.fc / 1e9:g} GHz; re-trace at the band frequency (or check_fc=False)")
    ps.validate()
    elem = np.zeros((1, 3)) if elem is None else np.atleast_2d(np.asarray(elem, float))
    if elem.ndim != 2 or elem.shape[1] != 3:
        raise ValueError("elem must have shape (M, 3)")
    M, R, N = len(elem), len(ps.rx), band.n_sc
    freqs, fc = band.freqs(), band.fc
    tau = ps.delays()

    if phase_mode == "geometric":
        theta = -2 * np.pi * fc * tau
    else:
        theta = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(tau))
    coef = ps.a * np.exp(1j * theta)                       # (P,) amplitude including carrier phase
    steer = np.any(elem != 0)
    if steer:
        shortest = np.min(ps.length) if len(ps.length) else np.inf
        if np.abs(elem).max() > 0.1 * shortest:
            warnings.warn("Array offsets exceed 10% of the shortest path length; the plane-wave "
                          "array model is inaccurate")

    H = np.full((R, M, N), np.nan + 1j * np.nan, dtype=dtype)
    df = freqs - fc
    step = max(R // 10, 1)
    for r in range(R):
        if ps.rx_valid[r]:
            sl = ps.rx_slice(r)
            if sl.stop == sl.start:
                H[r] = 0
            else:
                base = coef[sl, None] * np.exp(-2j * np.pi * tau[sl, None] * df[None, :])      # (K, N)
                if steer:
                    delta = elem @ ps.aoa[sl].T                                                 # (M, K)
                    S = np.exp(2j * np.pi * delta[:, :, None] * freqs[None, None, :] / C0)      # (M, K, N)
                    H[r] = np.einsum("kn,mkn->mn", base, S)
                else:
                    H[r] = base.sum(axis=0)[None, :]
        if progress and (r + 1) % step == 0:
            print(f"synthesized {r + 1}/{R} receivers")

    meta = dict(phase_mode=phase_mode, seed=int(seed) if phase_mode == "random_carrier" else None,
                stage1=ps.meta, fc_trace=ps.fc_trace,
                gain_convention="a includes spreading, reflections, polarization; propagation phase "
                                "applied here; isotropic vertically polarized antennas")
    return ChannelSet3D(H=H, freqs=freqs, rx=ps.rx.copy(), elem=elem, rx_valid=ps.rx_valid.copy(),
                        n_paths=ps.n_paths(), tx=ps.tx.copy(), band=band, meta=meta)


def make_channels3d(paths_file: str, band: BandConfig, out_file: str, **kw) -> ChannelSet3D:
    """Convenience pipeline: load a 3D path file, build channels (kwargs go to ``build_channels3d``), save."""
    from trace3d import load_pathset3d

    ch = build_channels3d(load_pathset3d(paths_file), band, **kw)
    save_channels3d(out_file, ch, source=paths_file)
    return ch