"""
channel2d.py -- Stage 2 of the 2D multipath simulator: path lists -> complex channel responses.

Reads a PathSet written by paths2d.py (stage 1), applies a frequency band, material (Fresnel)
reflection and free-space loss, and produces the complex channel H(f) for every receiver. The
result is saved to HDF5 so the analysis stage (fading fits, diversity) only loads arrays.

Channel model (per receiver, isotropic 0 dBi antennas, scalar field, quasi-static):

    H(f_n) = sum_k  a_k * exp(-j 2 pi f_n L_k / C0),
    a_k    = [lambda_c / (4 pi L_k)] * prod_over_bounces Gamma_i(theta_i, fc)

L_k is the unfolded path length, Gamma the Fresnel reflection coefficient of the wall hit,
theta the incidence angle from stage 1. Frequencies f_n = fc + (n - N/2) * df, n = 0..N-1, so
index N//2 is exactly the carrier (the narrowband channel). Materials and spreading are
evaluated at fc (valid because B << fc for the bands of interest).

Polarization (2D top-down): "TE" = E field perpendicular to the plane of incidence, i.e. out of
the 2D plane (vertical antennas); "TM" = E field in the plane. Both coefficients use the basis
that is consistent along a ray, so the same formulas apply to every bounce.

Sections: 1 constants and bands | 2 materials and Fresnel | 3 gains and synthesis |
4 ChannelSet and HDF5 | 5 time domain and delay spread | 6 selection and normalization helpers
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import h5py
import numpy as np

import paths2d as p2

C0 = p2.C0
EPS0 = 8.8541878128e-12       # vacuum permittivity, F/m
FORMAT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# 1. Bands
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandConfig:
    """Frequency plan of one channel simulation.

    Attributes
    ----------
    name : str          label used in file names and plots
    fc : float          carrier frequency in Hz
    bandwidth : float   simulated bandwidth in Hz
    n_sc : int          number of frequency samples (must be even); spacing is bandwidth / n_sc
    """

    name: str
    fc: float
    bandwidth: float
    n_sc: int

    def __post_init__(self):
        if self.n_sc % 2:
            raise ValueError("n_sc must be even so the carrier sits exactly on index n_sc // 2")

    @property
    def delta_f(self) -> float:
        """Frequency-sample spacing in Hz."""
        return self.bandwidth / self.n_sc

    @property
    def wavelength(self) -> float:
        """Carrier wavelength in metres."""
        return C0 / self.fc

    def freqs(self) -> np.ndarray:
        """Absolute frequency grid, shape (n_sc,): fc + (n - n_sc/2) * delta_f."""
        return self.fc + (np.arange(self.n_sc) - self.n_sc // 2) * self.delta_f


# Illustrative presets (edit freely). n_sc sets the sampling grid, not the real OFDM numerology;
# 1/delta_f is the CIR alias window and should exceed your longest path delay.
BANDS = {
    "nr_n78":              BandConfig("nr_n78", 3.5e9, 100e6, 256),
    "wifi_5ghz":           BandConfig("wifi_5ghz", 5.5e9, 80e6, 256),
    "upper_midband_13ghz": BandConfig("upper_midband_13ghz", 13e9, 400e6, 512),
    "nr_n257_28ghz":       BandConfig("nr_n257_28ghz", 28e9, 400e6, 512),
}


# ---------------------------------------------------------------------------
# 2. Materials and Fresnel coefficients
# ---------------------------------------------------------------------------

# ITU-R P.2040-3, Table 3: eps_r = a * f^b, sigma = c * f^d (f in GHz, sigma in S/m).
# Tuple: (a, b, c, d, f_min_GHz, f_max_GHz); the range is indicative (a warning is issued outside it).
ITU = {
    "vacuum":        (1.0,   0, 0.0,    0.0,    0.001, 100),
    "concrete":      (5.24,  0, 0.0462, 0.7822, 1,     100),
    "brick":         (3.91,  0, 0.0238, 0.16,   1,     10),
    "plasterboard":  (2.73,  0, 0.0085, 0.9395, 1,     100),
    "wood":          (1.99,  0, 0.0047, 1.0718, 0.001, 100),
    "glass":         (6.31,  0, 0.0036, 1.3394, 0.1,   100),
    "ceiling_board": (1.48,  0, 0.0011, 1.0750, 1,     100),
    "chipboard":     (2.58,  0, 0.0217, 0.78,   1,     100),
    "plywood":       (2.71,  0, 0.33,   0.0,    1,     40),
    "marble":        (7.074, 0, 0.0055, 0.9262, 1,     60),
    "floorboard":    (3.66,  0, 0.0044, 1.3515, 50,    100),
    "metal":         (1.0,   0, 1e7,    0.0,    1,     100),
}
DEFAULT_MATERIAL_MAP = {"wall": "concrete"}   # Scene material name -> ITU key


def eta(material, f_hz: float) -> complex:
    """Complex relative permittivity eps' - j sigma / (2 pi f eps0) of a material at f_hz.

    Parameters
    ----------
    material : str or (eps_r, sigma) tuple
        An ITU key from ``ITU`` or explicit constants (relative permittivity, conductivity S/m).
    f_hz : float
        Frequency in Hz.
    """
    if isinstance(material, str):
        if material not in ITU:
            raise KeyError(f"Unknown material {material!r}. Known: {sorted(ITU)}. "
                           "Or pass (eps_r, sigma) in material_map.")
        a, b, c, d, fmin, fmax = ITU[material]
        g = f_hz / 1e9
        if not fmin <= g <= fmax:
            warnings.warn(f"{material}: {g:g} GHz is outside the ITU-R P.2040 range [{fmin}, {fmax}] GHz")
        eps, sigma = a * g ** b, c * g ** d
    else:
        eps, sigma = material
    return eps - 1j * sigma / (2 * np.pi * f_hz * EPS0)


def fresnel(theta, eta_c: complex, pol: str = "TE") -> np.ndarray:
    """Fresnel reflection coefficient of a flat interface (air to material).

    Parameters
    ----------
    theta : array-like   incidence angle from the wall normal, radians (0 = normal incidence)
    eta_c : complex      complex relative permittivity of the wall (see ``eta``)
    pol : "TE" or "TM"   E perpendicular to / in the plane of incidence

    Returns
    -------
    Complex coefficient(s), same shape as theta. Limits: perfect conductor gives -1 (TE) and +1 (TM);
    TM vanishes at the Brewster angle of a lossless dielectric.
    """
    theta = np.asarray(theta, float)
    cos, sin2 = np.cos(theta), np.sin(theta) ** 2
    root = np.sqrt(eta_c - sin2 + 0j)
    if pol == "TE":
        return (cos - root) / (cos + root)
    if pol == "TM":
        return (eta_c * cos - root) / (eta_c * cos + root)
    raise ValueError("pol must be 'TE' or 'TM'")


# ---------------------------------------------------------------------------
# 3. Path gains and channel synthesis
# ---------------------------------------------------------------------------

def path_gains(ps: p2.PathSet, band: BandConfig, material_map: dict | None = None,
               pol: str = "TE", fspl: bool = True) -> np.ndarray:
    """Complex amplitude a_k of every path in a PathSet (without the propagation phase).

    Parameters
    ----------
    ps : PathSet from stage 1
    band : BandConfig        reflection coefficients and free-space loss are evaluated at band.fc
    material_map : dict      Scene material name -> ITU key or (eps_r, sigma); merged over the default
                             {"wall": "concrete"}. Names equal to an ITU key need no entry.
    pol : "TE" or "TM"
    fspl : bool              True: amplitude lambda_c / (4 pi L) (true link gain); False: 1 / L

    Returns
    -------
    (P,) complex array aligned with the per-path arrays of ``ps``.
    """
    mm = {**DEFAULT_MATERIAL_MAP, **(material_map or {})}
    names = ps.scene.material_names
    etas = {int(m): eta(mm.get(names[m], names[m]), band.fc) for m in np.unique(ps.scene.material)}
    g = np.ones(len(ps.length), complex)
    for k in range(ps.walls.shape[1]):
        w = ps.walls[:, k]
        idx = np.where(w >= 0)[0]
        if idx.size == 0:
            continue
        mat, th = ps.scene.material[w[idx]], ps.inc_angle[idx, k]
        for m in np.unique(mat):
            sel = mat == m
            g[idx[sel]] *= fresnel(th[sel], etas[int(m)], pol)
    return g * ((band.wavelength / (4 * np.pi)) if fspl else 1.0) / ps.length


def synthesize(lengths, gains, freqs) -> np.ndarray:
    """H(f) = sum_k gains_k * exp(-j 2 pi f lengths_k / C0) for one receiver.

    Parameters
    ----------
    lengths : (n,) path lengths in metres
    gains : (n,) complex amplitudes
    freqs : (N,) absolute frequencies in Hz

    Returns
    -------
    (N,) complex array. Zeros if there are no paths.
    """
    tau = np.asarray(lengths, float) / C0
    return np.asarray(gains) @ np.exp(-2j * np.pi * np.outer(tau, freqs))


# ---------------------------------------------------------------------------
# 4. ChannelSet and HDF5
# ---------------------------------------------------------------------------

@dataclass
class ChannelSet:
    """Complex channels of one Tx and many receivers for one band.

    Attributes
    ----------
    H : (R, N) complex      channel per receiver and frequency; NaN rows for invalid receivers,
                            zero rows for valid receivers with no path (complete shadow)
    freqs : (N,)            absolute frequencies (Hz); index N//2 is the carrier
    rx : (R, 2)             receiver positions (m)
    rx_valid : (R,) bool    False for receivers inside obstacles or outside the room
    n_paths : (R,) int      number of paths used per receiver
    tx : (2,)               transmitter position (m)
    band : BandConfig
    meta : dict             pol, fspl, material map, source file, stage-1 settings, ...
    """

    H: np.ndarray
    freqs: np.ndarray
    rx: np.ndarray
    rx_valid: np.ndarray
    n_paths: np.ndarray
    tx: np.ndarray
    band: BandConfig
    meta: dict = field(default_factory=dict)

    def narrowband(self) -> np.ndarray:
        """(R,) complex channel at the carrier frequency (the narrowband / flat-fading channel)."""
        return self.H[:, self.band.n_sc // 2]

    def envelope(self) -> np.ndarray:
        """(R,) magnitude of the narrowband channel."""
        return np.abs(self.narrowband())

    def wideband_gain_db(self) -> np.ndarray:
        """(R,) 10*log10 of the frequency-averaged |H|^2 (-inf for shadowed receivers)."""
        with np.errstate(divide="ignore"):
            return 10 * np.log10(np.mean(np.abs(self.H) ** 2, axis=1))


def build_channels(ps: p2.PathSet, band: BandConfig, material_map: dict | None = None,
                   pol: str = "TE", fspl: bool = True, dtype=np.complex128,
                   progress: bool = False) -> ChannelSet:
    """Turn a PathSet into a ChannelSet for one band.

    Parameters
    ----------
    ps, band, material_map, pol, fspl : as in ``path_gains``
    dtype : complex128 (default) or complex64 to halve memory
    progress : print coarse progress

    Returns
    -------
    ChannelSet. Delays are applied in double precision before casting to ``dtype``.
    """
    gains = path_gains(ps, band, material_map, pol, fspl)
    freqs = band.freqs()
    R = len(ps.rx)
    H = np.full((R, band.n_sc), np.nan + 1j * np.nan, dtype=dtype)
    step = max(R // 10, 1)
    for r in range(R):
        if ps.rx_valid[r]:
            sl = ps.rx_slice(r)
            H[r] = synthesize(ps.length[sl], gains[sl], freqs)
        if progress and (r + 1) % step == 0:
            print(f"synthesized {r + 1}/{R} receivers")
    mm, names = {**DEFAULT_MATERIAL_MAP, **(material_map or {})}, ps.scene.material_names
    used = {names[m]: mm.get(names[m], names[m]) for m in np.unique(ps.scene.material)}
    meta = dict(pol=pol, fspl=fspl,
                material_map={k: list(v) if isinstance(v, tuple) else v for k, v in used.items()},
                stage1=ps.meta, gain_convention="a = lambda_c/(4 pi L) * prod(Gamma); isotropic antennas")
    return ChannelSet(H=H, freqs=freqs, rx=ps.rx.copy(), rx_valid=ps.rx_valid.copy(),
                      n_paths=ps.n_paths(), tx=ps.tx.copy(), band=band, meta=meta)


def save_channels(filename: str, ch: ChannelSet, source: str | None = None) -> None:
    """Write a ChannelSet to HDF5 (gzip level 4). Layout: /H (R,N) complex, /freqs, /rx/{xy,valid,n_paths},
    /tx/xy, root attributes format_version, created_utc, band_json, meta_json, source."""
    with h5py.File(filename, "w") as f:
        f.attrs.update(format_version=FORMAT_VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
                       band_json=json.dumps(asdict(ch.band)), meta_json=json.dumps(ch.meta, default=float),
                       source=source or "")
        f.create_dataset("H", data=ch.H, compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("freqs", data=ch.freqs)
        g = f.create_group("rx")
        g.create_dataset("xy", data=ch.rx)
        g.create_dataset("valid", data=ch.rx_valid)
        g.create_dataset("n_paths", data=ch.n_paths)
        f.create_group("tx").create_dataset("xy", data=ch.tx)


def load_channels(filename: str) -> ChannelSet:
    """Read a file written by ``save_channels``. Raises ValueError on a different major format version."""
    with h5py.File(filename, "r") as f:
        ver = str(f.attrs["format_version"])
        if ver.split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise ValueError(f"Unsupported format_version {ver} (expected {FORMAT_VERSION})")
        meta = json.loads(f.attrs["meta_json"])
        meta["source"] = str(f.attrs["source"])
        return ChannelSet(H=f["H"][()], freqs=f["freqs"][()], rx=f["rx/xy"][()],
                          rx_valid=f["rx/valid"][()].astype(bool), n_paths=f["rx/n_paths"][()],
                          tx=f["tx/xy"][()], band=BandConfig(**json.loads(f.attrs["band_json"])), meta=meta)


def make_channels(paths_file: str, band: BandConfig, out_file: str, **kw) -> ChannelSet:
    """Convenience pipeline: load stage-1 paths, build channels (kwargs go to ``build_channels``), save."""
    ch = build_channels(p2.load_pathset(paths_file), band, **kw)
    save_channels(out_file, ch, source=paths_file)
    return ch


# ---------------------------------------------------------------------------
# 5. Time domain and delay spread
# ---------------------------------------------------------------------------

def cir(H, band: BandConfig, nfft: int | None = None, window: str | None = "hann"):
    """Channel impulse response by IFFT over the frequency samples.

    Parameters
    ----------
    H : (N,) or (R, N) complex
    band : BandConfig
    nfft : output length (>= N); zero-padding in frequency interpolates in time. Default N.
    window : "hann" (low sidelobes, peak amplitude is compensated) or None

    Returns
    -------
    t : (nfft,) delays in seconds, alias-free only below 1 / band.delta_f
    h : (R, nfft) complex; a single path of amplitude a peaks at |h| = a at its delay (no window).
    """
    H = np.atleast_2d(H)
    N, nfft = band.n_sc, nfft or band.n_sc
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(N) + 0.5) / N) if window == "hann" else np.ones(N)
    Hs = np.fft.ifftshift(H * w, axes=-1)
    out = np.zeros(H.shape[:-1] + (nfft,), complex)
    out[..., :N // 2] = Hs[..., :N // 2]
    out[..., nfft - (N - N // 2):] = Hs[..., N // 2:]
    h = np.fft.ifft(out, axis=-1) * nfft / (N * w.mean())
    return np.arange(nfft) / (nfft * band.delta_f), h


def rms_delay_spread(ps: p2.PathSet, gains: np.ndarray) -> np.ndarray:
    """RMS delay spread per receiver (seconds) from the discrete paths, power-weighted by |gain|^2.
    NaN for receivers without paths. Exact for the ray model (no CIR resolution limit)."""
    out = np.full(len(ps.rx), np.nan)
    for r in range(len(ps.rx)):
        sl = ps.rx_slice(r)
        if sl.stop == sl.start:
            continue
        p, tau = np.abs(gains[sl]) ** 2, ps.length[sl] / C0
        m = (p * tau).sum() / p.sum()
        out[r] = np.sqrt((p * (tau - m) ** 2).sum() / p.sum())
    return out


# ---------------------------------------------------------------------------
# 6. Selection and normalization helpers (inputs for the analysis stage)
# ---------------------------------------------------------------------------

def row_indices(rx: np.ndarray, y: float) -> np.ndarray:
    """Indices of the receivers on the grid row nearest height y, ordered by increasing x."""
    y0 = rx[np.argmin(np.abs(rx[:, 1] - y)), 1]
    idx = np.where(np.abs(rx[:, 1] - y0) < 1e-9)[0]
    return idx[np.argsort(rx[idx, 0])]


def stride_for(spacing_m: float, step_m: float):
    """Grid stride that best realizes a physical spacing on a grid of step ``step_m``.

    Returns (stride, actual_spacing_m, relative_error). Use it with ``ula_groups``; check the error
    (a coarse grid cannot realize lambda/4 spacings exactly)."""
    stride = max(int(round(spacing_m / step_m)), 1)
    actual = stride * step_m
    return stride, actual, (actual - spacing_m) / spacing_m


def ula_groups(n_points: int, stride: int, n_elem: int, hop: int = 1) -> np.ndarray:
    """All uniform-linear-array placements along a row of ``n_points`` receivers.

    Element j of group g is row position g*hop + j*stride. Returns (G, n_elem) positions; map them to
    receiver indices with ``row_indices(...)[groups]``. ``hop`` > 1 thins the placements (fewer, less
    correlated samples)."""
    starts = np.arange(0, n_points - stride * (n_elem - 1), hop)
    return starts[:, None] + stride * np.arange(n_elem)[None, :]


def local_normalize(env: np.ndarray, win: int) -> np.ndarray:
    """Divide an envelope by its local rms over a window of ``win`` samples (per axis).

    Works on a 1D row or a 2D grid (reshape to (ny, nx) first). The output has local mean-square ~1,
    which removes slow (shadowing / path-loss) variation so what remains is small-scale fading.
    Choose ``win`` so the window spans roughly 10 to 20 wavelengths."""
    from scipy.ndimage import uniform_filter

    local = np.sqrt(uniform_filter(np.asarray(env, float) ** 2, size=win, mode="reflect"))
    return np.divide(env, local, out=np.zeros_like(local), where=local > 0)


if __name__ == "__main__":
    # usage: python channel2d.py paths.h5 band_name out.h5 [TE|TM]
    if len(sys.argv) < 4:
        raise SystemExit(f"usage: python channel2d.py paths.h5 {{{'|'.join(BANDS)}}} out.h5 [TE|TM]")
    ch = make_channels(sys.argv[1], BANDS[sys.argv[2]], sys.argv[3],
                       pol=sys.argv[4] if len(sys.argv) > 4 else "TE", progress=True)
    ok = ch.rx_valid & (ch.n_paths > 0)
    print(f"{ch.band.name}: {ch.H.shape[0]} receivers, {ok.sum()} with signal, "
          f"narrowband gain {20 * np.log10(ch.envelope()[ok].min()):.1f} to "
          f"{20 * np.log10(ch.envelope()[ok].max()):.1f} dB")
