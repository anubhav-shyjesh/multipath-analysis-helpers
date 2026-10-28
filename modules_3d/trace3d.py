"""
trace3d.py -- Stage 1 of the 3D multipath simulator: ray-traced path lists on disk.

Sections
    1. Path-file schema and HDF5 I/O (no Sionna dependency, so the analysis environment can import it).
    2. Scene description (SceneSpec) and the Sionna tracer (trace_paths3d). Sionna is imported lazily
       inside the tracer, never at module level. Tested against sionna-rt 2.2.0.

Typical use (trace once per band, then build channels with channel3d.py):

    spec = SceneSpec(room=(20, 3, 3), boxes=((8, 0.8, 0, 9, 2.2, 1.0),))
    rx, xs, ys = make_rx_grid3d((4, 4.5), (1.5, 1.5), 1.5, 0.001)
    ps = trace_paths3d(spec, tx=(2, 1.5, 1.5), rx_points=rx, fc=5.5e9, max_order=3)
    save_pathset3d("hall_5p5.h5", ps)

Tracing is Monte Carlo path discovery (shoot-and-bounce): complete for the orders tested (<= 3 on an
empty box matches the image lattice exactly) but not guaranteed at higher orders (one of 129 paths missed
at order 4 in the box test). Verify completeness for your own scene and ``samples_per_src``.

A path file holds, for one transmitter and many receiver points, every propagation path found by the
tracer (line of sight plus specular reflections; no diffraction, no diffuse scattering). Antennas are
one fixed isotropic, vertically polarized element at both ends, so each path carries ONE complex
amplitude ``a`` that already contains free-space spreading, Fresnel reflection losses and
polarization mixing, evaluated at the trace frequency ``fc_trace``.

Conventions (identical to channel2d.py so the two stages agree):
    a_k  = complex path amplitude WITHOUT the propagation phase, i.e. the link-gain amplitude
           [lambda / (4 pi L_k)] * (product of reflection factors)
    H(f) = sum_k a_k exp(-j 2 pi f L_k / C0)
    aoa  = unit vector at the receiver pointing TOWARD where the ray comes from (back along the ray)
    aod  = unit vector at the transmitter pointing along the departing ray
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import h5py
import numpy as np

C0 = 299792458.0
FORMAT_VERSION = "1.0"


@dataclass
class PathSet3D:
    """All paths from one transmitter to R receiver points, stored flat (ragged by receiver).

    Attributes
    ----------
    tx : (3,) float          transmitter position (m)
    rx : (R, 3) float        receiver positions (m)
    rx_valid : (R,) bool     False for receivers the tracer rejected (inside geometry, outside scene)
    offsets : (R+1,) int     paths of receiver r are rows offsets[r]:offsets[r+1] of the arrays below
    length : (P,) float      unfolded path length (m)
    order : (P,) int         number of reflections (0 = line of sight)
    a : (P,) complex         amplitude without propagation phase (see module docstring)
    aoa : (P, 3) float       unit arrival vector at the receiver (points toward the source of the ray)
    aod : (P, 3) float       unit departure vector at the transmitter
    fc_trace : float         frequency (Hz) at which the tracer evaluated ``a``
    meta : dict              free-form settings (scene name, max_order, tracer version, ...)
    """

    tx: np.ndarray
    rx: np.ndarray
    rx_valid: np.ndarray
    offsets: np.ndarray
    length: np.ndarray
    order: np.ndarray
    a: np.ndarray
    aoa: np.ndarray
    aod: np.ndarray
    fc_trace: float
    meta: dict = field(default_factory=dict)

    def n_paths(self) -> np.ndarray:
        """(R,) number of paths per receiver."""
        return np.diff(self.offsets)

    def rx_slice(self, r: int) -> slice:
        """Slice selecting the paths of receiver ``r`` in the flat per-path arrays."""
        return slice(int(self.offsets[r]), int(self.offsets[r + 1]))

    def delays(self) -> np.ndarray:
        """(P,) path delays in seconds."""
        return self.length / C0

    def validate(self) -> None:
        """Raise ValueError if the arrays are inconsistent (shapes, offsets, unit vectors, lengths)."""
        R, P = len(self.rx), len(self.length)
        if self.tx.shape != (3,) or self.rx.shape != (R, 3):
            raise ValueError("tx must be (3,) and rx (R, 3)")
        if self.rx_valid.shape != (R,) or self.offsets.shape != (R + 1,):
            raise ValueError("rx_valid must be (R,) and offsets (R+1,)")
        if self.offsets[0] != 0 or self.offsets[-1] != P or np.any(np.diff(self.offsets) < 0):
            raise ValueError("offsets must start at 0, be non-decreasing and end at the number of paths")
        for name, arr, shape in (("order", self.order, (P,)), ("a", self.a, (P,)),
                                 ("aoa", self.aoa, (P, 3)), ("aod", self.aod, (P, 3))):
            if arr.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
        if P and (np.any(self.length <= 0) or not np.all(np.isfinite(self.length))):
            raise ValueError("path lengths must be finite and positive")
        for name, v in (("aoa", self.aoa), ("aod", self.aod)):
            if P and np.max(np.abs(np.linalg.norm(v, axis=1) - 1)) > 1e-6:
                raise ValueError(f"{name} must be unit vectors")
        if np.any(self.n_paths()[~self.rx_valid] > 0):
            raise ValueError("invalid receivers must have no paths")
        if self.fc_trace <= 0:
            raise ValueError("fc_trace must be positive")


def save_pathset3d(filename: str, ps: PathSet3D) -> None:
    """Write a PathSet3D to HDF5 (gzip level 4) after validating it.

    Layout: /tx/xyz, /rx/{xyz,valid}, /offsets, /paths/{length,order,a,aoa,aod}; root attributes
    format_version, kind="pathset3d", created_utc, fc_trace, meta_json."""
    ps.validate()
    with h5py.File(filename, "w") as f:
        f.attrs.update(format_version=FORMAT_VERSION, kind="pathset3d", fc_trace=float(ps.fc_trace),
                       created_utc=datetime.now(timezone.utc).isoformat(),
                       meta_json=json.dumps(ps.meta, default=float))
        f.create_group("tx").create_dataset("xyz", data=ps.tx)
        g = f.create_group("rx")
        g.create_dataset("xyz", data=ps.rx)
        g.create_dataset("valid", data=ps.rx_valid)
        f.create_dataset("offsets", data=ps.offsets)
        g = f.create_group("paths")
        for name in ("length", "order", "a", "aoa", "aod"):
            g.create_dataset(name, data=getattr(ps, name), compression="gzip", compression_opts=4,
                             shuffle=True)


def load_pathset3d(filename: str) -> PathSet3D:
    """Read a file written by ``save_pathset3d`` and validate it. Raises ValueError for a file of another
    kind (e.g. a 2D path file) or a different major format version."""
    with h5py.File(filename, "r") as f:
        if str(f.attrs.get("kind", "")) != "pathset3d":
            raise ValueError(f"{filename} is not a 3D path file")
        ver = str(f.attrs["format_version"])
        if ver.split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise ValueError(f"Unsupported format_version {ver} (expected {FORMAT_VERSION})")
        ps = PathSet3D(tx=f["tx/xyz"][()], rx=f["rx/xyz"][()], rx_valid=f["rx/valid"][()].astype(bool),
                       offsets=f["offsets"][()], length=f["paths/length"][()], order=f["paths/order"][()],
                       a=f["paths/a"][()], aoa=f["paths/aoa"][()], aod=f["paths/aod"][()],
                       fc_trace=float(f.attrs["fc_trace"]), meta=json.loads(f.attrs["meta_json"]))
    ps.validate()
    return ps


# ---------------------------------------------------------------------------
# 2. Scene description and Sionna tracer
# ---------------------------------------------------------------------------

_GEO_T = 0.1       # geometric thickness of the wall slabs outside the room (m); irrelevant to the physics


@dataclass(frozen=True)
class SceneSpec:
    """A rectangular room with optional box obstacles, all surfaces specular.

    Attributes
    ----------
    room : (Lx, Ly, Lz)       interior dimensions (m); the interior is [0,Lx] x [0,Ly] x [0,Lz]
    boxes : tuple of (x0, y0, z0, x1, y1, z1)   solid axis-aligned obstacles inside the room
    room_material, box_material : ITU material names understood by Sionna (concrete, brick, plasterboard,
        wood, glass, ceiling_board, chipboard, plywood, marble, floorboard, metal, ...)
    material_thickness : float
        Slab thickness (m) used by Sionna's reflection model. Large values (default 1.0) behave as a
        half-space, which matches channel2d's Fresnel model; small values add slab interference.
    """

    room: tuple
    boxes: tuple = ()
    room_material: str = "concrete"
    box_material: str = "concrete"
    material_thickness: float = 1.0

    def __post_init__(self):
        Lx, Ly, Lz = self.room
        if min(self.room) <= 0:
            raise ValueError("room dimensions must be positive")
        for b in self.boxes:
            x0, y0, z0, x1, y1, z1 = b
            if not (0 <= x0 < x1 <= Lx and 0 <= y0 < y1 <= Ly and 0 <= z0 < z1 <= Lz):
                raise ValueError(f"box {b} must satisfy 0 <= lo < hi <= room size on every axis")

    def valid(self, pts, margin: float = 1e-4) -> np.ndarray:
        """(R,) bool: True for points inside the room and outside every box (``margin`` metres of clearance)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        ok = np.all((pts > margin) & (pts < np.asarray(self.room) - margin), axis=1)
        for x0, y0, z0, x1, y1, z1 in self.boxes:
            lo, hi = np.array([x0, y0, z0]) - margin, np.array([x1, y1, z1]) + margin
            ok &= ~np.all((pts > lo) & (pts < hi), axis=1)
        return ok

    def xml(self) -> str:
        """Mitsuba scene XML: six wall slabs around the interior plus the boxes (one cube each)."""
        Lx, Ly, Lz = self.room
        T = _GEO_T
        slabs = [("floor", (-T, -T, -T), (Lx + T, Ly + T, 0), self.room_material),
                 ("ceiling", (-T, -T, Lz), (Lx + T, Ly + T, Lz + T), self.room_material),
                 ("wall_y0", (-T, -T, 0), (Lx + T, 0, Lz), self.room_material),
                 ("wall_y1", (-T, Ly, 0), (Lx + T, Ly + T, Lz), self.room_material),
                 ("wall_x0", (-T, 0, 0), (0, Ly, Lz), self.room_material),
                 ("wall_x1", (Lx, 0, 0), (Lx + T, Ly, Lz), self.room_material)]
        slabs += [(f"box{i}", b[:3], b[3:], self.box_material) for i, b in enumerate(self.boxes)]
        out = ['<scene version="2.1.0">']
        for m in sorted({s[3] for s in slabs}):
            out.append(f'<bsdf type="itu-radio-material" id="mat-itu_{m}"><string name="type" value="{m}"/>'
                       f'<float name="thickness" value="{self.material_thickness}"/></bsdf>')
        for name, lo, hi, m in slabs:
            c = [(l + h) / 2 for l, h in zip(lo, hi)]
            s = [(h - l) / 2 for l, h in zip(lo, hi)]
            out.append(f'<shape type="cube" id="{name}"><transform name="to_world">'
                       f'<scale x="{s[0]}" y="{s[1]}" z="{s[2]}"/><translate x="{c[0]}" y="{c[1]}" z="{c[2]}"/>'
                       f'</transform><ref id="mat-itu_{m}" name="bsdf"/></shape>')
        out.append("</scene>")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return dict(room=list(self.room), boxes=[list(b) for b in self.boxes], room_material=self.room_material,
                    box_material=self.box_material, material_thickness=self.material_thickness)


def make_rx_grid3d(x_range, y_range, z, step):
    """Horizontal receiver grid at height z. Returns (points (R, 3), xs, ys); points are ordered row by row
    (y outer loop, x inner loop), the same ordering as paths2d.make_rx_grid."""
    xs = np.arange(x_range[0], x_range[1] + step / 2, step)
    ys = np.arange(y_range[0], y_range[1] + step / 2, step)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.full(X.size, float(z))], xs, ys


def _import_sionna(variant: str | None = None):
    """Import sionna.rt lazily, choosing the Mitsuba variant first: ``variant`` if given, else the CUDA
    variant when the GPU works and the LLVM (CPU) variant otherwise."""
    import mitsuba as mi

    if variant is None:
        try:
            mi.set_variant("cuda_ad_mono_polarized")
            mi.Float(1.0)                        # forces the CUDA driver to load; fails without a usable GPU
            variant = "cuda_ad_mono_polarized"
        except Exception:
            variant = "llvm_ad_mono_polarized"
    mi.set_variant(variant)
    import sionna.rt as rt
    return rt, variant


def trace_paths3d(spec: SceneSpec, tx, rx_points, fc: float, max_order: int = 3,
                  samples_per_src: int = 1_000_000, batch: int = 200, variant: str | None = None,
                  progress: bool = False, meta: dict | None = None) -> PathSet3D:
    """Trace line-of-sight and specular paths from ``tx`` to every receiver point with Sionna RT.

    Settings fixed by design: one isotropic vertically polarized antenna at both ends; diffraction,
    diffuse scattering and refraction (transmission through walls) are OFF, so obstacles cast hard shadows.

    Parameters
    ----------
    spec : SceneSpec
    tx : (3,) transmitter position (must be inside the room, outside the boxes)
    rx_points : (R, 3) receiver positions. Points outside the room or inside a box get rx_valid False and no paths.
    fc : carrier frequency in Hz at which materials are evaluated (trace once per band)
    max_order : maximum number of reflections (Sionna ``max_depth``)
    samples_per_src : rays Sionna shoots from the transmitter to discover specular paths. Higher orders and
        larger rooms need more; check completeness against an analytic case (see test_trace3d.py).
    batch : receivers per Sionna call (cost is dominated by the shooting, so larger is faster, but memory grows)
    variant : Mitsuba variant, e.g. "llvm_ad_mono_polarized" to force the CPU; default picks CUDA if it works
    progress : print coarse progress
    meta : extra metadata stored in the file

    Returns
    -------
    PathSet3D with ``a`` de-phased (the propagation phase exp(-j 2 pi fc tau) Sionna includes is removed so
    the file follows the module convention), paths sorted by length within each receiver.
    """
    tx = np.asarray(tx, float)
    rx = np.atleast_2d(np.asarray(rx_points, float))
    if tx.shape != (3,) or rx.shape[1] != 3:
        raise ValueError("tx must be (3,) and rx_points (R, 3)")
    if not spec.valid(tx)[0]:
        raise ValueError("Transmitter is outside the room or inside an obstacle")
    rt, used_variant = _import_sionna(variant)

    scene = rt.load_scene_from_string(spec.xml())
    scene.frequency = fc
    arr = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.tx_array, scene.rx_array = arr, arr
    scene.add(rt.Transmitter("tx", position=tx.tolist()))
    solver = rt.PathSolver()

    R = len(rx)
    valid = spec.valid(rx)
    todo = np.where(valid)[0]
    per_rx = [None] * R
    for s in range(0, len(todo), batch):
        ids = todo[s:s + batch]
        for j, i in enumerate(ids):
            scene.add(rt.Receiver(f"rx{j}", position=rx[i].tolist()))
        paths = solver(scene, max_depth=max_order, samples_per_src=samples_per_src, los=True,
                       specular_reflection=True, diffuse_reflection=False, refraction=False,
                       diffraction=False, edge_diffraction=False, synthetic_array=True)
        a, tau = paths.cir(normalize_delays=False, out_type="numpy")        # (B,1,1,1,P,1), (B,1,P)
        a = np.asarray(a)[:, 0, 0, 0, :, 0]
        tau = np.asarray(tau, np.float64)[:, 0, :]
        ok = np.asarray(paths.valid)[:, 0, :].astype(bool)
        inter = np.asarray(paths.interactions)[:, :, 0, :]                  # (depth, B, P)
        order = (inter != 0).sum(axis=0)
        ang = {n: np.asarray(getattr(paths, n), np.float64)[:, 0, :] for n in ("theta_r", "phi_r", "theta_t", "phi_t")}
        for j, i in enumerate(ids):
            k = np.where(ok[j])[0]
            k = k[np.argsort(tau[j, k])]
            t = tau[j, k]
            vec = lambda th, ph: np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
            aoa = vec(ang["theta_r"][j, k], ang["phi_r"][j, k])
            aod = vec(ang["theta_t"][j, k], ang["phi_t"][j, k])
            per_rx[i] = (t * C0, order[j, k], a[j, k].astype(np.complex128) * np.exp(2j * np.pi * fc * t),
                         aoa / np.linalg.norm(aoa, axis=1, keepdims=True),
                         aod / np.linalg.norm(aod, axis=1, keepdims=True))
        for j in range(len(ids)):
            scene.remove(f"rx{j}")
        if progress:
            print(f"traced {min(s + batch, len(todo))}/{len(todo)} valid receivers")

    empty = (np.zeros(0), np.zeros(0, int), np.zeros(0, complex), np.zeros((0, 3)), np.zeros((0, 3)))
    parts = [p if p is not None else empty for p in per_rx]
    import importlib.metadata as md
    info = dict(spec=spec.to_dict(), max_order=max_order, samples_per_src=samples_per_src, variant=used_variant,
                sionna_rt=md.version("sionna-rt"), los=True, specular=True, diffuse=False, diffraction=False,
                refraction=False, antenna="iso, vertical polarization, same at Tx and Rx")
    info.update(meta or {})
    return PathSet3D(
        tx=tx, rx=rx, rx_valid=valid, offsets=np.concatenate([[0], np.cumsum([len(p[0]) for p in parts])]).astype(int),
        length=np.concatenate([p[0] for p in parts]), order=np.concatenate([p[1] for p in parts]).astype(int),
        a=np.concatenate([p[2] for p in parts]), aoa=np.concatenate([p[3] for p in parts]).reshape(-1, 3),
        aod=np.concatenate([p[4] for p in parts]).reshape(-1, 3), fc_trace=float(fc), meta=info)