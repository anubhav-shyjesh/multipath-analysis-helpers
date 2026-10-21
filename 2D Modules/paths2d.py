"""
paths2d.py -- Stage 1 of the 2D multipath simulator: geometry -> path lists -> HDF5.

This module is frequency-independent. It finds every specular propagation path
(direct + wall bounces up to a chosen order) between one transmitter and many
receiver points inside a 2D floor plan, and stores the geometric description of
each path. Band, bandwidth and reflection coefficients are applied later
(stage 2), so you trace once and synthesize channels for any band.

Conventions
-----------
* Units: metres and radians. Angles are measured from the +x axis, CCW positive.
* Walls are line segments; every wall is two-sided and opaque (no transmission).
* Wall ids are indices into Scene arrays, in the order the walls were added.
* A path's wall sequence and bounce points are stored in propagation order
  (index 0 is the bounce nearest the Tx).
* Padded entries are -1 (wall ids) or NaN (angles, points).

Sections
--------
1. Scene / geometry builders
2. Image computation
3. Path tracing
4. Receiver-position helpers
5. HDF5 writer / reader
6. Plot helper and demo
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import h5py
import numpy as np

C0 = 299_792_458.0          # speed of light, m/s
FORMAT_VERSION = "1.0"      # bump when the HDF5 layout changes
U_TOL = 1e-9                # tolerance on the along-wall parameter (0..1)


# ---------------------------------------------------------------------------
# 1. Scene / geometry
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """A 2D scene made of wall segments.

    Attributes
    ----------
    p0, p1 : (N, 2) float arrays
        Start and end point of each wall segment.
    material : (N,) int array
        Index into ``material_names`` for each wall.
    group : (N,) int array
        Group id of each wall. A group is one room outline, one obstacle
        polygon, or one free-standing wall.
    group_kind : list[str]
        ``"room"``, ``"obstacle"`` or ``"wall"`` for each group id. Used by
        ``free_space_mask``.
    material_names : list[str]
        Human-readable material labels (stage 2 maps these to permittivities).
    """

    p0: np.ndarray
    p1: np.ndarray
    material: np.ndarray
    group: np.ndarray
    group_kind: list
    material_names: list = field(default_factory=lambda: ["wall"])

    @property
    def n_walls(self) -> int:
        """Number of wall segments."""
        return len(self.p0)

    def normals(self) -> np.ndarray:
        """Return unit normals, shape (N, 2): the segment direction rotated +90 deg."""
        d = self.p1 - self.p0
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def lengths(self) -> np.ndarray:
        """Return the length of each wall segment, shape (N,)."""
        return np.linalg.norm(self.p1 - self.p0, axis=1)


def _append(scene: Scene, p0, p1, material: int, kind: str) -> Scene:
    """Return a new Scene with segments (p0[i] -> p1[i]) appended as one new group."""
    p0 = np.atleast_2d(np.asarray(p0, float))
    p1 = np.atleast_2d(np.asarray(p1, float))
    gid = len(scene.group_kind)
    return Scene(
        p0=np.vstack([scene.p0, p0]),
        p1=np.vstack([scene.p1, p1]),
        material=np.concatenate([scene.material, np.full(len(p0), material, int)]),
        group=np.concatenate([scene.group, np.full(len(p0), gid, int)]),
        group_kind=list(scene.group_kind) + [kind],
        material_names=list(scene.material_names),
    )


def _empty_scene(material_names) -> Scene:
    return Scene(
        p0=np.zeros((0, 2)), p1=np.zeros((0, 2)),
        material=np.zeros(0, int), group=np.zeros(0, int),
        group_kind=[], material_names=list(material_names),
    )


def rectangle_room(Lx: float, Ly: float, material: int = 0,
                   material_names=("wall",)) -> Scene:
    """Create a rectangular room (or hallway) with its corner at the origin.

    Parameters
    ----------
    Lx, Ly : float
        Room extent along x and y in metres (a hallway is just a long thin room).
    material : int
        Material index applied to all four walls.
    material_names : sequence of str
        Initial material label list.

    Returns
    -------
    Scene
        Wall ids: 0 = bottom (y=0), 1 = right (x=Lx), 2 = top (y=Ly), 3 = left (x=0).
        Group 0 has kind ``"room"``.
    """
    corners = np.array([[0, 0], [Lx, 0], [Lx, Ly], [0, Ly]], float)
    return _append(_empty_scene(material_names), corners, np.roll(corners, -1, axis=0),
                   material, "room")


def add_polygon(scene: Scene, vertices, material: int = 0, kind: str = "obstacle") -> Scene:
    """Add a closed polygon (obstacle) to a scene.

    Parameters
    ----------
    scene : Scene
    vertices : (V, 2) array-like
        Polygon corners in order; the last vertex connects back to the first.
    material : int
        Material index for all edges.
    kind : str
        ``"obstacle"`` (default) makes the interior solid so receivers inside
        it are masked out by ``free_space_mask``.

    Returns
    -------
    Scene
        A new scene (the input is not modified).
    """
    v = np.asarray(vertices, float)
    return _append(scene, v, np.roll(v, -1, axis=0), material, kind)


def add_box(scene: Scene, xmin: float, ymin: float, xmax: float, ymax: float,
            material: int = 0) -> Scene:
    """Add an axis-aligned rectangular obstacle. Returns a new Scene."""
    return add_polygon(scene, [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                       material, "obstacle")


def add_wall(scene: Scene, a, b, material: int = 0) -> Scene:
    """Add a single free-standing wall segment from ``a`` to ``b`` (a partition or screen).

    Returns a new Scene. Free-standing walls reflect and block from both sides
    and have no interior, so they do not affect ``free_space_mask``.
    """
    return _append(scene, np.asarray(a, float)[None], np.asarray(b, float)[None],
                   material, "wall")


def _inside_group(scene: Scene, gid: int, pts: np.ndarray) -> np.ndarray:
    """Even-odd ray-casting test: is each point inside the closed outline of group ``gid``?"""
    sel = scene.group == gid
    x0, y0 = scene.p0[sel, 0][None, :], scene.p0[sel, 1][None, :]
    x1, y1 = scene.p1[sel, 0][None, :], scene.p1[sel, 1][None, :]
    px, py = pts[:, 0:1], pts[:, 1:2]
    straddle = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    crossings = straddle & (px < xint)
    return (crossings.sum(axis=1) % 2) == 1


def free_space_mask(scene: Scene, pts) -> np.ndarray:
    """Return a boolean mask: True where a point is a legal Tx/Rx location.

    A point is legal if it lies inside every ``"room"`` group and outside every
    ``"obstacle"`` group. Points exactly on a wall are not reliably classified.

    Parameters
    ----------
    scene : Scene
    pts : (R, 2) array-like

    Returns
    -------
    (R,) bool array
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    ok = np.ones(len(pts), bool)
    for gid, kind in enumerate(scene.group_kind):
        if kind == "room":
            ok &= _inside_group(scene, gid, pts)
        elif kind == "obstacle":
            ok &= ~_inside_group(scene, gid, pts)
    return ok


# ---------------------------------------------------------------------------
# 2. Images
# ---------------------------------------------------------------------------

def mirror_points(pts, a, b) -> np.ndarray:
    """Mirror points across the infinite line through ``a`` and ``b``.

    Parameters
    ----------
    pts : (M, 2) or (2,) array-like
    a, b : (2,) array-like
        Two distinct points on the mirror line.

    Returns
    -------
    (M, 2) array of mirrored points.
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    a = np.asarray(a, float)
    d = np.asarray(b, float) - a
    n = np.array([-d[1], d[0]]) / np.hypot(d[0], d[1])
    dist = (pts - a) @ n
    return pts - 2.0 * dist[:, None] * n


@dataclass
class ImageSet:
    """All image sources of one transmitter, up to ``max_order`` reflections.

    Attributes
    ----------
    pos : (M, 2) float
        Image positions. Row 0 is the real source (order 0).
    order : (M,) int
        Number of reflections that created each image.
    walls : (M, max_order) int
        Wall sequence in propagation order (first wall hit after the Tx first),
        padded with -1.
    parent : (M,) int
        Index of the image one reflection earlier (-1 for the real source).
    """

    pos: np.ndarray
    order: np.ndarray
    walls: np.ndarray
    parent: np.ndarray


def compute_images(scene: Scene, tx, max_order: int) -> ImageSet:
    """Build every image source of ``tx`` up to ``max_order`` reflections.

    Each image of order n is the mirror of an order n-1 image across one wall,
    excluding the wall that produced the previous reflection (mirroring twice
    across the same wall returns to the previous image). Validity (does the
    path physically exist?) is NOT checked here; that depends on the receiver
    and is done in ``trace_paths``. The image set depends only on the Tx and
    the walls, so it is computed once per Tx.

    Cost grows as N * (N-1)^(order-1) for N walls, so keep ``max_order`` modest
    for scenes with many segments.

    Parameters
    ----------
    scene : Scene
    tx : (2,) array-like
    max_order : int
        Maximum number of reflections (0 = direct path only).

    Returns
    -------
    ImageSet
    """
    tx = np.asarray(tx, float)
    K = max_order
    pos = [tx[None, :]]
    walls = [np.full((1, K), -1, np.int32)]
    parent = [np.array([-1])]
    order = [np.zeros(1, int)]
    count = 1
    prev_pos, prev_walls, prev_start = pos[0], walls[0], 0

    for n in range(1, K + 1):
        new_pos, new_walls, new_par = [], [], []
        for w in range(scene.n_walls):
            if n > 1:
                sel = np.where(prev_walls[:, n - 2] != w)[0]
            else:
                sel = np.arange(len(prev_pos))
            if sel.size == 0:
                continue
            new_pos.append(mirror_points(prev_pos[sel], scene.p0[w], scene.p1[w]))
            nw = prev_walls[sel].copy()
            nw[:, n - 1] = w
            new_walls.append(nw)
            new_par.append(prev_start + sel)
        if not new_pos:
            break
        cur_pos = np.vstack(new_pos)
        cur_walls = np.vstack(new_walls)
        pos.append(cur_pos)
        walls.append(cur_walls)
        parent.append(np.concatenate(new_par))
        order.append(np.full(len(cur_pos), n, int))
        prev_pos, prev_walls, prev_start = cur_pos, cur_walls, count
        count += len(cur_pos)

    return ImageSet(pos=np.vstack(pos), order=np.concatenate(order),
                    walls=np.vstack(walls), parent=np.concatenate(parent))


# ---------------------------------------------------------------------------
# 3. Path tracing
# ---------------------------------------------------------------------------

def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D cross product a x b along the last axis."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _seg_intersect(a, b, c, d):
    """Intersect segments a->b and c->d (broadcasting over leading axes).

    Returns
    -------
    t : parameter along a->b of the intersection point (0..1 inside the segment)
    u : parameter along c->d of the intersection point (0..1 inside the segment)
    parallel : bool, True where the segments are (nearly) parallel; t, u are NaN there.
    """
    r, s = b - a, d - c
    denom = _cross2(r, s)
    scale = np.linalg.norm(r, axis=-1) * np.linalg.norm(s, axis=-1)
    parallel = np.abs(denom) <= 1e-12 * scale
    ac = c - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross2(ac, s) / denom
        u = _cross2(ac, r) / denom
    t = np.where(parallel, np.nan, t)
    u = np.where(parallel, np.nan, u)
    return t, u, parallel


def _leg_blocked(a, b, scene: Scene, skip_a, skip_b, eps_len: float) -> np.ndarray:
    """Test whether straight legs a->b are blocked by any wall.

    Parameters
    ----------
    a, b : (m, 2) leg endpoints
    skip_a, skip_b : (m,) int
        Wall ids that the leg starts / ends on (ignored in the test); -1 for none.
    eps_len : float
        Intersections within this distance (m) of a leg endpoint do not count.

    Returns
    -------
    (m,) bool, True if the leg is blocked.
    """
    t, u, par = _seg_intersect(a[:, None, :], b[:, None, :], scene.p0[None], scene.p1[None])
    length = np.maximum(np.linalg.norm(b - a, axis=1), 1e-12)
    tol = (eps_len / length)[:, None]
    with np.errstate(invalid="ignore"):
        hit = (~par) & (t > tol) & (t < 1 - tol) & (u >= -U_TOL) & (u <= 1 + U_TOL)
    ids = np.arange(scene.n_walls)[None, :]
    hit &= (ids != skip_a[:, None]) & (ids != skip_b[:, None])
    return hit.any(axis=1)


def _empty_paths(K: int) -> dict:
    return dict(length=np.zeros(0), order=np.zeros(0, np.int32),
                walls=np.zeros((0, K), np.int32), inc_angle=np.zeros((0, K)),
                points=np.zeros((0, K, 2)), aoa=np.zeros(0), aod=np.zeros(0))


def _trace_one(scene, normals, imgs: ImageSet, tx, rx, K, max_length, eps_len) -> dict:
    """Find all valid paths from ``tx`` to one receiver ``rx`` (internal).

    For each image of order n: draw the line from the Rx to the image, find
    where it crosses the last wall of the sequence, and repeat from that
    crossing toward the previous image, until the real source is reached.
    A path is valid if every crossing lies within its wall segment and no leg
    is blocked by another wall.
    """
    chunks = []
    for n in range(0, K + 1):
        idx = np.where(imgs.order == n)[0]
        m = idx.size
        if m == 0:
            continue
        alive = np.ones(m, bool)
        P = np.zeros((m, n, 2))
        cur = np.tile(rx, (m, 1))
        lev = idx.copy()
        for k in range(n, 0, -1):
            w = imgs.walls[idx, k - 1]
            I = imgs.pos[lev]
            t, u, par = _seg_intersect(cur, I, scene.p0[w], scene.p1[w])
            with np.errstate(invalid="ignore"):
                ok = (~par) & (t > 0) & (t < 1) & (u >= -U_TOL) & (u <= 1 + U_TOL)
            alive &= ok
            Pk = np.where(ok[:, None], cur + t[:, None] * (I - cur), cur)
            P[:, k - 1] = Pk
            cur = Pk
            lev = imgs.parent[lev]

        chain = np.concatenate([np.tile(tx, (m, 1, 1)), P, np.tile(rx, (m, 1, 1))], axis=1)
        wall_seq = imgs.walls[idx]
        sel = np.where(alive)[0]
        for j in range(n + 1):
            if sel.size == 0:
                break
            sa = wall_seq[sel, j - 1] if j >= 1 else np.full(sel.size, -1)
            sb = wall_seq[sel, j] if j < n else np.full(sel.size, -1)
            blocked = _leg_blocked(chain[sel, j], chain[sel, j + 1], scene, sa, sb, eps_len)
            sel = sel[~blocked]
        if sel.size == 0:
            continue

        ch = chain[sel]
        legs = np.diff(ch, axis=1)
        leglen = np.linalg.norm(legs, axis=2)
        length = leglen.sum(axis=1)
        keep = np.ones(len(sel), bool) if max_length is None else length <= max_length
        if not keep.any():
            continue
        sel, ch, legs, leglen, length = sel[keep], ch[keep], legs[keep], leglen[keep], length[keep]
        q = len(sel)

        walls_out = np.full((q, K), -1, np.int32)
        walls_out[:, :n] = wall_seq[sel, :n]
        inc = np.full((q, K), np.nan)
        pts = np.full((q, K, 2), np.nan)
        for k in range(1, n + 1):
            dhat = legs[:, k - 1] / leglen[:, k - 1, None]
            nrm = normals[walls_out[:, k - 1]]
            inc[:, k - 1] = np.arccos(np.clip(np.abs((dhat * nrm).sum(axis=1)), 0.0, 1.0))
            pts[:, k - 1] = ch[:, k]
        aod = np.arctan2(legs[:, 0, 1], legs[:, 0, 0])
        back = -legs[:, n]
        aoa = np.arctan2(back[:, 1], back[:, 0])
        chunks.append(dict(length=length, order=np.full(q, n, np.int32), walls=walls_out,
                           inc_angle=inc, points=pts, aoa=aoa, aod=aod))

    if not chunks:
        return _empty_paths(K)
    out = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    srt = np.argsort(out["length"], kind="stable")
    return {key: val[srt] for key, val in out.items()}


@dataclass
class PathSet:
    """Flat, ragged storage of all paths for one Tx and many receivers.

    Paths of receiver ``r`` occupy rows ``offsets[r]:offsets[r+1]`` of every
    per-path array, sorted by increasing length (first arrival first).

    Attributes
    ----------
    scene : Scene
    tx : (2,) float
    rx : (R, 2) float          receiver positions (m)
    rx_valid : (R,) bool       False for receivers in obstacles / outside the room (no paths)
    offsets : (R+1,) int64     CSR-style row pointers
    length : (P,) float        total path length (m); delay = length / C0
    order : (P,) int32         number of reflections
    walls : (P, K) int32       wall ids in propagation order, -1 padded
    inc_angle : (P, K) float   angle between incoming ray and wall normal at each bounce (rad, 0..pi/2)
    points : (P, K, 2) float   bounce coordinates in propagation order, NaN padded
    aoa : (P,) float           direction the Rx looks to see the arriving ray (rad)
    aod : (P,) float           direction the ray leaves the Tx (rad)
    meta : dict                run settings (max_order, max_length, eps_len, ...)
    """

    scene: Scene
    tx: np.ndarray
    rx: np.ndarray
    rx_valid: np.ndarray
    offsets: np.ndarray
    length: np.ndarray
    order: np.ndarray
    walls: np.ndarray
    inc_angle: np.ndarray
    points: np.ndarray
    aoa: np.ndarray
    aod: np.ndarray
    meta: dict = field(default_factory=dict)

    def n_paths(self) -> np.ndarray:
        """Number of paths per receiver, shape (R,)."""
        return np.diff(self.offsets)

    def rx_slice(self, r: int) -> slice:
        """Row slice into the per-path arrays for receiver ``r``."""
        return slice(int(self.offsets[r]), int(self.offsets[r + 1]))

    def delays(self) -> np.ndarray:
        """Propagation delay of every path in seconds, shape (P,)."""
        return self.length / C0


def trace_paths(scene: Scene, tx, rx_points, max_order: int,
                max_length: float | None = None, eps_len: float = 1e-9,
                progress: bool = False, meta: dict | None = None) -> PathSet:
    """Find all valid specular paths from one Tx to every receiver point.

    Parameters
    ----------
    scene : Scene
    tx : (2,) array-like
        Transmitter position (m). Must be in free space.
    rx_points : (R, 2) array-like
        Receiver positions (m). Points in obstacles or outside the room are
        kept in the output but flagged ``rx_valid=False`` and given no paths.
    max_order : int
        Maximum reflections. Contributions beyond order 4-6 are usually negligible.
    max_length : float or None
        Drop paths longer than this many metres (geometric pruning). None keeps all.
    eps_len : float
        Length tolerance (m) for occlusion tests near leg endpoints.
    progress : bool
        Print coarse progress.
    meta : dict or None
        Extra settings (band plans, seed, scene name, ...) stored alongside the paths.

    Returns
    -------
    PathSet
    """
    tx = np.asarray(tx, float)
    rx_points = np.atleast_2d(np.asarray(rx_points, float))
    if not free_space_mask(scene, tx[None])[0]:
        raise ValueError("Transmitter is inside an obstacle or outside the room.")

    imgs = compute_images(scene, tx, max_order)
    normals = scene.normals()
    valid = free_space_mask(scene, rx_points)

    R = len(rx_points)
    per_rx, counts = [], np.zeros(R, np.int64)
    step = max(R // 10, 1)
    for r in range(R):
        if valid[r]:
            res = _trace_one(scene, normals, imgs, tx, rx_points[r], max_order, max_length, eps_len)
        else:
            res = _empty_paths(max_order)
        per_rx.append(res)
        counts[r] = len(res["length"])
        if progress and (r + 1) % step == 0:
            print(f"traced {r + 1}/{R} receivers")

    offsets = np.concatenate([[0], np.cumsum(counts)])
    cat = {k: np.concatenate([p[k] for p in per_rx]) for k in per_rx[0]}
    info = dict(max_order=max_order, max_length=max_length, eps_len=eps_len)
    info.update(meta or {})
    return PathSet(scene=scene, tx=tx, rx=rx_points, rx_valid=valid, offsets=offsets,
                   meta=info, **cat)


# ---------------------------------------------------------------------------
# 4. Receiver-position helpers
# ---------------------------------------------------------------------------

def make_rx_grid(xlim, ylim, step: float):
    """Make a regular receiver grid in metres.

    Parameters
    ----------
    xlim, ylim : (min, max) tuples
    step : float
        Grid spacing in metres (use a metre-based spacing so one trace serves
        every band; pick a subset in stage 2 for a lambda/4 grid).

    Returns
    -------
    pts : (ny*nx, 2) array, row-major (x varies fastest); reshape to (ny, nx, 2).
    xs, ys : 1D coordinate vectors.
    """
    xs = np.arange(xlim[0], xlim[1] + 0.5 * step, step)
    ys = np.arange(ylim[0], ylim[1] + 0.5 * step, step)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], axis=1), xs, ys


def make_ula(center, spacing: float, n: int, angle: float = 0.0) -> np.ndarray:
    """Make a uniform linear array of ``n`` receiver points.

    Parameters
    ----------
    center : (2,) array-like   array centre (m)
    spacing : float            element spacing (m)
    n : int                    number of elements
    angle : float              array axis direction from +x (rad)

    Returns
    -------
    (n, 2) array of element positions.
    """
    offs = (np.arange(n) - (n - 1) / 2.0) * spacing
    d = np.array([np.cos(angle), np.sin(angle)])
    return np.asarray(center, float)[None, :] + offs[:, None] * d[None, :]


# ---------------------------------------------------------------------------
# 5. HDF5 writer / reader
# ---------------------------------------------------------------------------
# Layout:
#   /                attrs: format_version, created_utc, c0, config_json
#   /scene           p0, p1, material, group;  attrs: group_kind_json, material_names_json
#   /tx              xy
#   /rx              xy, valid, offsets
#   /paths           length, order, walls, inc_angle, points, aoa, aod

def _ds(group: h5py.Group, name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.size > 0:
        group.create_dataset(name, data=arr, compression="gzip", compression_opts=4, shuffle=True)
    else:
        group.create_dataset(name, data=arr)


def save_pathset(filename: str, ps: PathSet, config: dict | None = None) -> None:
    """Write a PathSet (with its scene and settings) to an HDF5 file.

    Parameters
    ----------
    filename : str
        Output path. An existing file is overwritten.
    ps : PathSet
    config : dict or None
        Extra JSON-serializable settings to record (scene name, seed, code version, ...).
        Stored together with ``ps.meta`` under the ``config_json`` attribute.
    """
    cfg = dict(ps.meta)
    cfg.update(config or {})
    with h5py.File(filename, "w") as f:
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["c0"] = C0
        f.attrs["config_json"] = json.dumps(cfg, default=float)

        g = f.create_group("scene")
        for name in ("p0", "p1", "material", "group"):
            _ds(g, name, getattr(ps.scene, name))
        g.attrs["group_kind_json"] = json.dumps(ps.scene.group_kind)
        g.attrs["material_names_json"] = json.dumps(ps.scene.material_names)

        f.create_group("tx").create_dataset("xy", data=ps.tx)
        g = f.create_group("rx")
        _ds(g, "xy", ps.rx)
        _ds(g, "valid", ps.rx_valid)
        _ds(g, "offsets", ps.offsets)

        g = f.create_group("paths")
        for name in ("length", "order", "walls", "inc_angle", "points", "aoa", "aod"):
            _ds(g, name, getattr(ps, name))


def load_pathset(filename: str) -> PathSet:
    """Read a PathSet written by ``save_pathset``.

    Raises
    ------
    ValueError if the file's ``format_version`` major number differs from this module's.
    """
    with h5py.File(filename, "r") as f:
        ver = str(f.attrs["format_version"])
        if ver.split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise ValueError(f"Unsupported format_version {ver} (expected {FORMAT_VERSION})")
        cfg = json.loads(f.attrs["config_json"])
        g = f["scene"]
        scene = Scene(
            p0=g["p0"][()], p1=g["p1"][()], material=g["material"][()], group=g["group"][()],
            group_kind=json.loads(g.attrs["group_kind_json"]),
            material_names=json.loads(g.attrs["material_names_json"]),
        )
        paths = {k: f["paths"][k][()] for k in
                 ("length", "order", "walls", "inc_angle", "points", "aoa", "aod")}
        return PathSet(scene=scene, tx=f["tx/xy"][()], rx=f["rx/xy"][()],
                       rx_valid=f["rx/valid"][()].astype(bool), offsets=f["rx/offsets"][()],
                       meta=cfg, **paths)


# ---------------------------------------------------------------------------
# 6. Plot helper and demo
# ---------------------------------------------------------------------------

def plot_paths(ps: PathSet, rx_index: int, ax=None):
    """Draw the scene, the Tx, one Rx, and all of its paths (for eyeballing).

    Parameters
    ----------
    ps : PathSet
    rx_index : int
        Which receiver to draw.
    ax : matplotlib Axes or None

    Returns
    -------
    The Axes used.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(8, 5))[1]
    for a, b in zip(ps.scene.p0, ps.scene.p1):
        ax.plot([a[0], b[0]], [a[1], b[1]], "k-", lw=2)
    rx = ps.rx[rx_index]
    sl = ps.rx_slice(rx_index)
    for i in range(sl.start, sl.stop):
        n = int(ps.order[i])
        chain = np.vstack([ps.tx, ps.points[i, :n], rx])
        ax.plot(chain[:, 0], chain[:, 1], lw=1, alpha=0.8 if n == 0 else 0.5,
                color="tab:red" if n == 0 else "tab:blue")
    ax.plot(*ps.tx, "g^", ms=10, label="Tx")
    ax.plot(*rx, "ro", ms=8, label="Rx")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"{sl.stop - sl.start} paths (red = direct)")
    return ax


if __name__ == "__main__":
    # Demo: 20 m x 3 m hallway with one box obstacle, 5 cm receiver grid.
    sc = add_box(rectangle_room(20.0, 3.0), 8.0, 0.8, 9.0, 2.2)
    pts, xs, ys = make_rx_grid((10.0, 12.0), (0.2, 2.8), 0.05)
    ps = trace_paths(sc, (2.0, 1.5), pts, max_order=4, progress=True,
                     meta={"scene": "hallway_with_box"})
    save_pathset("demo_paths.h5", ps)
    print("paths per Rx: min", ps.n_paths().min(), "max", ps.n_paths().max())
    ps2 = load_pathset("demo_paths.h5")
    print("round trip ok:", np.array_equal(ps.offsets, ps2.offsets))
