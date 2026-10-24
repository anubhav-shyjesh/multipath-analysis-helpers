"""Tests for trace3d: scene description (no Sionna) and the Sionna tracer (skipped if sionna-rt is absent).
Set TRACE3D_VARIANT to force a Mitsuba variant, e.g. llvm_ad_mono_polarized."""

import itertools
import os

import numpy as np
import pytest

import channel2d as c2
import channel3d as c3
import trace3d as t3

C0 = t3.C0
FC = 5.5e9
LAM = C0 / FC
VARIANT = os.environ.get("TRACE3D_VARIANT")


# ---------------- scene description (no Sionna) ----------------

def test_spec_valid_and_validation():
    spec = t3.SceneSpec((8, 5, 3), boxes=((3, 1, 0, 4, 2, 1),))
    pts = np.array([[1, 1, 1], [3.5, 1.5, 0.5], [9, 1, 1], [1, 1, 3.5], [3.5, 1.5, 1.5]])
    assert list(spec.valid(pts)) == [True, False, False, False, True]      # above the box is free
    with pytest.raises(ValueError):
        t3.SceneSpec((8, 5, 3), boxes=((7, 1, 0, 9, 2, 1),))
    assert spec.xml().count("<shape") == 7


def test_rx_grid3d():
    p, xs, ys = t3.make_rx_grid3d((1, 2), (0, 1), 1.5, 0.5)
    assert p.shape == (9, 3) and np.all(p[:, 2] == 1.5) and len(xs) == 3 and len(ys) == 3


# ---------------- Sionna tracer ----------------

def trace(*a, **k):
    pytest.importorskip("sionna.rt")
    return t3.trace_paths3d(*a, variant=VARIANT, **k)


def lattice_lengths(tx, rx, L, N):
    out = []
    for k in itertools.product(range(-N, N + 1), repeat=3):
        if sum(map(abs, k)) <= N:
            img = [k[i] * L[i] + (tx[i] if k[i] % 2 == 0 else L[i] - tx[i]) for i in range(3)]
            out.append(np.linalg.norm(np.array(img) - rx))
    return np.sort(out)


def test_empty_box_matches_image_lattice():
    spec = t3.SceneSpec((8.0, 5.0, 3.0))
    tx = np.array([1.5, 1.2, 1.6])
    rx = np.array([[5.0, 3.0, 1.1], [2.2, 4.1, 2.0], [7.0, 0.8, 0.7]])
    ps = trace(spec, tx, rx, FC, max_order=3)
    for r in range(3):
        got = ps.length[ps.rx_slice(r)]
        assert np.allclose(got, lattice_lengths(tx, rx[r], spec.room, 3), atol=1e-4)
    assert np.all(np.diff(ps.length[ps.rx_slice(0)]) >= 0)               # sorted


def test_los_amplitude_and_phase_removed():
    ps = trace(t3.SceneSpec((8.0, 5.0, 3.0)), [1, 1, 1.5], [[6, 3, 1.0]], FC, max_order=0)
    L = np.linalg.norm([5, 2, -0.5])
    assert len(ps.length) == 1 and ps.order[0] == 0 and np.isclose(ps.length[0], L, atol=1e-5)
    assert np.isclose(ps.a[0], LAM / (4 * np.pi * L), rtol=1e-4)           # real positive: no phase left
    u = np.array([-5, -2, 0.5]) / L
    assert np.allclose(ps.aoa[0], u, atol=1e-5) and np.allclose(ps.aod[0], -u, atol=1e-5)


def _theta_hat(u):
    th, ph = np.arccos(u[2]), np.arctan2(u[1], u[0])
    return np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])


def test_first_order_matches_polarized_fresnel():
    """Each single-bounce amplitude equals lambda/(4 pi L) times the Fresnel TE and TM coefficients applied
    to the vertical polarization vector (which has a TE and a TM part because the rays are tilted)."""
    Lx, Ly, Lz = 8.0, 5.0, 3.0
    spec = t3.SceneSpec((Lx, Ly, Lz), material_thickness=1.0)
    tx, rx = np.array([1.5, 1.2, 1.6]), np.array([5.0, 3.0, 1.1])
    ps = trace(spec, tx, [rx], FC, max_order=1)
    eta = c2.eta("concrete", FC)
    images = {"floor": ([tx[0], tx[1], -tx[2]], 2), "ceiling": ([tx[0], tx[1], 2 * Lz - tx[2]], 2),
              "x0": ([-tx[0], tx[1], tx[2]], 0), "x1": ([2 * Lx - tx[0], tx[1], tx[2]], 0),
              "y0": ([tx[0], -tx[1], tx[2]], 1), "y1": ([tx[0], 2 * Ly - tx[1], tx[2]], 1)}
    for name, (img, axis) in images.items():
        img = np.array(img)
        L = np.linalg.norm(rx - img)
        k = np.argmin(np.abs(ps.length - L))
        assert abs(ps.length[k] - L) < 1e-4 and ps.order[k] == 1, name
        n = np.zeros(3)
        n[axis] = 1
        d_out = (rx - img) / L                                              # direction after the bounce
        d_in = d_out - 2 * (d_out @ n) * n                                  # direction before it
        theta = np.arccos(abs(d_in @ n))
        e_te = np.cross(d_in, n)
        e_te /= np.linalg.norm(e_te)
        e_tm_in, e_tm_out = np.cross(e_te, d_in), np.cross(e_te, d_out)
        E = _theta_hat(d_in)
        E_out = (c2.fresnel(theta, eta, "TE") * (E @ e_te) * e_te
                 + c2.fresnel(theta, eta, "TM") * (E @ e_tm_in) * e_tm_out)
        expect = LAM / (4 * np.pi * L) * (_theta_hat(-d_out) @ E_out)
        assert np.isclose(ps.a[k], expect, rtol=1e-3, atol=0), (name, ps.a[k], expect)


def test_box_blocks_los_and_invalid_receivers():
    spec = t3.SceneSpec((8.0, 5.0, 3.0), boxes=((3.5, 0.0, 0.0, 4.5, 5.0, 3.0),))   # full-height wall
    tx = [1.0, 2.5, 1.5]
    rx = [[7.0, 2.5, 1.5], [2.0, 2.5, 1.5], [4.0, 2.5, 1.5]]                 # behind, in front, inside
    ps = trace(spec, tx, rx, FC, max_order=2)
    assert list(ps.rx_valid) == [True, True, False]
    assert ps.n_paths()[2] == 0
    assert 0 not in ps.order[ps.rx_slice(0)]                                # LOS blocked
    assert 0 in ps.order[ps.rx_slice(1)]                                    # LOS present


def test_batch_invariance_and_transmitter_check():
    """Batch size must not change the paths. Matching is order-free and tolerant at the single-precision
    level: paths of (almost) equal length can swap places and GPU arithmetic differs by a few ulps."""
    spec = t3.SceneSpec((8.0, 5.0, 3.0))
    rx, _, _ = t3.make_rx_grid3d((2, 3), (1, 2), 1.2, 0.5)
    p1 = trace(spec, [1, 1, 1.5], rx, FC, max_order=2, batch=1)
    p2 = trace(spec, [1, 1, 1.5], rx, FC, max_order=2, batch=100)
    assert np.array_equal(p1.offsets, p2.offsets)
    for r in range(len(rx)):
        s1, s2 = p1.rx_slice(r), p2.rx_slice(r)
        L1, L2, a1, a2 = p1.length[s1], p2.length[s2], p1.a[s1], p2.a[s2]
        same = (np.abs(L1[:, None] - L2[None, :]) < 1e-4) & \
               (np.abs(a1[:, None] - a2[None, :]) <= 2e-3 * np.abs(a1)[:, None])
        assert same.any(axis=1).all() and same.any(axis=0).all(), \
            f"receiver {r}: unmatched paths; max length diff {np.abs(L1 - L2).max():.2e} m, " \
            f"max |a| diff {np.abs(np.abs(a1) - np.abs(a2)).max():.2e}"
    with pytest.raises(ValueError, match="Transmitter"):
        t3.trace_paths3d(spec, [20, 1, 1], rx, FC)


def test_traced_file_feeds_channel3d(tmp_path):
    spec = t3.SceneSpec((8.0, 5.0, 3.0))
    rx, _, _ = t3.make_rx_grid3d((2, 3), (1, 2), 1.2, 0.5)
    ps = trace(spec, [1, 1, 1.5], rx, FC, max_order=3)
    t3.save_pathset3d(tmp_path / "p.h5", ps)
    q = t3.load_pathset3d(tmp_path / "p.h5")
    assert q.meta["diffraction"] is False and q.meta["spec"]["room"] == [8.0, 5.0, 3.0]
    band = c2.BandConfig("t", FC, 80e6, 256)
    for mode in c3.PHASE_MODES:
        ch = c3.build_channels3d(q, band, mode, elem=c3.make_ula3d(2, LAM / 2))
        assert np.all(np.isfinite(ch.H)) and ch.H.shape == (len(rx), 2, 256)
