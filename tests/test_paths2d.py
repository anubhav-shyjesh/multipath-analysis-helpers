"""
test_paths2d.py -- verification suite for paths2d.py (stage 1).

Run with:  pytest -v test_paths2d.py

Each test checks one independent property:

* test_mirror_points                  mirror formula on a known case
* test_images_no_consecutive_repeat   image recursion never reflects twice off the same wall in a row
* test_direct_path_only               order 0 gives one path with the right length and angles
* test_first_order_count_empty_room   empty rectangle: 1 direct + 4 first-order paths
* test_rectangle_matches_lattice      recursive engine == closed-form lattice image formula
* test_reciprocity_with_obstacle      swapping Tx and Rx gives the same path lengths/orders
* test_specular_law_and_geometry      every bounce lies on its wall and obeys angle in = angle out
* test_no_leg_crosses_a_wall          independent brute-force check that no leg is blocked
* test_obstacle_blocks_direct         box between Tx and Rx removes the direct path only
* test_free_space_mask                receivers inside obstacles / outside the room are flagged
* test_tx_in_obstacle_raises          invalid Tx is rejected
* test_max_length_prune               geometric pruning drops long paths
* test_max_order_zero                 degenerate zero-width arrays work
* test_hdf5_round_trip                save -> load reproduces every array and the scene
"""

import numpy as np
import pytest

import paths2d as p2


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def lattice_paths(src, rx, room, max_order):
    """Closed-form image lattice for an EMPTY rectangle.

    Image x = (1-2q)*xs + 2*m*Lx, with |m-q| + |m| reflections in x; same in y.
    Returns a list of (n_reflections, length) sorted by (order, length).
    """
    Lx, Ly = room
    out = []
    for qx in (0, 1):
        for qy in (0, 1):
            for mx in range(-max_order, max_order + 1):
                for my in range(-max_order, max_order + 1):
                    n = abs(mx - qx) + abs(mx) + abs(my - qy) + abs(my)
                    if n > max_order:
                        continue
                    ix = (1 - 2 * qx) * src[0] + 2 * mx * Lx
                    iy = (1 - 2 * qy) * src[1] + 2 * my * Ly
                    out.append((n, float(np.hypot(ix - rx[0], iy - rx[1]))))
    return sorted(out)


def engine_pairs(ps, r=0):
    """(order, length) pairs of receiver r from a PathSet, sorted like lattice_paths."""
    sl = ps.rx_slice(r)
    return sorted(zip(ps.order[sl].tolist(), ps.length[sl].tolist()))


def properly_cross(p, q, a, b, tol=1e-9):
    """True if open segments p-q and a-b cross at a single interior point (orientation test)."""
    def orient(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    d1, d2 = orient(p, q, a), orient(p, q, b)
    d3, d4 = orient(a, b, p), orient(a, b, q)
    return (d1 * d2 < -tol) and (d3 * d4 < -tol)


@pytest.fixture
def box_scene():
    """10 x 8 room with a 2 x 2 box obstacle in the middle."""
    return p2.add_box(p2.rectangle_room(10.0, 8.0), 4.0, 3.0, 6.0, 5.0)


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

def test_mirror_points():
    out = p2.mirror_points([[2.0, 3.0]], [0.0, 0.0], [5.0, 0.0])   # across the x axis
    assert np.allclose(out, [[2.0, -3.0]])
    out = p2.mirror_points([[2.0, 3.0]], [4.0, 0.0], [4.0, 9.0])   # across x = 4
    assert np.allclose(out, [[6.0, 3.0]])


def test_images_no_consecutive_repeat(box_scene):
    imgs = p2.compute_images(box_scene, (1.0, 1.0), 4)
    for n in range(2, 5):
        w = imgs.walls[imgs.order == n]
        assert np.all(w[:, n - 1] != w[:, n - 2])
    assert np.sum(imgs.order == 0) == 1


def test_direct_path_only():
    sc = p2.rectangle_room(10.0, 8.0)
    ps = p2.trace_paths(sc, (2.0, 3.0), [[8.0, 7.0]], max_order=0)
    assert ps.n_paths()[0] == 1
    assert ps.length[0] == pytest.approx(np.hypot(6.0, 4.0))
    assert ps.aod[0] == pytest.approx(np.arctan2(4.0, 6.0))
    assert ps.aoa[0] == pytest.approx(np.arctan2(-4.0, -6.0))   # Rx looks back toward the Tx


def test_first_order_count_empty_room():
    sc = p2.rectangle_room(10.0, 8.0)
    ps = p2.trace_paths(sc, (2.0, 3.0), [[8.0, 5.0]], max_order=1)
    assert sorted(ps.order.tolist()) == [0, 1, 1, 1, 1]


@pytest.mark.parametrize("max_order", [1, 2, 3, 4, 5])
def test_rectangle_matches_lattice(max_order):
    src, rx, room = (2.0, 3.0), (8.0, 5.0), (10.0, 8.0)
    ps = p2.trace_paths(p2.rectangle_room(*room), src, [rx], max_order=max_order)
    got = engine_pairs(ps)
    want = lattice_paths(src, rx, room, max_order)
    assert len(got) == len(want)
    assert [g[0] for g in got] == [w[0] for w in want]
    assert np.allclose([g[1] for g in got], [w[1] for w in want], atol=1e-9)


def test_reciprocity_with_obstacle(box_scene):
    A, B = (1.0, 4.0), (9.0, 3.5)
    ab = p2.trace_paths(box_scene, A, [B], max_order=3)
    ba = p2.trace_paths(box_scene, B, [A], max_order=3)
    got, want = engine_pairs(ab), engine_pairs(ba)
    assert len(got) == len(want) > 0
    assert [g[0] for g in got] == [w[0] for w in want]
    assert np.allclose([g[1] for g in got], [w[1] for w in want], atol=1e-9)


def test_specular_law_and_geometry(box_scene):
    tx, rx = np.array([1.0, 6.5]), np.array([8.0, 1.5])
    ps = p2.trace_paths(box_scene, tx, [rx], max_order=3)
    normals = box_scene.normals()
    assert ps.n_paths()[0] > 1
    for i in range(ps.offsets[0], ps.offsets[1]):
        n = int(ps.order[i])
        chain = np.vstack([tx, ps.points[i, :n], rx])
        assert np.linalg.norm(np.diff(chain, axis=0), axis=1).sum() == pytest.approx(ps.length[i])
        for k in range(1, n + 1):
            w = int(ps.walls[i, k - 1])
            P = chain[k]
            a, b = box_scene.p0[w], box_scene.p1[w]
            seg = b - a
            u = np.dot(P - a, seg) / np.dot(seg, seg)
            assert -1e-9 <= u <= 1 + 1e-9                                   # bounce is on the segment
            cross = seg[0] * (P - a)[1] - seg[1] * (P - a)[0]
            assert abs(cross) / np.linalg.norm(seg) < 1e-9                  # and on its line
            d_in, d_out = chain[k] - chain[k - 1], chain[k + 1] - chain[k]
            d_in, d_out = d_in / np.linalg.norm(d_in), d_out / np.linalg.norm(d_out)
            refl = d_in - 2 * np.dot(d_in, normals[w]) * normals[w]
            assert np.allclose(refl, d_out, atol=1e-9)                      # angle in == angle out
            assert ps.inc_angle[i, k - 1] == pytest.approx(np.arccos(abs(np.dot(d_in, normals[w]))))


def test_no_leg_crosses_a_wall(box_scene):
    tx, rx = np.array([1.0, 4.0]), np.array([9.0, 4.0])
    ps = p2.trace_paths(box_scene, tx, [rx], max_order=4)
    assert ps.n_paths()[0] > 0
    for i in range(ps.offsets[0], ps.offsets[1]):
        n = int(ps.order[i])
        chain = np.vstack([tx, ps.points[i, :n], rx])
        for j in range(len(chain) - 1):
            for a, b in zip(box_scene.p0, box_scene.p1):
                assert not properly_cross(chain[j], chain[j + 1], a, b)


def test_obstacle_blocks_direct(box_scene):
    tx, rx = (1.0, 4.0), (9.0, 4.0)
    ps = p2.trace_paths(box_scene, tx, [rx], max_order=2)
    assert 0 not in ps.order.tolist()          # direct path is blocked by the box
    assert (ps.order == 1).sum() >= 2          # top/bottom wall bounces clear the box
    clear = p2.trace_paths(p2.rectangle_room(10.0, 8.0), tx, [rx], max_order=2)
    assert 0 in clear.order.tolist()           # same endpoints, no box: direct path exists


def test_free_space_mask(box_scene):
    pts = [[5.0, 4.0],      # inside the box
           [1.0, 1.0],      # free
           [11.0, 4.0],     # outside the room
           [3.9, 4.0]]      # just left of the box
    assert p2.free_space_mask(box_scene, pts).tolist() == [False, True, False, True]
    ps = p2.trace_paths(box_scene, (1.0, 1.0), pts, max_order=2)
    assert ps.rx_valid.tolist() == [False, True, False, True]
    assert ps.n_paths()[0] == 0 and ps.n_paths()[2] == 0


def test_tx_in_obstacle_raises(box_scene):
    with pytest.raises(ValueError):
        p2.trace_paths(box_scene, (5.0, 4.0), [[1.0, 1.0]], max_order=1)


def test_max_length_prune():
    sc = p2.rectangle_room(10.0, 8.0)
    full = p2.trace_paths(sc, (2.0, 3.0), [[8.0, 5.0]], max_order=4)
    cut = p2.trace_paths(sc, (2.0, 3.0), [[8.0, 5.0]], max_order=4, max_length=20.0)
    assert cut.n_paths()[0] < full.n_paths()[0]
    assert np.all(cut.length <= 20.0)


def test_max_order_zero(box_scene):
    ps = p2.trace_paths(box_scene, (1.0, 1.0), [[2.0, 7.0], [9.0, 1.0]], max_order=0)
    assert ps.walls.shape[1] == 0
    assert np.all(ps.order == 0) and ps.n_paths().tolist() == [1, 1]


def test_paths_sorted_by_length(box_scene):
    ps = p2.trace_paths(box_scene, (1.0, 6.5), [[8.0, 1.5]], max_order=3)
    assert np.all(np.diff(ps.length) >= 0)


def test_hdf5_round_trip(tmp_path, box_scene):
    pts, _, _ = p2.make_rx_grid((6.5, 8.5), (1.0, 2.0), 0.5)
    ps = p2.trace_paths(box_scene, (1.0, 1.0), pts, max_order=3, meta={"seed": 7})
    f = str(tmp_path / "paths.h5")
    p2.save_pathset(f, ps, config={"note": "test"})
    back = p2.load_pathset(f)
    for name in ("tx", "rx", "rx_valid", "offsets", "length", "order", "walls",
                 "inc_angle", "points", "aoa", "aod"):
        assert np.array_equal(getattr(ps, name), getattr(back, name), equal_nan=True), name
    for name in ("p0", "p1", "material", "group"):
        assert np.array_equal(getattr(ps.scene, name), getattr(back.scene, name)), name
    assert back.scene.group_kind == ps.scene.group_kind
    assert back.meta["seed"] == 7 and back.meta["note"] == "test"
