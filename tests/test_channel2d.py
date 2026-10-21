"""
test_channel2d.py -- verification suite for channel2d.py (stage 2).  Run: pytest -v test_channel2d.py

Physics checks against closed forms (not just "it runs"):
* Fresnel: normal incidence, perfect-conductor limits, Brewster angle, passivity, grazing incidence
* ITU material formula
* Frequency grid: carrier on index N//2
* Single path: free-space path loss in dB and delay recovered from the phase slope
* CIR: peak at the right delay with the right amplitude (no window)
* Two equal paths: comb-filter nulls at f = (n + 1/2) / delta_tau
* Perfect-conductor room: path gains equal (+/-)^order * lambda/(4 pi L) for TE / TM
* build_channels equals a manual synthesis; invalid receivers are NaN; shadowed receivers are zero
* HDF5 round trip; selection helpers; local normalization; rms delay spread
"""

import numpy as np
import pytest

import channel2d as c2
import paths2d as p2

BAND = c2.BANDS["wifi_5ghz"]


@pytest.fixture
def box_scene():
    return p2.add_box(p2.rectangle_room(10.0, 8.0), 4.0, 3.0, 6.0, 5.0)


# ------------------------------------------------------------------ materials and Fresnel

def test_itu_formula():
    f = 5e9
    a, b, cc, d, *_ = c2.ITU["concrete"]
    expect = a - 1j * cc * 5.0 ** d / (2 * np.pi * f * c2.EPS0)
    assert c2.eta("concrete", f) == pytest.approx(expect)
    assert c2.eta((4.0, 0.0), f) == pytest.approx(4.0)
    assert c2.eta("vacuum", f) == pytest.approx(1.0)


def test_unknown_material_message():
    with pytest.raises(KeyError, match="Unknown material"):
        c2.eta("unobtainium", 5e9)


def test_fresnel_normal_incidence():
    assert c2.fresnel(0.0, 4.0, "TE") == pytest.approx(-1 / 3)    # (1-2)/(1+2)
    assert c2.fresnel(0.0, 4.0, "TM") == pytest.approx(+1 / 3)


def test_fresnel_perfect_conductor_limits():
    e = c2.eta("metal", 5.5e9)
    th = np.array([0.0, 0.4, 1.0])
    assert np.allclose(c2.fresnel(th, e, "TE"), -1, atol=2e-3)
    assert np.allclose(c2.fresnel(th, e, "TM"), +1, atol=2e-3)


def test_fresnel_brewster():
    assert abs(c2.fresnel(np.arctan(2.0), 4.0, "TM")) < 1e-12     # tan(theta_B) = sqrt(eps)


def test_fresnel_passive_and_grazing():
    th = np.linspace(0, np.pi / 2 - 1e-9, 200)
    for mat in ("concrete", "glass", "brick", "wood"):
        e = c2.eta(mat, 5e9)
        for pol in ("TE", "TM"):
            g = c2.fresnel(th, e, pol)
            assert np.all(np.abs(g) <= 1 + 1e-12)
            assert g[-1] == pytest.approx(-1, abs=1e-3)           # grazing: total reflection, sign -1


# ------------------------------------------------------------------ band grid

def test_band_grid():
    f = BAND.freqs()
    assert len(f) == BAND.n_sc and f[BAND.n_sc // 2] == pytest.approx(BAND.fc)
    assert np.allclose(np.diff(f), BAND.delta_f)
    assert f[0] == pytest.approx(BAND.fc - BAND.bandwidth / 2)
    with pytest.raises(ValueError):
        c2.BandConfig("odd", 5e9, 20e6, 255)


# ------------------------------------------------------------------ single path and CIR

def test_single_path_free_space_loss_and_delay():
    d, f = 12.0, BAND.freqs()
    lam = BAND.wavelength
    H = c2.synthesize([d], [lam / (4 * np.pi * d)], f)
    assert np.allclose(20 * np.log10(np.abs(H)), -20 * np.log10(4 * np.pi * d / lam))   # Friis, flat in f
    slope = np.angle(H[1:] / H[:-1])                                                    # = -2 pi df tau
    assert np.allclose(slope, -2 * np.pi * BAND.delta_f * d / c2.C0, atol=1e-9)


def test_free_space_loss_through_pipeline():
    sc = p2.rectangle_room(30.0, 30.0)
    ps = p2.trace_paths(sc, (5.0, 15.0), [[17.0, 15.0]], max_order=0)
    ch = c2.build_channels(ps, BAND)
    expect = -20 * np.log10(4 * np.pi * 12.0 / BAND.wavelength)
    assert 20 * np.log10(ch.envelope()[0]) == pytest.approx(expect)


def test_cir_peak_delay_and_amplitude():
    nfft = 2048
    tstep = 1 / (nfft * BAND.delta_f)
    tau = 40 * tstep                                   # exactly on the interpolated time grid
    a = 3.7e-5
    H = c2.synthesize([tau * c2.C0], [a], BAND.freqs())
    t, h = c2.cir(H, BAND, nfft=nfft, window=None)
    k = int(np.argmax(np.abs(h[0])))
    assert t[k] == pytest.approx(tau) and abs(h[0, k]) == pytest.approx(a, rel=1e-9)
    t, hw = c2.cir(H, BAND, nfft=nfft, window="hann")  # window changes sidelobes, not the peak amplitude
    assert abs(hw[0, k]) == pytest.approx(a, rel=1e-6)


# ------------------------------------------------------------------ two paths

def test_two_path_comb_nulls():
    dtau = 10e-9                                       # nulls at (n + 1/2) / dtau = 50 MHz, 150 MHz, ...
    f = np.arange(5.40e9, 5.60e9 + 1, 1e6)
    H = c2.synthesize([30.0, 30.0 + dtau * c2.C0], [1.0, 1.0], f)
    mag = np.abs(H)
    nulls = f[1:-1][(mag[1:-1] < mag[:-2]) & (mag[1:-1] < mag[2:])]
    assert np.allclose(nulls, [5.45e9, 5.55e9], atol=1e6)
    assert mag[np.argmin(np.abs(f - 5.45e9))] < 1e-9


# ------------------------------------------------------------------ perfect-conductor room

@pytest.mark.parametrize("pol,sign", [("TE", -1), ("TM", +1)])
def test_pec_room_gains(pol, sign):
    sc = p2.rectangle_room(10.0, 8.0)
    ps = p2.trace_paths(sc, (2.0, 3.0), [[8.0, 5.0]], max_order=3)
    g = c2.path_gains(ps, BAND, material_map={"wall": "metal"}, pol=pol)
    expect = (sign ** ps.order) * BAND.wavelength / (4 * np.pi * ps.length)
    assert np.allclose(g, expect, rtol=3e-3, atol=0)


def test_material_map_changes_gain():
    sc = p2.rectangle_room(10.0, 8.0)
    ps = p2.trace_paths(sc, (2.0, 3.0), [[8.0, 5.0]], max_order=1)
    gc = c2.path_gains(ps, BAND)                                   # default: concrete
    gg = c2.path_gains(ps, BAND, material_map={"wall": "glass"})
    direct = ps.order == 0
    assert np.allclose(gc[direct], gg[direct])                     # direct path unaffected
    assert not np.allclose(gc[~direct], gg[~direct])
    assert np.all(np.abs(gc[~direct]) < BAND.wavelength / (4 * np.pi * ps.length[~direct]))   # |Gamma| < 1


# ------------------------------------------------------------------ build_channels

def test_build_matches_manual_and_flags(box_scene):
    rx = [[1.0, 1.0], [5.0, 4.0], [9.0, 4.0]]                      # free, inside box, behind box
    ps = p2.trace_paths(box_scene, (1.0, 4.0), rx, max_order=0)    # order 0: behind-box receiver has no path
    ch = c2.build_channels(ps, BAND)
    g = c2.path_gains(ps, BAND)
    sl = ps.rx_slice(0)
    assert np.allclose(ch.H[0], c2.synthesize(ps.length[sl], g[sl], BAND.freqs()))
    assert np.all(np.isnan(ch.H[1]))                               # invalid receiver
    assert np.all(ch.H[2] == 0) and ch.n_paths[2] == 0 and ch.rx_valid[2]   # complete shadow
    assert ch.H.shape == (3, BAND.n_sc) and ch.freqs[BAND.n_sc // 2] == BAND.fc
    assert np.allclose(ch.narrowband()[0], ch.H[0, BAND.n_sc // 2])


def test_complex64_option(box_scene):
    ps = p2.trace_paths(box_scene, (1.0, 1.0), [[2.0, 7.0]], max_order=2)
    a, b = c2.build_channels(ps, BAND), c2.build_channels(ps, BAND, dtype=np.complex64)
    assert b.H.dtype == np.complex64 and np.allclose(a.H, b.H, rtol=1e-4, atol=1e-9)


def test_hdf5_round_trip(tmp_path, box_scene):
    rx, _, _ = p2.make_rx_grid((6.5, 8.5), (1.0, 2.0), 0.5)
    ps = p2.trace_paths(box_scene, (1.0, 1.0), rx, max_order=2, meta={"seed": 3})
    ch = c2.build_channels(ps, BAND, pol="TM")
    f = str(tmp_path / "ch.h5")
    c2.save_channels(f, ch, source="paths.h5")
    back = c2.load_channels(f)
    assert np.array_equal(back.H, ch.H, equal_nan=True)
    for name in ("freqs", "rx", "rx_valid", "n_paths", "tx"):
        assert np.array_equal(getattr(back, name), getattr(ch, name)), name
    assert back.band == ch.band and back.meta["pol"] == "TM"
    assert back.meta["source"] == "paths.h5" and back.meta["stage1"]["seed"] == 3


# ------------------------------------------------------------------ delay spread

class _FakePaths:
    def __init__(self, lengths):
        self.length, self.rx = np.asarray(lengths, float), np.zeros((1, 2))

    def rx_slice(self, r):
        return slice(0, len(self.length))


def test_rms_delay_spread_two_equal_paths():
    ps = _FakePaths([30.0, 30.0 + 10e-9 * c2.C0])
    assert c2.rms_delay_spread(ps, np.array([1.0, 1.0]))[0] == pytest.approx(5e-9)   # half the separation


# ------------------------------------------------------------------ selection and normalization

def test_row_indices_and_ula_groups():
    rx, xs, ys = p2.make_rx_grid((0.0, 1.0), (0.0, 0.5), 0.1)
    row = c2.row_indices(rx, 0.31)
    assert len(row) == len(xs) and np.all(np.diff(rx[row, 0]) > 0) and np.allclose(rx[row, 1], 0.3)
    grp = c2.ula_groups(len(row), stride=3, n_elem=4)
    assert grp.shape == (len(row) - 9, 4) and np.all(np.diff(grp, axis=1) == 3) and grp.max() == len(row) - 1
    assert c2.ula_groups(len(row), 3, 4, hop=2).shape[0] == (len(row) - 9 + 1) // 2


def test_stride_for():
    stride, actual, err = c2.stride_for(0.0259, 0.001)
    assert stride == 26 and actual == pytest.approx(0.026) and abs(err) < 0.01


def test_local_normalize():
    assert np.allclose(c2.local_normalize(np.full(200, 3.0), 21), 1.0)
    rng = np.random.default_rng(0)
    env = np.abs(rng.normal(size=4000) + 1j * rng.normal(size=4000)) * np.linspace(1, 20, 4000)   # slow trend
    out = c2.local_normalize(env, 101)
    assert np.mean(out[200:-200] ** 2) == pytest.approx(1.0, abs=0.05)
    assert np.all(c2.local_normalize(np.zeros(50), 5) == 0)                                          # no 0/0
    assert c2.local_normalize(np.ones((20, 30)), 5).shape == (20, 30)                                 # 2D grid
