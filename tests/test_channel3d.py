"""Tests for trace3d (path-file schema/I-O) and channel3d (3D channel synthesis). No Sionna needed."""

import numpy as np
import pytest

import channel2d as c2
import channel3d as c3
import paths2d as p2
import trace3d as t3

C0 = t3.C0
BAND = c2.BANDS["wifi_5ghz"]


def make_ps(lengths, a, aoa, rx=None, fc=None, valid=None, counts=None):
    """PathSet3D from flat arrays. counts = paths per receiver (default one receiver gets all)."""
    lengths = np.asarray(lengths, float)
    P = len(lengths)
    counts = [P] if counts is None else counts
    R = len(counts)
    aoa = np.asarray(aoa, float).reshape(P, 3)
    return t3.PathSet3D(
        tx=np.zeros(3), rx=np.zeros((R, 3)) if rx is None else rx,
        rx_valid=np.ones(R, bool) if valid is None else np.asarray(valid, bool),
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(int), length=lengths,
        order=np.zeros(P, int), a=np.asarray(a, complex), aoa=aoa, aod=-aoa,
        fc_trace=BAND.fc if fc is None else fc)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------- path file ----------------

def test_pathset_roundtrip(tmp_path):
    ps = make_ps([5, 7.5], [1e-3, 2e-4j], [[1, 0, 0], [0, 0, 1]])
    ps.meta = {"scene": "box", "max_order": 3}
    t3.save_pathset3d(tmp_path / "p.h5", ps)
    q = t3.load_pathset3d(tmp_path / "p.h5")
    for k in ("tx", "rx", "rx_valid", "offsets", "length", "order", "a", "aoa", "aod"):
        assert np.array_equal(getattr(ps, k), getattr(q, k))
    assert q.fc_trace == ps.fc_trace and q.meta == ps.meta


def test_pathset_validation():
    ps = make_ps([5], [1e-3], [[1, 0, 0]])
    ps.aoa = np.array([[2.0, 0, 0]])
    with pytest.raises(ValueError, match="unit"):
        ps.validate()
    ps = make_ps([5, 6], [1, 1], [[1, 0, 0]] * 2, counts=[1, 1], valid=[True, False])
    with pytest.raises(ValueError, match="invalid receivers"):
        ps.validate()
    ps = make_ps([-5], [1], [[1, 0, 0]])
    with pytest.raises(ValueError, match="positive"):
        ps.validate()


def test_wrong_file_kind(tmp_path):
    c2.save_channels(tmp_path / "c2.h5", c2.ChannelSet(
        H=np.zeros((1, 4), complex), freqs=np.zeros(4), rx=np.zeros((1, 2)), rx_valid=np.ones(1, bool),
        n_paths=np.zeros(1, int), tx=np.zeros(2), band=c2.BandConfig("x", 1e9, 1e6, 4)))
    with pytest.raises(ValueError, match="not a 3D"):
        c3.load_channels3d(tmp_path / "c2.h5")
    with pytest.raises(ValueError, match="not a 3D"):
        t3.load_pathset3d(tmp_path / "c2.h5")


# ---------------- synthesis ----------------

def test_geometric_matches_channel2d():
    """Same physics through both pipelines gives the same H: 2D trace -> 2D channels vs 3D file."""
    scene = p2.rectangle_room(8.0, 5.0)
    rx, _, _ = p2.make_rx_grid((3.0, 4.0), (2.0, 2.5), 0.25)
    ps2 = p2.trace_paths(scene, (1.0, 1.0), rx, max_order=3)
    H2 = c2.build_channels(ps2, BAND).H
    gains = c2.path_gains(ps2, BAND)
    u = np.stack([np.cos(ps2.aoa), np.sin(ps2.aoa), np.zeros_like(ps2.aoa)], 1)
    ps3 = t3.PathSet3D(tx=np.array([1.0, 1.0, 0]), rx=np.c_[ps2.rx, np.zeros(len(ps2.rx))],
                       rx_valid=ps2.rx_valid, offsets=ps2.offsets, length=ps2.length, order=ps2.order,
                       a=gains, aoa=u, aod=u, fc_trace=BAND.fc)
    H3 = c3.build_channels3d(ps3, BAND, "geometric").H[:, 0, :]
    ok = ps2.rx_valid
    assert np.allclose(H3[ok], H2[ok], rtol=1e-9, atol=1e-18)


def test_single_path_carrier_phase_and_gain_convention():
    L, a = 12.345, 3e-4 * np.exp(0.4j)
    ps = make_ps([L], [a], [[1, 0, 0]])
    ch = c3.build_channels3d(ps, BAND, "geometric")
    assert np.isclose(ch.narrowband()[0, 0], a * np.exp(-2j * np.pi * BAND.fc * L / C0), rtol=1e-9)
    # amplitude is flat across the band for one path
    assert np.allclose(np.abs(ch.H[0, 0]), abs(a))


def test_array_steering_matches_true_distances():
    """Far source: element phases follow exact distances to the source (plane-wave error negligible)."""
    d, direction = 2000.0, unit([0.6, 0.3, 0.74])
    src = d * direction
    elem = c3.make_ula3d(4, 0.4 * BAND.wavelength, axis=(1, 1, 0))
    ps = make_ps([d], [1.0], direction[None, :])
    ch = c3.build_channels3d(ps, BAND, "geometric", elem=elem)
    dist = np.linalg.norm(src[None, :] - elem, axis=1)
    f = ch.freqs
    expect = np.exp(-2j * np.pi * f[None, :] * dist[:, None] / C0)
    assert np.allclose(ch.H[0], expect, atol=2e-5)


def test_random_mode_reproducible_and_envelope_free_of_phase():
    ps = make_ps([5, 6, 7], [1e-3, 5e-4, 2e-4], [[1, 0, 0]] * 3)
    a = c3.build_channels3d(ps, BAND, seed=3)
    b = c3.build_channels3d(ps, BAND, seed=3)
    c = c3.build_channels3d(ps, BAND, seed=4)
    assert np.array_equal(a.H, b.H) and not np.allclose(a.H, c.H)
    # single path: envelope is the amplitude whatever the phase
    one = make_ps([5], [1e-3], [[1, 0, 0]])
    assert np.allclose(np.abs(c3.build_channels3d(one, BAND, seed=9).H), 1e-3)


def test_random_mode_keeps_delay_structure():
    """Random carrier phases leave the delays alone: CIR peaks sit at L/C0 whatever the phases."""
    L1, L2 = 10.0, 60.0
    ps = make_ps([L1, L2], [1e-3, 1e-3], [[1, 0, 0]] * 2)
    H = c3.build_channels3d(ps, BAND, seed=1).H[0, 0]
    t, h = c2.cir(H, BAND, nfft=4096, window=None)
    mag, split = np.abs(h[0]), 35.0 / C0
    lo, hi = t < split, t >= split
    assert abs(t[lo][np.argmax(mag[lo])] - L1 / C0) < 1e-9
    assert abs(t[hi][np.argmax(mag[hi])] - L2 / C0) < 1e-9


def test_rayleigh_and_rician_moments():
    rng = np.random.default_rng(0)
    R, K = 6000, 40
    def run(los_power_ratio):
        lengths = rng.uniform(5, 30, R * (K + 1))
        a = np.ones(R * (K + 1)) * np.sqrt(1.0 / K)
        a[::K + 1] = np.sqrt(los_power_ratio)                      # first path of every receiver is LOS
        ps = make_ps(lengths, a, np.tile([1.0, 0, 0], (len(a), 1)), rx=np.zeros((R, 3)), counts=[K + 1] * R)
        e = c3.build_channels3d(ps, BAND, seed=5).envelope()[:, 0]
        return np.mean(e ** 4) / np.mean(e ** 2) ** 2, np.mean(e ** 2 < 0.1 * np.mean(e ** 2))
    ratio, tail = run(0.0)                                         # scatter power 1 + 1/K, no strong LOS
    assert abs(ratio - 2.0) < 0.12 and abs(tail - (1 - np.exp(-0.1))) < 0.012
    Kf = 4.0                                                       # Rician K = LOS power / scatter power
    ratio, _ = run(Kf)
    assert abs(ratio - (Kf ** 2 + 4 * Kf + 2) / (Kf + 1) ** 2) < 0.08


def test_isotropic_scattering_array_correlation_is_sinc():
    """3D isotropic arrivals + array steering: complex correlation of two elements = sin(kd)/(kd)."""
    rng = np.random.default_rng(1)
    R, K = 4000, 40
    u = unit(rng.normal(size=(R * K, 3)))
    ps = make_ps(rng.uniform(5, 30, R * K), np.ones(R * K), u, rx=np.zeros((R, 3)), counts=[K] * R)
    d = 0.3 * BAND.wavelength
    H = c3.build_channels3d(ps, BAND, elem=c3.make_ula3d(2, d, (0, 0, 1)), seed=2).narrowband()
    rho = np.mean(H[:, 0] * np.conj(H[:, 1])) / np.sqrt(np.mean(abs(H[:, 0]) ** 2) * np.mean(abs(H[:, 1]) ** 2))
    kd = 2 * np.pi * d / BAND.wavelength
    assert abs(rho - np.sin(kd) / kd) < 0.06


def test_invalid_and_empty_receivers():
    ps = make_ps([5], [1e-3], [[1, 0, 0]], counts=[1, 0, 0], valid=[True, True, False],
                 rx=np.zeros((3, 3)))
    ch = c3.build_channels3d(ps, BAND)
    assert np.all(np.isfinite(ch.H[0])) and np.all(ch.H[1] == 0) and np.all(np.isnan(ch.H[2]))
    assert list(ch.n_paths) == [1, 0, 0]


def test_errors():
    ps = make_ps([5], [1e-3], [[1, 0, 0]])
    with pytest.raises(ValueError, match="phase_mode"):
        c3.build_channels3d(ps, BAND, "magic")
    with pytest.raises(ValueError, match="traced at"):
        c3.build_channels3d(ps, c2.BANDS["nr_n78"])
    c3.build_channels3d(ps, c2.BANDS["nr_n78"], check_fc=False)
    with pytest.raises(ValueError, match="elem"):
        c3.build_channels3d(ps, BAND, elem=np.zeros((2, 2)))
    with pytest.warns(UserWarning, match="plane-wave"):
        c3.build_channels3d(ps, BAND, elem=c3.make_ula3d(2, 2.0))


def test_channels_roundtrip_and_cir(tmp_path):
    ps = make_ps([5, 60], [1e-3, 4e-4], [[1, 0, 0], [0, 1, 0]])
    ch = c3.build_channels3d(ps, BAND, elem=c3.make_ula3d(3, 0.02), seed=7)
    c3.save_channels3d(tmp_path / "c.h5", ch, source="p.h5")
    q = c3.load_channels3d(tmp_path / "c.h5")
    assert np.array_equal(q.H, ch.H) and np.array_equal(q.elem, ch.elem) and q.band == ch.band
    assert q.meta["phase_mode"] == "random_carrier" and q.meta["seed"] == 7 and q.meta["source"] == "p.h5"
    assert q.H.shape == (1, 3, BAND.n_sc) and q.envelope().shape == (1, 3)
    t, h = c3.cir(ch.H, BAND, nfft=4096, window=None)
    assert h.shape == (1, 3, 4096)
    assert np.isclose(np.abs(h[0, 0, t < 35 / C0]).max(), 1e-3, rtol=0.03)
