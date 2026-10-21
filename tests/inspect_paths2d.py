"""inspect_paths.py -- quick looks at traced paths.  Usage: python inspect_paths.py paths.h5
Power values are a geometry-only proxy, (GAMMA**order / length)**2, not a real channel (that is stage 2)."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import paths2d as p2

GAMMA = 0.7  # placeholder per-bounce reflection loss, used only for the proxies below


def nearest(ps, xy):
    """Index of the receiver closest to point xy (m)."""
    return int(np.argmin(np.linalg.norm(ps.rx - np.asarray(xy, float), axis=1)))


def line_idx(ps, y):
    """Indices of receivers on the grid row nearest height y, ordered by x."""
    y0 = ps.rx[np.argmin(np.abs(ps.rx[:, 1] - y)), 1]
    idx = np.where(np.abs(ps.rx[:, 1] - y0) < 1e-9)[0]
    return idx[np.argsort(ps.rx[idx, 0])]


def path_power(ps, i):
    """Relative power of each path of receiver i (geometry-only proxy)."""
    sl = ps.rx_slice(i)
    return (GAMMA ** ps.order[sl] / ps.length[sl]) ** 2


def summary(ps, i):
    """Metrics for receiver i: path count, LOS flag, first arrival, rms delay spread, K-like ratio."""
    sl, p = ps.rx_slice(i), path_power(ps, i)
    if p.size == 0:
        return dict(rx=i, n=0)
    tau, o = ps.length[sl] / p2.C0, ps.order[sl]
    m = (p * tau).sum() / p.sum()
    k = 10 * np.log10(p[o == 0].sum() / p[o > 0].sum()) if (o == 0).any() and p.size > 1 else np.nan
    return dict(rx=i, n=p.size, los=bool((o == 0).any()), first_ns=tau[0] * 1e9,
                rms_ns=np.sqrt((p * (tau - m) ** 2).sum() / p.sum()) * 1e9, k_db=k)


def print_table(ps, idxs):
    """Print one summary row per receiver index."""
    print(f"{'rx':>6} {'x':>8} {'y':>8} {'paths':>6} {'LOS':>5} {'first ns':>9} {'rms ns':>8} {'K dB':>7}")
    for i in idxs:
        s = summary(ps, i)
        print(f"{i:6d} {ps.rx[i, 0]:8.3f} {ps.rx[i, 1]:8.3f} {s['n']:6d} {str(s.get('los', '-')):>5} "
              f"{s.get('first_ns', np.nan):9.2f} {s.get('rms_ns', np.nan):8.2f} {s.get('k_db', np.nan):7.1f}")


def compare_rx(ps, targets, fname=None):
    """Side-by-side path plots for the receivers nearest each (x, y) target. Returns their indices."""
    idxs = [nearest(ps, t) for t in targets]
    fig, axes = plt.subplots(len(idxs), 1, figsize=(10, 3.2 * len(idxs)), squeeze=False)
    for ax, i in zip(axes[:, 0], idxs):
        p2.plot_paths(ps, i, ax=ax)
    fig.tight_layout()
    fig.savefig(fname, dpi=150) if fname else None
    return idxs


def array_view(ps, idxs, fname=None):
    """Array check: element layout in the scene, plus each element's power-delay profile (proxy)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    for w0, w1 in zip(ps.scene.p0, ps.scene.p1):
        a.plot([w0[0], w1[0]], [w0[1], w1[1]], "k-")
    a.plot(*ps.tx, "g^", ms=10)
    for n, i in enumerate(idxs):
        a.plot(*ps.rx[i], "o", color=f"C{n}")
        sl = ps.rx_slice(i)
        b.plot(ps.length[sl] / p2.C0 * 1e9, 10 * np.log10(path_power(ps, i)), "o", ms=4,
               color=f"C{n}", alpha=0.7, label=f"rx {i}")
    a.set_aspect("equal"), b.legend(), b.set_xlabel("delay (ns)"), b.set_ylabel("relative power (dB)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150) if fname else None
    print_table(ps, idxs)


def line_profile(ps, y, fname=None):
    """Path count, rms delay spread and K-like ratio along the receiver row nearest height y."""
    idx = line_idx(ps, y)
    s = [summary(ps, i) for i in idx]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    for a, key, lab in zip(ax, ("n", "rms_ns", "k_db"), ("paths", "rms delay spread (ns)", "K proxy (dB)")):
        a.plot(ps.rx[idx, 0], [r.get(key, np.nan) for r in s]), a.set_ylabel(lab)
    ax[-1].set_xlabel("x (m)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150) if fname else None


if __name__ == "__main__":
    ps = p2.load_pathset(sys.argv[1])
    xmin, xmax, ym = ps.rx[:, 0].min(), ps.rx[:, 0].max(), float(np.median(ps.rx[:, 1]))
    idxs = compare_rx(ps, [(x, ym) for x in np.linspace(xmin, xmax, 3)], "compare_rx.png")
    print_table(ps, idxs)
    row = line_idx(ps, ym)
    array_view(ps, row[:: max(len(row) // 4, 1)][:4], "array_view.png")
    line_profile(ps, ym, "line_profile.png")
    plt.show()