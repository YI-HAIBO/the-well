# -*- coding: utf-8 -*-
"""#8/#9 (Round 38, v2): spectral-decay regularity (implied alpha) across all 8 systems.

Same estimator family as the paper's E4a (run_e4.py e4a_spectral_slope): keep modes
|kx|,|ky| <= K, rel-L2 truncation error vs N=(2K+1)^2, log-log fit on the informative
band (1e-4<err<0.8); asymptotic tail = last 60% of informative points; alpha = -2*slope.
Robustness upgrade: per-snapshot fits over up to 3 trajectories x 3 time steps,
report the MEDIAN alpha (and min/max). Sanity anchor: shear pressure median should
reproduce the paper's implied alpha = 4.27.
"""
import io, os, sys, json, time
import numpy as np, h5py, fsspec, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "spectral_breadth_8systems.json")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

Ks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def fit_one(u):
    """u: (H, W) single snapshot -> (slope_asym, alpha_asym, slope_full, alpha_full)."""
    H, W = u.shape
    uh = np.fft.fft2(np.asarray(u, dtype=np.float64))
    norm_u = np.linalg.norm(u)
    kx = np.fft.fftfreq(H) * H
    ky = np.fft.fftfreq(W) * W
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    pts = []
    for K in Ks:
        mask = (np.abs(KX) <= K) & (np.abs(KY) <= K)
        rec = np.fft.ifft2(uh * mask).real
        err = float(np.linalg.norm(u - rec) / norm_u)
        pts.append((int(mask.sum()), err))
    fit_pts = [(n, e) for n, e in pts if 1e-4 < e < 0.8]

    def fit(sub):
        xs = np.log(np.array([n for n, _ in sub], dtype=float))
        ys = np.log(np.array([e for _, e in sub], dtype=float))
        A = np.vstack([xs, np.ones_like(xs)]).T
        sl, ic = np.linalg.lstsq(A, ys, rcond=None)[0]
        yh = A @ np.array([sl, ic])
        ssr = float(np.sum((ys - yh) ** 2)); sst = float(np.sum((ys - ys.mean()) ** 2))
        return sl, (1 - ssr / sst if sst > 0 else float("nan"))

    sl_f, r2_f = fit(fit_pts)
    tail = fit_pts[int(len(fit_pts) * 0.4):]
    sl_a, r2_a = fit(tail)
    return sl_a, -2 * sl_a, sl_f, -2 * sl_f, r2_a, r2_f


def fit_field(snaps):
    """snaps: list of (H, W). Median over snapshots."""
    alphas_a, alphas_f, r2s = [], [], []
    for u in snaps:
        sl_a, al_a, sl_f, al_f, r2_a, r2_f = fit_one(u)
        alphas_a.append(al_a); alphas_f.append(al_f); r2s.append(r2_a)
    return {"n_snapshots": len(snaps),
            "alpha_asym_median": round(float(np.median(alphas_a)), 3),
            "alpha_asym_min": round(float(np.min(alphas_a)), 3),
            "alpha_asym_max": round(float(np.max(alphas_a)), 3),
            "alpha_full_median": round(float(np.median(alphas_f)), 3),
            "r2_asym_median": round(float(np.median(r2s)), 4)}


def fetch_native(regname, field_hint, split):
    from the_well.utils import download as d
    R = yaml.safe_load(io.open(d.WELL_REGISTRY, encoding="utf-8"))
    url = R[regname][split][0]
    fs, _ = fsspec.url_to_fs(url, cache_type="blockcache", block_size=8 * 1024 * 1024)
    with fs.open(url, "rb") as f, h5py.File(f, "r") as h5:
        fgroup = "t0_fields" if "t0_fields" in h5 else "t1_fields"
        fn = list(h5[fgroup].attrs.get("field_names", []))
        field = None
        for cand in [field_hint, "tracer", "A", "density", "pressure", "height", "velocity", "buoyancy", "speed_of_sound"]:
            for n_ in fn:
                if cand in n_ and h5[fgroup][n_].attrs.get("time_varying"):
                    field = n_; break
            if field: break
        if field is None: field = fn[0]
        dset = h5[fgroup][field]
        N = min(3, dset.shape[0]); T = min(9, dset.shape[1])
        arr = dset[:N, :T].astype(np.float64)
        print(f"  {regname}/{field}: shape={dset.shape} -> read {arr.shape}", flush=True)
        return arr


def main():
    sets = {}
    z = np.load(os.path.join(BASE, "data", "slice_train.npz"))
    sets["shear"] = [z["pressure"][i, t] for i in range(3) for t in (5, 10, 15)]
    print("shear pressure native", z["pressure"].shape[2:], "x9 snapshots", flush=True)
    import h5py as hp
    tp = os.path.join(BASE, "turb_rad_layer.hdf5")
    with hp.File(tp, "r") as f:
        g = "t0_fields" if "t0_fields" in f else list(f.keys())[0]
        key = "density" if "density" in f[g] else "density"
        d = f[g][key]
        T = d.shape[1]
        ts = [t for t in (5, 10, 15) if t < T] or [T // 2]
        sets["turb"] = [d[i, t] for i in range(min(3, d.shape[0])) for t in ts]
        print("turb", key, "native", d.shape[2:], "x", len(sets["turb"]), "snapshots", flush=True)
    plan = [("gray", "gray_scott_reaction_diffusion", "A"),
            ("euler", "euler_multi_quadrants_periodicBC", "density"),
            ("visco", "viscoelastic_instability", "pressure"),
            ("rb", "rayleigh_benard", "pressure"),
            ("ac", "acoustic_scattering_inclusions", "pressure"),
            ("ps", "planetswe", "height")]
    for key, reg, fld in plan:
        try:
            arr = fetch_native(reg, fld, "train")
            sets[key] = [arr[i, t] for i in range(arr.shape[0]) for t in (1, 4, 7) if t < arr.shape[1]]
        except Exception as e:
            print("FETCH FAIL", key, repr(e)[:200], flush=True)

    res = {}
    for key, snaps in sets.items():
        r = fit_field(snaps)
        r["shape"] = [int(snaps[0].shape[0]), int(snaps[0].shape[1])]
        res[key] = r
        print(f"{key:>6}: {r['shape'][0]}x{r['shape'][1]} n={r['n_snapshots']}  "
              f"alpha_asym median={r['alpha_asym_median']:.2f} [{r['alpha_asym_min']:.2f},{r['alpha_asym_max']:.2f}]  "
              f"alpha_full median={r['alpha_full_median']:.2f}  r2={r['r2_asym_median']:.3f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("Saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
