"""
Compute the finite-time Lyapunov exponent (FTLE) field from particle trajectories.

Usage
-----
python ftle_extraction.py pos_2day_aug124.npy ftle_2day_aug124.npy
python ftle_extraction.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lcs_common2 import POS_DIR, FTLE_DIR, DT_DEFAULT_H, EARTH_RADIUS_M, metadata_path

DEG2M = np.pi / 180.0 * EARTH_RADIUS_M


def compute_ftle(p0, pT, T):
    """
    p0, pT : (ny, nx, 2) arrays of (lon, lat) at t0 and t0+T (deg). NaN = lost.
    T      : integration time (any unit; FTLE is returned in 1/that unit).
    Returns (ny, nx) FTLE with NaN on the outer ring and wherever the
    centred-difference stencil touches a lost particle.
    """
    x0, y0 = p0[..., 0], p0[..., 1]
    xf, yf = pT[..., 0], pT[..., 1]

    c = (slice(1, -1), slice(1, -1))
    e, w = (slice(1, -1), slice(2, None)), (slice(1, -1), slice(None, -2))
    n, s = (slice(2, None), slice(1, -1)), (slice(None, -2), slice(1, -1))

    # initial separations (m)
    dX = (x0[e] - x0[w]) * DEG2M * np.cos(np.radians(y0[c]))
    dY = (y0[n] - y0[s]) * DEG2M

    # final separations (m), signed; east-west uses the mean final latitude of the pair
    cos_ew = np.cos(np.radians(0.5 * (yf[e] + yf[w])))
    cos_ns = np.cos(np.radians(0.5 * (yf[n] + yf[s])))
    dxdX = (xf[e] - xf[w]) * DEG2M * cos_ew / dX
    dydX = (yf[e] - yf[w]) * DEG2M / dX
    dxdY = (xf[n] - xf[s]) * DEG2M * cos_ns / dY
    dydY = (yf[n] - yf[s]) * DEG2M / dY

    # right Cauchy-Green tensor C = G^T G, G = [[dxdX, dxdY], [dydX, dydY]]
    C11 = dxdX ** 2 + dydX ** 2
    C22 = dxdY ** 2 + dydY ** 2
    C12 = dxdX * dxdY + dydX * dydY
    lam_max = 0.5 * (C11 + C22) + np.sqrt(0.25 * (C11 - C22) ** 2 + C12 ** 2)

    ftle = np.full(x0.shape, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        inner = np.log(lam_max) / (2.0 * T)
    inner[~np.isfinite(inner)] = np.nan
    ftle[c] = inner
    return ftle


def resolve_input(arg):
    p = Path(arg)
    if p.exists():
        return p
    p = POS_DIR / arg
    if p.suffix != '.npy':
        p = p.with_suffix('.npy')
    if not p.exists():
        sys.exit(f"Cannot find particle file '{arg}' (looked in {POS_DIR})")
    return p


def selftest():
    """Analytic checks: FTLE of known linear flow maps."""
    lon = np.linspace(-10, -9, 81)
    lat = np.linspace(53, 54, 81)
    LON, LAT = np.meshgrid(lon, lat)
    p0 = np.stack([LON, LAT], axis=-1)
    lon_c, lat_c = -9.5, 53.5
    X = (LON - lon_c) * DEG2M * np.cos(np.radians(lat_c))
    Y = (LAT - lat_c) * DEG2M

    def to_deg(x, y):
        return np.stack([lon_c + x / (DEG2M * np.cos(np.radians(lat_c))), lat_c + y / DEG2M], axis=-1)

    T, lam = 48.0, 0.02
    k = (slice(10, -10), slice(10, -10))
    # 1) pure strain: x -> x e^{lam T}, y -> y e^{-lam T}  => FTLE = lam
    f = compute_ftle(p0, to_deg(X * np.exp(lam * T), Y * np.exp(-lam * T)), T)
    print(f"strain:           expected {lam:.4f}, got {np.nanmedian(f[k]):.4f}")
    # 2) strain then 60 deg rotation: rotation must not change the FTLE
    th = np.radians(60)
    xs, ys = X * np.exp(lam * T), Y * np.exp(-lam * T)
    f2 = compute_ftle(p0, to_deg(np.cos(th) * xs - np.sin(th) * ys, np.sin(th) * xs + np.cos(th) * ys), T)
    print(f"strain+rotation:  expected {lam:.4f}, got {np.nanmedian(f2[k]):.4f}")
    # 3) simple shear x -> x + g y: lam_max of C = 1 + g^2/2 + g sqrt(1 + g^2/4)
    g = 3.0
    f3 = compute_ftle(p0, to_deg(X + g * Y, Y), T)
    exp3 = np.log(1 + g ** 2 / 2 + g * np.sqrt(1 + g ** 2 / 4)) / (2 * T)
    print(f"shear:            expected {exp3:.4f}, got {np.nanmedian(f3[k]):.4f}")
    # 4) a lost particle blanks its 4 stencil neighbours only
    pT = to_deg(X, Y).copy()
    pT[40, 40] = np.nan
    f4 = compute_ftle(p0, pT, T)
    print(f"lost particle:    NaN interior points = {int(np.isnan(f4[1:-1, 1:-1]).sum())} (expected 4)")
    ok = (abs(np.nanmedian(f[k]) - lam) < 1e-3 and abs(np.nanmedian(f2[k]) - lam) < 1e-3
          and abs(np.nanmedian(f3[k]) - exp3) < 1e-3 and np.isnan(f4[1:-1, 1:-1]).sum() == 4)
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input_pos', nargs='?', help='particle positions file (in data/Particle positions, or a path)')
    ap.add_argument('output_ftle_file', nargs='?', help='output file name (saved in data/FTLE extracted)')
    ap.add_argument('--T', type=float, default=None, help='override integration time (hours)')
    ap.add_argument('--dt', type=float, default=DT_DEFAULT_H,
                    help='integration step (h), only used for legacy files without JSON metadata')
    ap.add_argument('--selftest', action='store_true', help='run analytic tests and exit')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.input_pos or not args.output_ftle_file:
        ap.error('input_pos and output_ftle_file are required')

    in_path = resolve_input(args.input_pos)
    pos = np.load(in_path, mmap_mode='r')          # only the first and last steps are read
    p0, pT = np.array(pos[0]), np.array(pos[-1])

    meta_file = metadata_path(in_path)
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    if args.T is not None:
        T = args.T
    elif 'T_hours' in meta:
        T = meta['T_hours']
    else:
        T = (pos.shape[0] - 1) * args.dt
        print(f"WARNING: no metadata for {in_path.name}; assuming T = (n_steps-1)*dt = {T} h. "
              "Legacy files from the old tracking script contain frozen (not NaN) coastal "
              "particles, so their FTLE near the coast is unreliable -- re-run the tracking.")

    ftle = compute_ftle(p0, pT, T)
    # Land: particles that never moved (freeze mode) or were lost (lost mode) have
    # no FTLE of their own. Store them as NaN so later steps need only this file.
    never_moved = np.all(p0 == pT, axis=-1) | np.isnan(pT[..., 0])
    ftle[never_moved] = np.nan

    out_path = FTLE_DIR / args.output_ftle_file
    np.save(out_path, ftle)
    lons, lats = p0[0, :, 0], p0[:, 0, 1]
    metadata_path(out_path).write_text(json.dumps({
        'source_positions': in_path.name, 'T_hours': T, 'units': '1/hour',
        'direction': meta.get('direction', 'forward (assumed)'),
        'release_time': meta.get('release_time'),
        'lcs_type': 'attracting' if meta.get('direction') == 'backward' else 'repelling',
        'coast_mode': meta.get('coast_mode', 'freeze'),
        'land_is_nan': True,
        # grid of the FTLE cells (= particle seed positions): value = start + step * index
        'grid': {'lon_start': float(lons[0]), 'lon_step': float(lons[1] - lons[0]), 'nx': int(len(lons)),
                 'lat_start': float(lats[0]), 'lat_step': float(lats[1] - lats[0]), 'ny': int(len(lats))},
    }, indent=2))
    valid = np.isfinite(ftle)
    print(f"Saved {out_path}: T = {T} h, {100 * valid.mean():.1f}% of grid valid, "
          f"FTLE median {np.nanmedian(ftle):.4f}, 99th pct {np.nanpercentile(ftle, 99):.4f} 1/h")


if __name__ == '__main__':
    main()
