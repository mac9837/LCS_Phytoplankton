"""
Release virtual particles on a rectilinear grid and advect them through the
IMI NE Atlantic surface velocity field (Numba-parallelised).

Example
-------
python particle_tracking_parallelised.py IMI_NEATL_673e_f015_5722.csv pos_2day_aug124.npy --sim_time 48
python particle_tracking_parallelised.py IMI_NEATL_ab09_9d57_f28d.csv IMI_NEATL_673e_f015_5722.csv pos_9day_aug124.npy --sim_time 216
python particle_tracking_parallelised.py --plan     # print the window and files needed for every week
"""

import argparse
import faulthandler
import json
import math
import time
from pathlib import Path

faulthandler.enable()   # a crash in compiled code prints an error instead of exiting silently

import numpy as np
import pandas as pd
from numba import njit, prange

from lcs_common2 import (VEL_DIR, POS_DIR, LAT_SPACING, LON_SPACING, TIME_STEP_H,
                        DT_DEFAULT_H, EARTH_RADIUS_M, WEEK_LABELS, VELOCITY_FILES,
                        metadata_path, sampling_reference_time, week_from_label)

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
from lcs_common2 import SEED_NPX as NPX, SEED_NPY as NPY, SEED_RP as RP, SEED_X0 as X0, SEED_Y0 as Y0  # noqa: E402
# (seeding rectangle: set in lcs_common2.py so the other scripts can rebuild the grid)

INTEGRATORS = {'euler': 1, 'rk3': 2, 'rk4': 3}
VEL_COLS = ['time', 'latitude', 'longitude', 'sea_surface_x_velocity', 'sea_surface_y_velocity']


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_velocity_csv(path):
    df = pd.read_csv(path, skiprows=[1], usecols=VEL_COLS,       # row 1 = units
                     dtype={'latitude': np.float64, 'longitude': np.float64,
                            'sea_surface_x_velocity': np.float32,
                            'sea_surface_y_velocity': np.float32})
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%SZ')
    print(f"Loaded {Path(path).name}: {df['time'].min()} -> {df['time'].max()}")
    return df


def integration_window(end, sim_time_h):
    """(start, end) of the integration window ending at `end`."""
    end = pd.Timestamp(end)
    return end - pd.Timedelta(hours=sim_time_h), end


# Nominal coverage of each weekly file (00:00 Wednesday -> 00:00 Wednesday;
# jul4 starts Thu 25/7 00:00 and oct1 starts Wed 25/9 08:00, as noted in the
# original script). Used only to suggest which files a run needs.
FILE_COVERAGE = {
    'jul4': ('2024-07-25 00:00', '2024-08-01 00:00'),
    'aug1': ('2024-07-31 00:00', '2024-08-07 00:00'),
    'aug2': ('2024-08-07 00:00', '2024-08-14 00:00'),
    'aug3': ('2024-08-14 00:00', '2024-08-21 00:00'),
    'aug4': ('2024-08-21 00:00', '2024-08-28 00:00'),
    'sep1': ('2024-08-28 00:00', '2024-09-04 00:00'),
    'sep2': ('2024-09-04 00:00', '2024-09-11 00:00'),
    'sep3': ('2024-09-11 00:00', '2024-09-18 00:00'),
    'sep4': ('2024-09-18 00:00', '2024-09-25 00:00'),
    'oct1': ('2024-09-25 08:00', '2024-10-02 08:00'),
    'oct2': ('2024-10-02 00:00', '2024-10-09 00:00'),
    'oct4': ('2024-10-16 00:00', '2024-10-23 00:00'),
}


def files_needed(t_start, t_end):
    """Smallest run of consecutive weekly files (labels, in order) covering the
    window, and whether the window is fully covered."""
    cov = [(lab, pd.Timestamp(a), pd.Timestamp(b)) for lab, (a, b) in FILE_COVERAGE.items()]
    last = [k for k, (_, a, b) in enumerate(cov) if a <= t_end <= b]
    if not last:
        return [], False
    k = last[0]
    chosen = [cov[k]]
    while chosen[0][1] > t_start and k > 0 and cov[k - 1][2] >= chosen[0][1]:   # extend back without gaps
        k -= 1
        chosen.insert(0, cov[k])
    return [c[0] for c in chosen], chosen[0][1] <= t_start


def print_plan(sim_times=(48, 72, 96, 120, 132, 168, 216)):
    print("Integration windows (all end at 12:00 on the Monday of the sampling week)\n")
    for w, lab in WEEK_LABELS.items():
        end = sampling_reference_time(w)
        print(f"week {w:2d} ({lab}), sampling week starts Mon {end:%d %b}")
        for T in sim_times:
            t0, t1 = integration_window(end, T)
            labels, ok = files_needed(t0, t1)
            note = '' if ok else '   <-- NOT fully covered by the available files'
            print(f"   {T:4.0f} h: {t0:%a %d %b %H:%M} -> {t1:%a %d %b %H:%M}   files: {' + '.join(labels)}{note}")
    print("\nFile names: " + ', '.join(f"{l} = {f}" for l, f in VELOCITY_FILES.items()))


def build_velocity_arrays(frames, t_start, t_end):
    """Join files, keep [t_start, t_end], and grid onto (nt, ny, nx) arrays."""
    lat_min = max(f['latitude'].min() for f in frames)
    lat_max = min(f['latitude'].max() for f in frames)
    lon_min = max(f['longitude'].min() for f in frames)
    lon_max = min(f['longitude'].max() for f in frames)

    df = pd.concat(frames, ignore_index=True)
    df = df[(df['time'] >= t_start) & (df['time'] <= t_end)
            & df['latitude'].between(lat_min, lat_max)
            & df['longitude'].between(lon_min, lon_max)]
    # where consecutive files overlap, keep the later file's values (as before)
    df = df.drop_duplicates(subset=['time', 'latitude', 'longitude'], keep='last')

    times = np.sort(df['time'].unique())
    n_needed = int(round((t_end - t_start) / pd.Timedelta(hours=TIME_STEP_H))) + 1
    if len(times) != n_needed or pd.Timestamp(times[0]) != t_start or pd.Timestamp(times[-1]) != t_end:
        raise ValueError(
            f"Velocity files do not cover the integration window {t_start} -> {t_end} "
            f"at {TIME_STEP_H} h steps (found {len(times)} of {n_needed} time levels, "
            f"{pd.Timestamp(times[0]) if len(times) else None} -> {pd.Timestamp(times[-1]) if len(times) else None}). "
            f"Pass the neighbouring week's file as well.")

    ny = int(round((lat_max - lat_min) / LAT_SPACING)) + 1
    nx = int(round((lon_max - lon_min) / LON_SPACING)) + 1
    it = np.searchsorted(times, df['time'].to_numpy())
    iy = np.rint((df['latitude'].to_numpy() - lat_min) / LAT_SPACING).astype(np.int64)
    ix = np.rint((df['longitude'].to_numpy() - lon_min) / LON_SPACING).astype(np.int64)

    U = np.full((len(times), ny, nx), np.nan, dtype=np.float32)
    V = np.full((len(times), ny, nx), np.nan, dtype=np.float32)
    U[it, iy, ix] = df['sea_surface_x_velocity'].to_numpy()
    V[it, iy, ix] = df['sea_surface_y_velocity'].to_numpy()
    return U, V, lat_min, lon_min


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True)
def interp_vel(t, lon, lat, U, V, dt_time, lat0, dlat, lon0, dlon):
    """Trilinear interpolation. Returns NaN outside the grid or if any of the
    8 surrounding nodes is land (NaN) -- such particles are treated as lost."""
    nt, ny, nx = U.shape
    ft = t / dt_time
    if ft < 0.0:
        ft = 0.0
    if ft > nt - 1.0:
        ft = nt - 1.0
    it = int(ft)
    if it > nt - 2:
        it = nt - 2
    wt = ft - it

    fy = (lat - lat0) / dlat
    fx = (lon - lon0) / dlon
    # Written so that a NaN position also counts as outside: every comparison
    # with NaN is False, so `not (...)` is True. (The form `fy < 0 or ...` let
    # NaN through, int(NaN) became a garbage index, and Numba -- which does no
    # bounds checking -- read outside the array and crashed the process.)
    if not (fy >= 0.0 and fx >= 0.0 and fy <= ny - 1.0 and fx <= nx - 1.0):
        return np.nan, np.nan
    iy = int(fy)
    ix = int(fx)
    if iy > ny - 2:
        iy = ny - 2
    if ix > nx - 2:
        ix = nx - 2
    wy = fy - iy
    wx = fx - ix

    u = 0.0
    v = 0.0
    for a in range(2):
        wa = wt if a == 1 else 1.0 - wt
        for b in range(2):
            wb = wy if b == 1 else 1.0 - wy
            for c in range(2):
                wc = wx if c == 1 else 1.0 - wx
                w = wa * wb * wc
                u += w * U[it + a, iy + b, ix + c]
                v += w * V[it + a, iy + b, ix + c]
    return u, v


@njit(cache=True)
def displace(lon, lat, u, v, dt_s):
    """Move a point by velocity (u, v) [m/s] for dt_s seconds; returns degrees."""
    dlon = u * dt_s / (EARTH_RADIUS_M * math.cos(math.radians(lat))) * 180.0 / math.pi
    dlat = v * dt_s / EARTH_RADIUS_M * 180.0 / math.pi
    return lon + dlon, lat + dlat


@njit(cache=True)
def step(lon, lat, t, dt_h, U, V, dt_time, lat0, dlat, lon0, dlon, method):
    """One integration step. t and dt_h in hours, velocities in m/s."""
    # Any stage that lands on land / outside the grid ends the step with NaN
    # (the particle is then marked lost).
    dt_s = dt_h * 3600.0
    k1u, k1v = interp_vel(t, lon, lat, U, V, dt_time, lat0, dlat, lon0, dlon)
    if np.isnan(k1u) or np.isnan(k1v):
        return np.nan, np.nan
    if method == 1:                                   # forward Euler
        return displace(lon, lat, k1u, k1v, dt_s)

    if method == 2:                                   # Kutta's third-order method
        l2, a2 = displace(lon, lat, k1u, k1v, 0.5 * dt_s)
        k2u, k2v = interp_vel(t + 0.5 * dt_h, l2, a2, U, V, dt_time, lat0, dlat, lon0, dlon)
        if np.isnan(k2u) or np.isnan(k2v):
            return np.nan, np.nan
        l3, a3 = displace(lon, lat, -k1u + 2.0 * k2u, -k1v + 2.0 * k2v, dt_s)
        k3u, k3v = interp_vel(t + dt_h, l3, a3, U, V, dt_time, lat0, dlat, lon0, dlon)
        if np.isnan(k3u) or np.isnan(k3v):
            return np.nan, np.nan
        return displace(lon, lat, (k1u + 4.0 * k2u + k3u) / 6.0,
                        (k1v + 4.0 * k2v + k3v) / 6.0, dt_s)

    # classical RK4
    l2, a2 = displace(lon, lat, k1u, k1v, 0.5 * dt_s)
    k2u, k2v = interp_vel(t + 0.5 * dt_h, l2, a2, U, V, dt_time, lat0, dlat, lon0, dlon)
    if np.isnan(k2u) or np.isnan(k2v):
        return np.nan, np.nan
    l3, a3 = displace(lon, lat, k2u, k2v, 0.5 * dt_s)
    k3u, k3v = interp_vel(t + 0.5 * dt_h, l3, a3, U, V, dt_time, lat0, dlat, lon0, dlon)
    if np.isnan(k3u) or np.isnan(k3v):
        return np.nan, np.nan
    l4, a4 = displace(lon, lat, k3u, k3v, dt_s)
    k4u, k4v = interp_vel(t + dt_h, l4, a4, U, V, dt_time, lat0, dlat, lon0, dlon)
    if np.isnan(k4u) or np.isnan(k4v):
        return np.nan, np.nan
    return displace(lon, lat, (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0,
                    (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0, dt_s)


@njit(parallel=True, cache=True)
def integrate_all(pos0, nsteps, dt_h, U, V, dt_time, lat0, dlat, lon0, dlon, method, freeze):
    """Returns (nsteps+1, Nparticles, 2).

    freeze=True  A step that would hit land or leave the
                 grid leaves the particle where it is; it tries again on the next
                 step, so it can move off once the flow allows. Particles that
                 start on land keep their initial position throughout
    freeze=False The particle is NaN from that step on; particles
                 starting on land are NaN after t=0."""
    n = pos0.shape[0]
    out = np.full((nsteps + 1, n, 2), np.nan)
    for p in prange(n):
        lon = pos0[p, 0]
        lat = pos0[p, 1]
        out[0, p, 0] = lon
        out[0, p, 1] = lat
        u0, v0 = interp_vel(0.0, lon, lat, U, V, dt_time, lat0, dlat, lon0, dlon)
        if np.isnan(u0) or np.isnan(v0):
            if freeze:
                for s in range(nsteps):
                    out[s + 1, p, 0] = lon
                    out[s + 1, p, 1] = lat
            continue
        for s in range(nsteps):
            newlon, newlat = step(lon, lat, s * dt_h, dt_h, U, V, dt_time, lat0, dlat, lon0, dlon, method)
            if np.isnan(newlon) or np.isnan(newlat):
                if not freeze:
                    break
                newlon = lon                     # stay put this step
                newlat = lat
            lon = newlon
            lat = newlat
            out[s + 1, p, 0] = lon
            out[s + 1, p, 1] = lat
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def seed_particles():
    i = np.arange(NPX)
    j = np.arange(NPY)
    lons = X0 - NPX * LON_SPACING * RP / 2.0 + LON_SPACING * RP * i
    lats = Y0 - NPY * LAT_SPACING * RP / 2.0 + LAT_SPACING * RP * j
    LON, LAT = np.meshgrid(lons, lats)               # (NPY, NPX), longitude fastest
    return np.column_stack([LON.ravel(), LAT.ravel()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*',
                    help='One or more consecutive weekly velocity CSVs (chronological order), '
                         'followed by the output file name (saved in data/Particle positions)')
    ap.add_argument('--sim_time', type=float, default=48.0,
                    help='integration time in hours BEFORE 12:00 on the Monday of the sampling week '
                         '(48 = Saturday 12:00 -> Monday 12:00)')
    ap.add_argument('--week', default=None,
                    help="sampling week as an index (1) or label (aug1); default: read from the output file name")
    ap.add_argument('--end', default=None,
                    help='absolute end time, e.g. 2024-08-05T12:00 (overrides the sampling-week rule)')
    ap.add_argument('--plan', action='store_true',
                    help='print the integration window and velocity files needed for every week, then exit')
    ap.add_argument('--dt', type=float, default=DT_DEFAULT_H, help='integration step (h)')
    ap.add_argument('--integrator', choices=list(INTEGRATORS), default='rk3')
    ap.add_argument('--coast', choices=['freeze', 'lost'], default='freeze',
                    help="particles reaching land / the grid edge: 'freeze' = stay in place (original "
                         "behaviour, default); 'lost' = NaN from then on")
    ap.add_argument('--backward', action='store_true', help='integrate backward in time (attracting LCS)')
    ap.add_argument('--animate', action='store_true', help='show an animation when finished')
    args = ap.parse_args()

    if args.plan:
        print_plan()
        return
    if len(args.files) < 2:
        ap.error('give at least one velocity file and an output file name')
    vel_files = [VEL_DIR / f for f in args.files[:-1]]
    output_path = POS_DIR / args.files[-1]

    nsteps = args.sim_time / args.dt
    if abs(nsteps - round(nsteps)) > 1e-9:
        ap.error('sim_time must be a whole number of integration steps')
    nsteps = int(round(nsteps))

    # --- integration window: ends at 12:00 on the Monday of the sampling week
    week = None
    if args.end:
        end = pd.Timestamp(args.end)
    else:
        if args.week is not None:
            week = int(args.week) if str(args.week).isdigit() else week_from_label(args.week)
        else:
            week = week_from_label(output_path.name)
        if week is None or week not in WEEK_LABELS:
            ap.error("cannot tell the sampling week: pass --week (e.g. --week aug1) or --end, "
                     "or include the label in the output name (e.g. pos_2day_aug124.npy)")
        end = sampling_reference_time(week)
    t_start, t_end = integration_window(end, args.sim_time)
    release = t_end if args.backward else t_start
    print(f"Sampling week {WEEK_LABELS.get(week, '?')}: window {t_start:%a %d %b %Y %H:%M} -> "
          f"{t_end:%a %d %b %Y %H:%M} ({args.sim_time:g} h), integrating "
          f"{'backward from' if args.backward else 'forward from'} {release:%a %d %b %H:%M}")

    frames = [read_velocity_csv(f) for f in vel_files]
    try:
        U, V, lat0, lon0 = build_velocity_arrays(frames, t_start, t_end)
    except ValueError as err:
        labels, _ = files_needed(t_start, t_end)
        raise SystemExit(f"{err}\nThis window needs: "
                         + ', '.join(f"{VELOCITY_FILES[l]} ({l})" for l in labels))
    del frames
    if args.backward:
        # dx/dt = u(x,t) backward from t_r  <=>  dx/dtau = -u(x, t_r - tau) forward
        U = np.ascontiguousarray(-U[::-1])
        V = np.ascontiguousarray(-V[::-1])

    pos0 = seed_particles()
    print(f"Integrating {len(pos0)} particles, {nsteps} steps of {args.dt} h ({args.integrator}) ...")
    print("(the first chunk includes Numba compilation, so it is slower)")
    tic = time.time()
    out = np.empty((nsteps + 1, len(pos0), 2))
    n_chunks = 20
    bounds = np.linspace(0, len(pos0), n_chunks + 1).astype(int)
    for k in range(n_chunks):
        a, b = bounds[k], bounds[k + 1]
        out[:, a:b] = integrate_all(pos0[a:b], nsteps, args.dt, U, V, TIME_STEP_H,
                                    lat0, LAT_SPACING, lon0, LON_SPACING, INTEGRATORS[args.integrator],
                                    args.coast == 'freeze')
        el = time.time() - tic
        print(f"  {100 * (k + 1) / n_chunks:3.0f}% of particles done, {el:6.0f} s elapsed, "
              f"~{el / (k + 1) * (n_chunks - k - 1):5.0f} s left", flush=True)
    print(f"Integration complete in {time.time() - tic:.1f} s")

    pos = out.reshape((nsteps + 1, NPY, NPX, 2))
    if args.coast == 'freeze':
        never_moved = np.all(pos[-1] == pos[0], axis=-1) & np.all(pos[1] == pos[0], axis=-1)
        stuck_at_end = np.all(pos[-1] == pos[-2], axis=-1) & ~never_moved
        print(f"Particles that never moved (land / edge): {100 * never_moved.mean():.1f}%; "
              f"held at the coast or edge at the final step: "
              f"{100 * stuck_at_end.sum() / max((~never_moved).sum(), 1):.1f}% of moving particles")
    else:
        final_valid = ~np.isnan(pos[-1, ..., 0])
        started_ocean = ~np.isnan(pos[1, ..., 0]) if nsteps >= 1 else final_valid
        print(f"Particles starting on land: {100 * (1 - started_ocean.mean()):.1f}%; "
              f"lost during integration (coast / domain edge): "
              f"{100 * (started_ocean & ~final_valid).sum() / max(started_ocean.sum(), 1):.1f}% of ocean particles")

    np.save(output_path, pos)
    meta = {
        'velocity_files': [f.name for f in vel_files],
        'sampling_week': WEEK_LABELS.get(week),
        'reference_time': str(t_end),
        'timing_rule': 'window ends at 12:00 on the Monday of the sampling week' if not args.end else 'explicit --end',
        'release_time': str(release),
        'direction': 'backward' if args.backward else 'forward',
        'window_start': str(t_start), 'window_end': str(t_end),
        'T_hours': nsteps * args.dt,
        'dt_hours': args.dt, 'nsteps': nsteps, 'integrator': args.integrator,
        'seeding': {'npx': NPX, 'npy': NPY, 'rp': RP, 'x0': X0, 'y0': Y0},
        'coast_mode': args.coast,
        'coast_rule': ('particles reaching land / grid edge stay in place for that step; land particles keep '
                       'their initial position' if args.coast == 'freeze'
                       else 'particles reaching land / grid edge are NaN from then on'),
    }
    metadata_path(output_path).write_text(json.dumps(meta, indent=2))
    print(f"Saved {output_path} (+ .json metadata)")

    if args.animate:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        pts = ax.scatter(pos[0, ..., 0].ravel(), pos[0, ..., 1].ravel(), s=0.1, c='k')
        ax.set_xlim(np.nanmin(pos[0, ..., 0]), np.nanmax(pos[0, ..., 0]))
        ax.set_ylim(np.nanmin(pos[0, ..., 1]), np.nanmax(pos[0, ..., 1]))
        txt = ax.set_title('')

        def update(k):
            pts.set_offsets(pos[k].reshape(-1, 2))
            txt.set_text(f'{k * args.dt:.1f} h')
            return pts,
        _anim = FuncAnimation(fig, update, frames=range(0, nsteps + 1, max(1, nsteps // 50)), interval=200)
        plt.show()


if __name__ == '__main__':
    main()
