"""
Shortest sea-path distance (avoiding land) between every pair of sampling sites.

Method: the full-resolution GSHHS coastline is rasterised onto a fine grid
(GRID_DEG, about 200 x 300 m at 53N). Sea cells connected to the open ocean form
a graph with 8-neighbour moves (no cutting across land corners); each site is
moved to its nearest open-sea cell (the distance moved is reported); Dijkstra's
algorithm gives the shortest path through water between each pair of sites.

Output: data/sea_path_distances.csv  (site_a, site_b, sea_km, straight_km,
snap_a_km, snap_b_km) for every pair of sites in SITES. Pairs are unordered.
lcs_bc_analysis.py reads it with --distance sea (the default).

Usage:  python sea_path_distances.py           (first run downloads the coastline once)
"""
import math
import time

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from lcs_common2 import SITES, SITE_COORDS, DATA_DIR, EARTH_RADIUS_M, haversine_km

GRID_DEG = 0.0025                     # grid spacing (deg)
EXTENT = (-11.0, -7.8, 51.2, 55.4)    # lon_min, lon_max, lat_min, lat_max (covers all sites)
MAX_SNAP_KM = 5.0
OUT = DATA_DIR / 'sea_path_distances.csv'
DEG2M = math.pi / 180 * EARTH_RADIUS_M


def land_mask(lons, lats):
    import shapely
    from shapely.geometry import box
    from lcs_visualisation2 import land_feature
    ext = [EXTENT[0] - 0.2, EXTENT[1] + 0.2, EXTENT[2] - 0.2, EXTENT[3] + 0.2]
    feat = land_feature(ext)
    if feat is None:
        raise SystemExit("Coastline data unavailable (see the note above); cannot compute sea paths.")
    clip = box(ext[0], ext[2], ext[1], ext[3])
    geoms = [g.intersection(clip) for g in feat.intersecting_geometries(ext)]
    land = shapely.union_all([g for g in geoms if not g.is_empty])
    shapely.prepare(land)
    LON, LAT = np.meshgrid(lons, lats)
    return shapely.contains_xy(land, LON, LAT)


def sea_graph(sea, dx_rows, dy):
    ny, nx = sea.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    r, c, w = [], [], []

    def add(m, a, b, wt):
        r.append(a[m]); c.append(b[m]); w.append(np.broadcast_to(wt, m.shape)[m])
    dx = dx_rows[:, None]
    add(sea[:, :-1] & sea[:, 1:], idx[:, :-1], idx[:, 1:], dx)
    add(sea[:-1] & sea[1:], idx[:-1], idx[1:], np.array(dy))
    diag = np.sqrt((0.5 * (dx[:-1] + dx[1:])) ** 2 + dy ** 2)
    both = sea[:-1, :-1] & sea[1:, 1:] & sea[:-1, 1:] & sea[1:, :-1]      # no corner cutting
    add(both, idx[:-1, :-1], idx[1:, 1:], diag)
    add(both, idx[:-1, 1:], idx[1:, :-1], diag)
    return coo_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))),
                      shape=(ny * nx, ny * nx)).tocsr()


def main():
    tic = time.time()
    lons = np.arange(EXTENT[0], EXTENT[1] + 1e-9, GRID_DEG)
    lats = np.arange(EXTENT[2], EXTENT[3] + 1e-9, GRID_DEG)
    print(f"Rasterising coastline onto {len(lats)} x {len(lons)} grid ({GRID_DEG} deg) ...", flush=True)
    land = land_mask(lons, lats)
    lab, n = ndimage.label(~land)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    sea = lab == sizes.argmax()                        # water connected to the open ocean
    dy = GRID_DEG * DEG2M
    dx_rows = GRID_DEG * DEG2M * np.cos(np.radians(lats))
    G = sea_graph(sea, dx_rows, dy)
    dist_to_sea, inds = ndimage.distance_transform_edt(~sea, sampling=(dy, float(dx_rows.mean())), return_indices=True)

    names = [s[0] for s in SITES]
    cells, snaps = {}, {}
    for s in names:
        lon, lat = SITE_COORDS[s]
        j, i = int(round((lat - lats[0]) / GRID_DEG)), int(round((lon - lons[0]) / GRID_DEG))
        snaps[s] = dist_to_sea[j, i] / 1000
        cells[s] = (int(inds[0, j, i]), int(inds[1, j, i])) if snaps[s] <= MAX_SNAP_KM else None
    far = {s: round(v, 2) for s, v in snaps.items() if v > 0.5}
    if far:
        print(f"Sites moved more than 0.5 km to reach connected open water: {far}")

    nx = len(lons)
    rows = []
    for k, a in enumerate(names):
        if cells[a] is None:
            continue
        d = dijkstra(G, directed=False, indices=cells[a][0] * nx + cells[a][1])
        for b in names[k + 1:]:
            if cells[b] is None:
                continue
            sea_km = d[cells[b][0] * nx + cells[b][1]] / 1000
            rows.append(dict(site_a=a, site_b=b, sea_km=round(sea_km, 2), straight_km=round(haversine_km(a, b), 2),
                             snap_a_km=round(snaps[a], 2), snap_b_km=round(snaps[b], 2)))
    out = pd.DataFrame(rows)
    bad = out[~np.isfinite(out.sea_km)]
    if len(bad):
        print(f"WARNING: {len(bad)} pair(s) not connected by water on this grid")
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} site pairs to {OUT}  ({time.time() - tic:.0f} s)")


if __name__ == '__main__':
    main()
