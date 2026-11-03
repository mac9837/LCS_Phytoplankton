"""
Map thresholded FTLE ridges (candidate repelling LCS) with the phytoplankton
sites.

Workflow
--------
1. python lcs_visualisation2.py             -> images in results/LCS images
2. Score each image: one 0/1 per numbered site pair (0, 1, 2, ... south -> north),
   entered as that week's list in legacy_lcs_scores.py
3. python lcs_bc_analysis.py
"""
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
import warnings

# Hide library noise that is not about this script: matplotlib's maths-text
# parser warns on newer pyparsing versions, and Cartopy announces downloads.
warnings.filterwarnings('ignore', module=r'matplotlib\._mathtext')
warnings.filterwarnings('ignore', module=r'matplotlib\._fontconfig_pattern')
warnings.filterwarnings('ignore', message=r'Downloading: ')

import numpy as np
import matplotlib
matplotlib.use('Agg')                    # images are written to files, not shown
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from lcs_common2 import (WEEK_LABELS, SITE_COORDS, LCS_IMAGE_DIR,
                        ftle_path, metadata_path, load_phyto, select_samples, week_pairs, seed_axes,
                        FTLE_DIR, ftle_names)
from scipy import ndimage

try:                                     # pip install cartopy   (replaces the deprecated Basemap)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:                      # plots still work, just without coastlines
    ccrs = cfeature = None
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIMESCALES = [2, 3, 4, 5, 7, 9]
WEEKS = list(range(0, 11))
THRESHOLD_PERCENTILE = 94       # same threshold for every week and timescale
DPI = 300
COASTLINE = 'gshhs_f'           # 'gshhs_f' (full resolution, best for narrow bays), 'gshhs_h',
                                # or 'ne_10m' (Natural Earth). Downloaded once on first use.


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
FAILED_READS = []


def _load_npy(path, attempts=3, **kw):
    for k in range(attempts):
        try:
            return np.load(path, **kw)
        except OSError as err:
            last = err
            time.sleep(2 * (k + 1))
    raise OSError(f"cannot read {Path(path).name} ({last})")


_NOTED = set()


def load_fields(timescale, week):
    """Return lons (nx,), lats (ny,), FTLE (ny, nx) with land / invalid cells as NaN.
    Only the FTLE file and its .json are read -- never the large particle file."""
    ftle = _load_npy(ftle_path(timescale, week)).astype(float)
    meta_file = metadata_path(ftle_path(timescale, week))
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    meta.setdefault('coast_mode', 'freeze')

    if 'grid' in meta:
        # current ftle_extraction.py: grid stored, land already NaN
        g = meta['grid']
        lons = g['lon_start'] + g['lon_step'] * np.arange(g['nx'])
        lats = g['lat_start'] + g['lat_step'] * np.arange(g['ny'])
    else:
        lons, lats = seed_axes()
        if 'old' not in _NOTED:
            print("note: FTLE file(s) without stored grid/land mask -- grid rebuilt from the "
                  "seeding settings and land detected from the FTLE. Re-running ftle_extraction.py "
                  "stores both exactly.", flush=True)
            _NOTED.add('old')
        with np.errstate(invalid='ignore'):
            still = np.abs(ftle) < 1e-9
        land = ndimage.binary_dilation(still, structure=ndimage.generate_binary_structure(2, 1))
        ftle[land] = np.nan
    if ftle.shape != (len(lats), len(lons)):
        raise ValueError(f"FTLE shape {ftle.shape} does not match its grid ({len(lats)} x {len(lons)}) "
                         f"for {timescale}-day week {week}")
    return lons, lats, ftle, meta


def ridge_mask(ftle, percentile=THRESHOLD_PERCENTILE):
    thr = np.nanpercentile(ftle, percentile)
    with np.errstate(invalid='ignore'):
        return ftle > thr, thr


def label_for(timescale, meta):
    T = meta.get('T_hours')
    return f"{T / 24:g}-day" if T else f"{timescale}-day"


def _ftle_meta(timescale, week):
    meta_file = metadata_path(ftle_path(timescale, week))
    return json.loads(meta_file.read_text()) if meta_file.exists() else {}


def labelled_title(timescale, week):
    return f"Week {week} ({WEEK_LABELS[week]}), {label_for(timescale, _ftle_meta(timescale, week))} integration"


def length_mismatch(timescale, week):
    """True if the integration time stored with the FTLE file is more than
    0.75 days away from the timescale in its file name """
    T_hours = _ftle_meta(timescale, week).get('T_hours')
    return T_hours is not None and abs(T_hours / 24 - timescale) > 0.75


def labelled_image(timescale, week):
    """Descriptive file name, e.g. LCS_4day_aug2.png (4.5-day -> LCS_4.5day_jul4.png).
    If the stored integration time disagrees with the file name, both are given
    (LCS_5day_aug1_from4dayfiles.png) so it cannot overwrite the real 5-day image."""
    days = label_for(timescale, _ftle_meta(timescale, week)).replace('-', '')
    extra = f"_from{timescale}dayfiles" if length_mismatch(timescale, week) else ""
    return f"LCS_{days}_{WEEK_LABELS[week]}{extra}.png"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_LAND = {}


def land_feature(extent):
    """Coastline/land polygons for the map, trying the configured source first.
    Each source is fetched once and cached by
    Cartopy; if none can be loaded the maps are drawn without coastlines."""
    if 'feature' in _LAND:
        return _LAND['feature']
    feature = None
    if cfeature is not None:
        order = [COASTLINE] + [c for c in ('gshhs_h', 'ne_10m') if c != COASTLINE]
        for src in order:
            try:
                if src.startswith('gshhs'):
                    f = _gshhs(src[-1], extent)
                else:
                    f = cfeature.NaturalEarthFeature('physical', 'land', '10m')
                    next(iter(f.intersecting_geometries(extent)), None)   # forces the download now
                feature = f
                if src != COASTLINE:
                    print(f"note: using '{src}' coastline ('{COASTLINE}' unavailable)")
                break
            except Exception as err:                                  # offline, blocked, etc.
                print(f"note: coastline '{src}' unavailable ({type(err).__name__}: {err})")
    if feature is None:
        print("note: drawing maps without coastlines")
    _LAND['feature'] = feature
    return feature


GSHHS_ZIP_URL = ('https://github.com/GenericMappingTools/gshhg-gmt/releases/download/'
                 '2.3.7/gshhg-shp-2.3.7.zip')


def _gshhs(scale, extent):
    """GSHHS land polygons. Older Cartopy versions download from NOAA addresses
    that now return 404; if that happens, fetch the same archive from the GSHHG
    GitHub release and unpack it where Cartopy looks, then try again."""
    feature = cfeature.GSHHSFeature(scale=scale, levels=[1])
    try:
        next(iter(feature.intersecting_geometries(extent)), None)
        return feature
    except Exception as err:
        print(f"note: Cartopy could not download GSHHS ({type(err).__name__}: {err}); "
              "trying the GSHHG GitHub release instead")
    import cartopy
    import cartopy.io.shapereader  # noqa: F401  (registers the GSHHS downloader)
    downloader = cartopy.config['downloaders'].get(('shapefiles', 'gshhs'))
    try:
        target = Path(downloader.target_path({'config': cartopy.config, 'scale': scale, 'level': 1}))
    except Exception:                           # standard Cartopy layout
        target = Path(cartopy.config['data_dir']) / 'shapefiles' / 'gshhs' / scale / f'GSHHS_{scale}_L1.shp'
    target.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target.parent / 'gshhg-shp-2.3.7.zip'
    if not zip_path.exists():
        print(f"downloading coastline archive (~150 MB, once only) to {target.parent} ...", flush=True)
        part = zip_path.with_suffix('.part')
        with urllib.request.urlopen(GSHHS_ZIP_URL, timeout=120) as r, open(part, 'wb') as out:
            shutil.copyfileobj(r, out)
        part.replace(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for ext in ('.shp', '.shx', '.dbf', '.prj'):
            member = f'GSHHS_shp/{scale}/GSHHS_{scale}_L1{ext}'
            if member in z.namelist():
                with z.open(member) as src, open(target.with_suffix(ext), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
    feature = cfeature.GSHHSFeature(scale=scale, levels=[1])
    next(iter(feature.intersecting_geometries(extent)), None)
    return feature


FAILED_SAVES = []


def save_figure(fig, out_file, attempts=5):
    """Save via a temporary file, retrying for a few seconds. On Windows a PNG
    that is open in an image viewer, or being synced by OneDrive, cannot be
    overwritten ('Invalid argument' / 'Permission denied'). A file that still
    cannot be written is reported and skipped instead of stopping the run."""
    out_file = Path(out_file)
    tmp = out_file.with_name(f"~tmp_{out_file.name}")
    for k in range(attempts):
        try:
            fig.savefig(tmp, dpi=DPI, bbox_inches='tight')
            os.replace(tmp, out_file)
            return True
        except OSError as err:
            last = err
            time.sleep(1 + k)
    try:
        tmp.unlink()
    except OSError:
        pass
    print(f"FAILED to save {out_file.name}: {last}. Is it open in an image viewer, or is OneDrive "
          "syncing that folder? Close it and re-run.")
    FAILED_SAVES.append(out_file.name)
    return False


def report_failed_saves():
    if FAILED_SAVES:
        print(f"\n{len(FAILED_SAVES)} image(s) could not be saved: {', '.join(FAILED_SAVES)}")
    if FAILED_READS:
        print(f"\n{len(FAILED_READS)} field(s) could not be read and were skipped: {', '.join(FAILED_READS)}\n"
              "These files are most likely still being synced by OneDrive, or are 'online-only'. "
              "Right-click the data folder > 'Always keep on this device', wait for the sync to "
              "finish, then re-run.")


def map_axes(fig, *subplot_args):
    """A map axes: Cartopy if installed, else a plain axes."""
    if ccrs is not None:
        return fig.add_subplot(*subplot_args, projection=ccrs.PlateCarree())
    return fig.add_subplot(*subplot_args)


def draw_base(ax, lons, lats, ftle, ridges):
    dx, dy = lons[1] - lons[0], lats[1] - lats[0]
    extent = [lons[0] - dx / 2, lons[-1] + dx / 2, lats[0] - dy / 2, lats[-1] + dy / 2]
    img = np.zeros(ftle.shape)                    # 0 ocean, 1 ridge, 2 FTLE undefined
    img[ridges] = 1
    img[np.isnan(ftle)] = 2
    cmap = mcolors.ListedColormap(['#dbeeff', 'black', '#bdbdbd'])
    if ccrs is not None:
        pc = ccrs.PlateCarree()
        ax.imshow(img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest', origin='lower',
                  extent=extent, transform=pc, zorder=1)
        land = land_feature(extent)
        if land is not None:
            ax.add_feature(land, facecolor='#7aa874', edgecolor='none', zorder=2)
            # ridges again ABOVE the land fill, so ridges in model-ocean cells that the
            # (finer) coastline calls land -- narrow inlets -- are not hidden
            ax.imshow(np.ma.masked_where(~ridges, ridges.astype(float)),
                      cmap=mcolors.ListedColormap(['black']), interpolation='nearest',
                      origin='lower', extent=extent, transform=pc, zorder=2.5)
            ax.add_feature(land, facecolor='none', edgecolor='black', linewidth=0.4, zorder=3)
        ax.set_extent(extent, crs=pc)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='grey', linestyle=':')
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 7}
    else:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest', origin='lower', extent=extent)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    # true shape: 1 deg of longitude is cos(lat) times shorter than 1 deg of latitude
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lats))))


def plot_week(timescale, week, sites, pairs, title, out_file):
    """Returns True if saved, False if the image could not be written, and
    None if the input files could not be read (field skipped)."""
    try:
        lons, lats, ftle, meta = load_fields(timescale, week)
    except OSError as err:
        print(f"SKIPPED {timescale}-day {WEEK_LABELS[week]}: {err}", flush=True)
        FAILED_READS.append(f"{timescale}-day {WEEK_LABELS[week]}")
        return None
    ridges, _ = ridge_mask(ftle)

    fig = plt.figure(figsize=(8, 10))
    ax = map_axes(fig, 1, 1, 1)
    draw_base(ax, lons, lats, ftle, ridges)
    ax.set_title(title, fontsize=10)

    for k, (a, b) in enumerate(pairs):            # numbered pair segments
        (x1, y1), (x2, y2) = SITE_COORDS[a], SITE_COORDS[b]
        ax.plot([x1, x2], [y1, y2], color='orange', lw=0.8, zorder=4)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(k), fontsize=6, color='darkorange',
                ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.8))
    xs = [SITE_COORDS[s][0] for s in sites]
    ys = [SITE_COORDS[s][1] for s in sites]
    ax.scatter(xs, ys, s=12, color='lime', edgecolor='k', linewidth=0.4, zorder=5)
    texts = [ax.text(x + 0.02, y, s, fontsize=6, color='darkgreen', zorder=6) for s, x, y in zip(sites, xs, ys)]
    if adjust_text is not None:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='darkgreen', lw=0.4))

    ok = save_figure(fig, out_file)
    plt.close(fig)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    LCS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    phyto = load_phyto()
    jobs, missing = [], {}
    for T in TIMESCALES:
        for week in WEEKS:
            if not ftle_path(T, week).exists():
                missing.setdefault(T, []).append(WEEK_LABELS[week])
                continue
            dfw = select_samples(phyto[phyto['week_idx'] == week], verbose=False)
            if dfw['site'].nunique() < 2:
                continue
            jobs.append((T, week, dfw))
    if missing:
        print(f"FTLE files not found in {FTLE_DIR} (these fields are skipped):")
        for T, weeks in missing.items():
            which = 'all weeks' if len(weeks) == len(WEEKS) else ', '.join(weeks)
            print(f"  {T}-day: {which}   (looked for {ftle_names(T)})")
    mismatched = [(T, week) for T, week, _ in jobs if length_mismatch(T, week)]
    if mismatched:
        print("WARNING: these files' stored integration time does not match their file name "
              "-- check that the right runs are saved under the right names:")
        for T, week in mismatched:
            print(f"  ftle_{T}day_{WEEK_LABELS[week]}: stored T = {_ftle_meta(T, week)['T_hours']:g} h")
    if not jobs:
        raise SystemExit("No matching FTLE files found -- check the patterns in lcs_common.py")

    for T, week, dfw in jobs:
        pairs = week_pairs(dfw)
        sites = [pairs[0][0]] + [b for _, b in pairs]

        title = labelled_title(T, week)
        image = labelled_image(T, week)
        ok = plot_week(T, week, sites, pairs, title, LCS_IMAGE_DIR / image)
        if ok:
            print(f"saved {image}  ({len(pairs)} pairs)", flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        report_failed_saves()
