"""
Shared configuration for the LCS / phytoplankton pipeline.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
VEL_DIR = DATA_DIR / 'Weekly velocity fields'
POS_DIR = DATA_DIR / 'Particle positions 2'
FTLE_DIR = DATA_DIR / 'FTLE extracted 2'
PHYTO_CSV = DATA_DIR / 'phytoplankton_8weeksdata.csv'
RESULTS_DIR = BASE_DIR / 'results 2'
LCS_IMAGE_DIR = RESULTS_DIR / 'LCS images 2'
SCORING_DIR = RESULTS_DIR / 'LCS scoring 2'
AUTO_DIR = RESULTS_DIR / 'LCS auto'

# ---------------------------------------------------------------------------
# Grid / integration constants
# ---------------------------------------------------------------------------
LAT_SPACING = 0.0125    # velocity grid latitude spacing (deg)
LON_SPACING = 0.0125    # velocity grid longitude spacing (deg)
TIME_STEP_H = 1.0       # velocity file time step (h)
DT_DEFAULT_H = 0.5      # default integration step (h)
EARTH_RADIUS_M = 6371000.0

# Particle seeding rectangle (used by the tracking, and to rebuild the FTLE grid
# for FTLE files that do not store it)
SEED_NPX = 528           # particles in x (even)
SEED_NPY = 720           # particles in y (even)
SEED_RP = 0.5            # particle spacing as a fraction of the velocity grid spacing
SEED_X0, SEED_Y0 = -9.65, 52.95   # centre of the seeding rectangle


def seed_axes():
    """Longitudes (nx,) and latitudes (ny,) of the particle seed grid = FTLE grid."""
    i, j = np.arange(SEED_NPX), np.arange(SEED_NPY)
    lons = SEED_X0 - SEED_NPX * LON_SPACING * SEED_RP / 2.0 + LON_SPACING * SEED_RP * i
    lats = SEED_Y0 - SEED_NPY * LAT_SPACING * SEED_RP / 2.0 + LAT_SPACING * SEED_RP * j
    return lons, lats

# ---------------------------------------------------------------------------
# Weeks
# ---------------------------------------------------------------------------
# analysis week index -> label used in file names
WEEK_LABELS = {
    0: 'jul4', 1: 'aug1', 2: 'aug2', 3: 'aug3', 4: 'aug4',
    5: 'sep1', 6: 'sep2', 7: 'sep3', 8: 'sep4',
    9: 'oct1', 10: 'oct2', 12: 'oct4',
}
PHYTO_WEEK_OFFSET = 31   # 'Week' column in the phytoplankton CSV = analysis week + 31
SAMPLING_YEAR = 2024     # 'Week' is taken to be the ISO week number in this year
REFERENCE_HOUR = 12      # integrations END at 12:00 on the Monday of the sampling week


def sampling_reference_time(week_idx):
    """12:00 on the Monday of the sampling week for an analysis week index
    (e.g. week 1 -> ISO week 32 -> Monday 5 Aug 2024 12:00)."""
    monday = pd.Timestamp.fromisocalendar(SAMPLING_YEAR, week_idx + PHYTO_WEEK_OFFSET, 1)
    return monday + pd.Timedelta(hours=REFERENCE_HOUR)


def week_from_label(text):
    """Analysis week index from a label such as 'aug1' or a file name such as
    'pos_9day_aug124.npy'. Returns None if no unique label is found."""
    hits = [w for w, lab in WEEK_LABELS.items() if f"{lab}24" in text or text == lab]
    return hits[0] if len(hits) == 1 else None

# Velocity files (each covers 00:00 Wednesday -> 00:00 Wednesday)
VELOCITY_FILES = {
    'jul4': 'IMI_NEATL_ab09_9d57_f28d.csv',   # starts 25/7/24 (Thursday)
    'aug1': 'IMI_NEATL_673e_f015_5722.csv',
    'aug2': 'IMI_NEATL_1a1a_51f2_bdb6.csv',
    'aug3': 'IMI_NEATL_32ff_702e_32a1.csv',
    'aug4': 'IMI_NEATL_99bf_1fd6_3f63.csv',
    'sep1': 'IMI_NEATL_ad2e_896b_d298.csv',
    'sep2': 'IMI_NEATL_ca4b_6cd3_a2c7.csv',
    'sep3': 'IMI_NEATL_c743_a77e_a398.csv',
    'sep4': 'IMI_NEATL_5948_bc96_b9a4.csv',
    'oct1': 'IMI_NEATL_3958_9580_c085.csv',   # starts 25/9/24 08:00
    'oct2': 'IMI_NEATL_a058_d706_3951.csv',
    'oct4': 'IMI_NEATL_d2e7_ae28_4910.csv',
}

# ---------------------------------------------------------------------------
# Particle-position and FTLE file names
# ---------------------------------------------------------------------------
# Each timescale lists the accepted names, tried in order; the first that exists
# is used. The 3-day runs from the old scripts were called 'pos2_3day...' /
# 'ftle2_3day...', so that name is accepted as well as the standard one.
POS_PATTERNS = {
    2: ['pos_2day_{week}24.npy'],
    3: ['pos_3day_{week}24.npy', 'pos2_3day_{week}24.npy'],
    4: ['pos_4day_{week}24.npy'],
    5: ['pos_5day_{week}24.npy'],
    7: ['pos_7day_{week}24.npy'],
    9: ['pos_9day_{week}24.npy'],
}
FTLE_PATTERNS = {
    2: ['ftle_2day_{week}24.npy'],
    3: ['ftle_3day_{week}24.npy', 'ftle2_3day_{week}24.npy'],
    4: ['ftle_4day_{week}24.npy'],
    5: ['ftle_5day_{week}24.npy'],
    7: ['ftle_7day_{week}24.npy'],
    9: ['ftle_9day_{week}24.npy'],
}


def _first_existing(folder, patterns, week_idx):
    paths = [folder / p.format(week=WEEK_LABELS[week_idx]) for p in patterns]
    return next((p for p in paths if p.exists()), paths[0])


def pos_path(timescale, week_idx):
    return _first_existing(POS_DIR, POS_PATTERNS[timescale], week_idx)


def ftle_path(timescale, week_idx):
    return _first_existing(FTLE_DIR, FTLE_PATTERNS[timescale], week_idx)


def ftle_names(timescale):
    """The accepted FTLE file names for a timescale (for messages)."""
    return ' or '.join(p.format(week='<week>') for p in FTLE_PATTERNS[timescale])


def metadata_path(npy_path):
    """Sidecar JSON written next to every positions / FTLE file."""
    return Path(npy_path).with_suffix('.json')


# ---------------------------------------------------------------------------
# Sites, ordered south -> north along the coast
# (short name, [sampling-point names: preferred first, then aliases], lon, lat)
#
# ---------------------------------------------------------------------------
SITES = [
    ("Oysterh'n", ['Oysterhaven (CK-ON-ON)'], -8.450718, 51.687882),
    ('Sherkin W', ['Kinish (CK-SW-KH)'], -9.437943, 51.485338),
    ('Sherkin E', ['Sherkin North (CK-SN-SN)'], -9.368076, 51.523057),
    ('Roaringw.', ['Roaringwater Bay (CK-RB-RB)', 'Carrigviglash (CK-RB-CH)', 'Reenmurrish (CK-RB-RH)'], -9.432449, 51.532882),
    ('Dunmanus', ['Dunmanus Inner (CK-DB-DI)', 'Dunmanus Bay (CK-DB-DB)', 'Dunmanus Outer (CK-DB-DO)'], -9.746246, 51.542492),
    ('Bantry A', ['South Chapel (CK-BS-SC)'], -9.473305, 51.685222),
    ('Bantry B', ['Newtown (CK-NN-NN)'], -9.454851, 51.695544),
    ('Bantry C', ['North Chapel (CK-BM-NC)', 'Whiddy Point (CK-BM-WP)'], -9.471846, 51.708576),
    ('Bantry D', ['Snave (CK-BN-SE)'], -9.477425, 51.715331),
    ('Bantry E', ['Glengarriff (CK-GF-GF)'], -9.535789, 51.728464),
    ('Bantry F', ['Adrigole (CK-AE-AE)'], -9.716635, 51.673087),
    ('Bantry G', ['Castletownbere (CK-CE-CE)', 'Beare Island (CK-CE-BI)'], -9.827785, 51.649555),
    ('Kenmare A', ['Cleandra (CK-CA-CA)'], -9.930611, 51.752115),
    ('Kenmare B', ['Ardgroom (CK-AM-AM)'], -9.864349, 51.761040),
    ('Kenmare C', ['Kilmakilloge (KY-KE-KE)'], -9.827614, 51.777612),
    ('Kenmare D', ['Templenoe (KY-KI-TE)'], -9.679642, 51.855503),
    ('Kenmare E', ['Tahilla (KY-TA-TA)'], -9.820404, 51.822410),
    ('Valentia', ['Portmagee Channel (KY-VR-PC)', 'Cahirciveen (KY-VR-CN)'], -10.322685, 51.944265),
    ('Dingle', ['Glenbeigh (KY-CH-GH)', 'Banc Fluich (KY-CH-BF)', 'Cromane East (KY-CH-CE)'], -9.843750, 52.146552),
    ('Tralee', ['Fenit (KY-TB-FT)'], -9.943314, 52.327367),
    ('Ballylong.', ['Ballylongford (KY-BD-BD)'], -9.567375, 52.577393),
    ("Poulnas'y", ['Poulnasherry (CE-PY-PY)'], -9.558449, 52.658685),
    ('Carrigaholt', ['Carrigaholt (CE-CT-CT)'], -9.667969, 52.606592),
    ('Ballyv.', ['Poulnaclough (CE-BN-PH)', 'Ballyvaughan (CE-BN-BN)'], -9.150581, 53.132354),
    ('Aughinish', ['Aughinish (CE-AH-AH)'], -9.042091, 53.157271),
    ('Kinvarra', ['Kinvarra (GY-KA-KA)'], -8.959522, 53.167665),
    ('Clarinb.', ['Killeenaran (GY-CE-KN)', 'Island eddy (GY-CE-IE)'], -8.978920, 53.201406),
    ('Tawin', ['Mweeloon Bay (GY-MW-MB)'], -9.014711, 53.216159),
    ('Kilkieran', ['Rosmuc (GY-KN-RC)'], -9.701614, 53.335178),
    ('Clifden S', ['Mannin (GY-MN-MN)'], -10.083046, 53.459642),
    ('Clifden M', ['Cliffden Outer (GY-CN-CO)', 'Ardbear (GY-CN-AR)'], -10.102787, 53.491620),
    ('Clifden N', ['Gubbaros (GY-SB-GS)'], -10.140038, 53.522146),
    ('Ballynakill', ['Ballynakill (GY-BL-BL)'], -10.053864, 53.586391),
    ('Killary I', ['Killary Harbour Inner (GY-KI-KI)'], -9.779549, 53.600044),
    ('Killary M', ['Killary Harbour Middle (GY-KM-KM)'], -9.809761, 53.610636),
    ('Killary O', ['Killary Harbour Outer (GY-KO-KO)'], -9.841690, 53.620819),
    ('Clew S', ['Rosmoney (MO-CS-RY)', 'Clew Bay South (MO-CS-CS)', 'Murrisk (MO-CS-MK)'], -9.645653, 53.811599),
    ('Clew N', ['Inishlaughill (MO-CN-IL)', 'Inisquirk (MO-CN-IK)', 'Clew Bay North (MO-CN-LN)', 'Clew Bay North (MO-CN-CN)'], -9.715347, 53.884107),
    ('Achill S', ['Corraun (MO-AS-CN)'], -9.937134, 53.888356),
    ('Achill N', ['Dooriel Creek (MO-AN-DC)', 'Bealacragher (MO-AN-BR)', 'Bunacurry (MO-AN-BY)'], -9.941940, 54.023705),
    ('Blacksod', ['Elly Bay (MO-BB-EB)', 'Belmullet (MO-BB-BT)'], -10.041847, 54.153991),
    ('Kilalla', ['Killala Bay (MO-KB-KB)'], -9.160194, 54.221088),
    ('Sligo', ['Sligo Harbour (SO-SH-SH)'], -8.537521, 54.295223),
    ('Drumcliff', ['Drumcliff Bay (SO-DB-DB)'], -8.577919, 54.334743),
    ('Donegal E', ['Mountcharles (DL-DH-MS)', 'Donegal Harbour (DL-DH-DH)', 'Laghy Channel (DL-DH-LC)'], -8.225327, 54.614033),
    ('Donegal W', ['McSwynes Bay (DL-BS-MS)', 'Bruckless (DL-BS-BS)'], -8.428345, 54.613635),
    ('Gweebarra S', ['Loughras Beg (DL-LB-LB)'], -8.564758, 54.781946),
    ('Gweebarra M', ['Gweebarra (DL-GA-GA)'], -8.396988, 54.858430),
    ('Gweebarra N', ['Tra Eanach (DL-TE-TE)'], -8.384399, 54.881084),
    ('Dungloe', ['Dungloe South (DL-DE-DS)'], -8.411865, 54.953700),
    ('Gweedore', ['Carrickfin (DL-GE-CN)'], -8.327179, 55.077122),
]

SITE_ORDER = {s[0]: k for k, s in enumerate(SITES)}
SITE_COORDS = {s[0]: (s[2], s[3]) for s in SITES}
POINT_TO_SITE = {}
POINT_PRIORITY = {}
for _short, _names, _lon, _lat in SITES:
    for _k, _name in enumerate(_names):
        POINT_TO_SITE[_name] = _short
        POINT_PRIORITY[_name] = _k


def haversine_km(a, b):
    """Great-circle distance between two site short names (km)."""
    (lon1, lat1), (lon2, lat2) = SITE_COORDS[a], SITE_COORDS[b]
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M / 1000 * np.arcsin(np.sqrt(h))


# ---------------------------------------------------------------------------
# Phytoplankton data
# ---------------------------------------------------------------------------
def load_phyto(path=PHYTO_CSV):
    """Load the phytoplankton CSV and attach site short names and week index."""
    df = pd.read_csv(path)
    unknown = sorted(set(df['Sampling Point']) - set(POINT_TO_SITE))
    if unknown:
        warnings.warn(f"{len(unknown)} sampling point(s) not in SITES and dropped: {unknown}")
        df = df[df['Sampling Point'].isin(POINT_TO_SITE)].copy()
    df['site'] = df['Sampling Point'].map(POINT_TO_SITE)
    df['point_priority'] = df['Sampling Point'].map(POINT_PRIORITY)
    df['week_idx'] = df['Week'] - PHYTO_WEEK_OFFSET
    df['Cells/Litre'] = pd.to_numeric(df['Cells/Litre'], errors='coerce')
    n_bad = df['Cells/Litre'].isna().sum()
    if n_bad:
        warnings.warn(f"{n_bad} row(s) with non-numeric Cells/Litre treated as 0")
        df['Cells/Litre'] = df['Cells/Litre'].fillna(0.0)
    df['sample_date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    return df


def select_samples(df_week, verbose=True):
    """
    Keep one sample per site for the week: the preferred sampling point
    (first name listed in SITES) and, within it, the earliest sample date.
    """
    keep = []
    for site, g in df_week.groupby('site'):
        g = g[g['point_priority'] == g['point_priority'].min()]
        dates = g['sample_date'].dropna().unique()
        if len(dates) > 1:
            first = min(dates)
            if verbose:
                print(f"  note: {site} sampled on {len(dates)} dates this week; using {pd.Timestamp(first).date()}")
            g = g[g['sample_date'] == first]
        keep.append(g)
    return pd.concat(keep) if keep else df_week.iloc[0:0]


def week_pairs(df_week):
    """Consecutive (south -> north) pairs of the sites sampled in this week."""
    sites = sorted(df_week['site'].unique(), key=SITE_ORDER.get)
    return list(zip(sites[:-1], sites[1:]))
