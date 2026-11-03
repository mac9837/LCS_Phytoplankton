"""
Do neighbouring sites separated by a repelling LCS have more dissimilar phytoplankton
communities (genus-level Bray-Curtis) than sites that are not?

Two analyses, each for the whole community, nanophytoplankton and microphytoplankton:

1. Distance effect: partial Spearman correlation between Bray-Curtis dissimilarity and
   sea-path distance, controlling for sampling week (Bray-Curtis and distance are
   ranked, and each week's mean rank is removed from both before correlating).
   This is the covariate step of the Quade analysis below.
2. LCS effect (Quade's rank analysis of covariance, Quade 1967): Bray-Curtis and
   distance are ranked, Bray-Curtis ranks are regressed on distance ranks and sampling
   week, and the residuals of LCS-separated and non-separated pairs are compared with
   a one-sided Mann-Whitney U test. Effect size r = Z / sqrt(N), with a bootstrap
   confidence interval (site pairs resampled) for error bars.

LCS scores are read from legacy_lcs_scores2.py (one 0/1 list per week, in the order of
consecutive south -> north site pairs, as numbered on the LCS images).

Usage
-----
python lcs_bc_analysis2.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from lcs_common2 import (DATA_DIR, RESULTS_DIR, PHYTO_CSV, load_phyto, select_samples,
                         week_pairs)
from legacy_lcs_scores2 import LEGACY_SCORES

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIMESCALES = [2, 3, 4, 5, 7, 9]
WEEKS = range(0, 11)            # weeks without scores drop out
SUBSETS = {'all': None, 'nano': ['Nano'], 'micro': ['Small Micro', 'Large Micro']}
N_BOOT = 2000                   # bootstrap resamples for the effect-size confidence intervals
CI_LEVEL = 0.90                 # 0.90 = two-sided 90% CI, matching the one-sided test at alpha = 0.05
SEED = 1

# Categories not resolved to genus, and heterotrophic ciliates (not phytoplankton).
# Changes in their composition cannot be detected, so they are left out of the
# Bray-Curtis comparison (matched on the 'Species' column).
EXCLUDED_TAXA = [
    'Microflagellate spp.',
    'Pennate diatom', 'Centric diatom',
    'Naked dinoflagellate', 'Armoured dinoflagellate',
    'Dinoflagellate cyst', 'Dinoflagellate cyst (smooth)', 'Dinoflagellate cyst (spiny)',
    'Cilliates', 'Ciliates', 'Tintinnids', 'Favella ehrenbergii',
]


# ---------------------------------------------------------------------------
# Bray-Curtis
# ---------------------------------------------------------------------------
def genus_vector(df_site, size_classes=None):
    """Total cells/L per genus for one site-sample (optionally one size class)."""
    d = df_site if size_classes is None else df_site[df_site['Size Class'].isin(size_classes)]
    d = d.dropna(subset=['Genus'])
    return d.groupby(d['Genus'].astype(str).str.strip())['Cells/Litre'].sum()


def resolve_ambiguous(a, b):
    """Relabel 'A/B' at one site to whichever of A or B is present at the other
    (matched against the other site's ORIGINAL labels, so the result is symmetric)."""
    def remap(s, other):
        out = {}
        for lab, val in s.items():
            target = lab
            if '/' in lab:
                hits = [p.strip() for p in lab.split('/') if p.strip() in other]
                if hits:
                    target = hits[0]
            out[target] = out.get(target, 0.0) + val
        return pd.Series(out, dtype=float)
    la, lb = set(a.index), set(b.index)
    return remap(a, lb), remap(b, la)


def bray_curtis(a, b):
    a, b = resolve_ambiguous(a, b)
    a, b = a.align(b, fill_value=0.0)                 # raw abundances (cells/L)
    denom = (a + b).sum()
    return np.nan if denom == 0 else float((a - b).abs().sum() / denom)


# ---------------------------------------------------------------------------
# Pair table
# ---------------------------------------------------------------------------
def build_pair_table(phyto, weeks):
    """Site pairs are defined from ALL records (so excluding taxa never changes
    which sites are paired, keeping the pairs aligned with the LCS scores);
    Bray-Curtis is then computed without the taxa in EXCLUDED_TAXA."""
    rows = []
    excl = set(t.strip() for t in EXCLUDED_TAXA)
    for week in weeks:
        dfw = select_samples(phyto[phyto['week_idx'] == week], verbose=False)
        if dfw['site'].nunique() < 2:
            continue
        kept = dfw[~dfw['Species'].astype(str).str.strip().isin(excl)]
        for k, (a, b) in enumerate(week_pairs(dfw)):
            da, db = kept[kept['site'] == a], kept[kept['site'] == b]
            row = {'week': week, 'pair_index': k, 'site_a': a, 'site_b': b}
            for name, classes in SUBSETS.items():
                row[f'bc_{name}'] = bray_curtis(genus_vector(da, classes), genus_vector(db, classes))
            rows.append(row)
    return pd.DataFrame(rows)


def add_sea_distances(pairs):
    sea_file = DATA_DIR / 'sea_path_distances.csv'
    if not sea_file.exists():
        raise SystemExit(f"{sea_file} not found: run sea_path_distances.py first")
    sp = pd.read_csv(sea_file)
    lut = {tuple(sorted((r.site_a, r.site_b))): r.sea_km for r in sp.itertuples()}   # pairs are unordered
    pairs['sea_km'] = [lut.get(tuple(sorted((a, b))), np.nan) for a, b in zip(pairs['site_a'], pairs['site_b'])]
    missing = pairs[pairs['sea_km'].isna()]
    if len(missing):
        print(f"  {len(missing)} pair(s) have no sea-path distance and are excluded: "
              f"{sorted(set(missing['site_a'] + '-' + missing['site_b']))}")
    return pairs


# ---------------------------------------------------------------------------
# LCS labels
# ---------------------------------------------------------------------------
def add_lcs_scores(pairs, scores):
    """Attach the per-week 0/1 lists as columns lcs_<T>; stop on any length mismatch."""
    for T, weeks in scores.items():
        col = f'lcs_{T}'
        pairs[col] = np.nan
        for week, arr in weeks.items():
            sub = pairs[pairs['week'] == week].sort_values('pair_index')
            if len(sub) != len(arr):
                raise ValueError(
                    f"{T}-day scores for week {week} have {len(arr)} entries but the data give "
                    f"{len(sub)} consecutive pairs ({', '.join(sub['site_a'] + '-' + sub['site_b'])}).")
            pairs.loc[sub.index, col] = arr
    return pairs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def week_centred_ranks(x, week):
    """Ranks of x with each week's mean rank removed (i.e. ranks adjusted for week)."""
    r = stats.rankdata(x).astype(float)
    for w in np.unique(week):
        g = week == w
        r[g] -= r[g].mean()
    return r


def distance_test(bc, dist, week):
    """Partial Spearman correlation of Bray-Curtis with distance, controlling for week.
    p-values from the t-distribution with n - 2 - (number of weeks - 1) degrees of freedom."""
    rb, rd = week_centred_ranks(bc, week), week_centred_ranks(dist, week)
    rho = (rb @ rd) / np.sqrt((rb @ rb) * (rd @ rd))
    df = len(bc) - 2 - (len(np.unique(week)) - 1)
    t = rho * np.sqrt(df / (1 - rho ** 2))
    return dict(rho=rho, p_two_sided=2 * stats.t.sf(abs(t), df), p_one_sided=stats.t.sf(t, df))


def quade_test(bc, lcs, dist, week):
    """Quade's rank ANCOVA: residuals of Bray-Curtis ranks after regressing on distance
    ranks and week, compared between LCS and non-LCS pairs with a one-sided
    Mann-Whitney U test. Returns the effect size r = Z / sqrt(N) and p."""
    rb, rd = stats.rankdata(bc), stats.rankdata(dist)
    Z = np.column_stack([rd, pd.get_dummies(week).to_numpy(float)])
    resid = rb - Z @ (np.linalg.pinv(Z) @ rb)
    g1, g0 = resid[lcs == 1], resid[lcs == 0]
    n1, n0 = len(g1), len(g0)
    if n1 == 0 or n0 == 0:
        return dict(r=np.nan, p=np.nan)
    U, p = stats.mannwhitneyu(g1, g0, alternative='greater')
    z = (U - n1 * n0 / 2) / np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12)
    return dict(r=z / np.sqrt(n1 + n0), p=p)


def quade_r_ci(bc, lcs, dist, week, pair, rng):
    """Bootstrap confidence interval for the Quade effect size r. Whole site pairs are
    resampled with replacement (all of a pair's weeks kept together, because the same
    pairs recur across weeks), and the full Quade procedure is repeated on each
    resample. Percentile interval at CI_LEVEL."""
    ids = pd.factorize(pair)[0]
    members = [np.flatnonzero(ids == k) for k in range(ids.max() + 1)]
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if lcs[idx].min() == lcs[idx].max():
            continue
        r = quade_test(bc[idx], lcs[idx], dist[idx], week[idx])['r']
        if np.isfinite(r):
            vals.append(r)
    if len(vals) < 100:
        return np.nan, np.nan
    a = (1 - CI_LEVEL) / 2 * 100
    return tuple(np.percentile(vals, [a, 100 - a]))


def run_distance_tests(pairs, dist_col):
    rows = []
    for subset in SUBSETS:
        d = pairs.dropna(subset=[f'bc_{subset}', dist_col])
        rows.append(dict(subset=subset, n_pairs=len(d),
                         **distance_test(d[f'bc_{subset}'].to_numpy(float), d[dist_col].to_numpy(float),
                                         d['week'].to_numpy())))
    return pd.DataFrame(rows)


def run_lcs_tests(pairs, dist_col, rng):
    rows = []
    for T in TIMESCALES:
        col = f'lcs_{T}'
        if col not in pairs:
            continue
        for subset in SUBSETS:
            d = pairs.dropna(subset=[col, f'bc_{subset}', dist_col])
            if d.empty:
                continue
            bc = d[f'bc_{subset}'].to_numpy(float)
            lcs = d[col].to_numpy(int)
            dist = d[dist_col].to_numpy(float)
            week = d['week'].to_numpy()
            pair = np.array([' | '.join(sorted(p)) for p in zip(d['site_a'], d['site_b'])])
            res = quade_test(bc, lcs, dist, week)
            lo, hi = quade_r_ci(bc, lcs, dist, week, pair, rng)
            rows.append(dict(timescale=T, subset=subset, n_pairs=len(d), n_lcs=int(lcs.sum()),
                             r=res['r'], ci_lo=lo, ci_hi=hi, p=res['p']))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    phyto = load_phyto()
    print(f"Data: {PHYTO_CSV}")
    pairs = build_pair_table(phyto, WEEKS)
    print(f"{len(pairs)} consecutive site pairs over {pairs['week'].nunique()} weeks")
    pairs = add_sea_distances(pairs)
    pairs = add_lcs_scores(pairs, LEGACY_SCORES)
    pairs.to_csv(RESULTS_DIR / 'pair_table.csv', index=False)

    dist_res = run_distance_tests(pairs, 'sea_km')
    lcs_res = run_lcs_tests(pairs, 'sea_km', rng)
    dist_res.to_csv(RESULTS_DIR / 'distance_test.csv', index=False)
    lcs_res.to_csv(RESULTS_DIR / 'lcs_bc_tests.csv', index=False)

    with pd.option_context('display.width', 200, 'display.max_rows', 200, 'display.float_format', '{:.3f}'.format):
        print("\n1. DISTANCE EFFECT: partial Spearman correlation of Bray-Curtis with sea-path "
              "distance, controlling for sampling week")
        print(dist_res.to_string(index=False))
        print("\n2. LCS EFFECT: Quade rank ANCOVA (sea-path distance and sampling week), "
              "one-sided Mann-Whitney U on the residuals")
        print(lcs_res.to_string(index=False))
    print(f"\nr = Z / sqrt(N); ci_lo, ci_hi = {100 * CI_LEVEL:.0f}% bootstrap confidence interval of r "
          f"({N_BOOT} resamples of site pairs)")
    print(f"Saved pair_table.csv, distance_test.csv and lcs_bc_tests.csv in {RESULTS_DIR}")


if __name__ == '__main__':
    main()
