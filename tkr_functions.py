"""
tkr_functions.py
=================

Function library for Transfer-Kernel Ranging (TKR), the point-source methane
localisation and quantification method described in:

    Klappenbach et al., "Novel method to locate and quantify point-source
    methane emissions using time series of ground-based column observations"
    (EGUsphere, 2026, https://doi.org/10.5194/egusphere-2026-204).

This file collects every function used by the accompanying pipeline notebook
(`TKR_full_pipeline.ipynb`), grouped by the stage of the method they belong
to. Where a function implements a specific step of the paper, the relevant
section/equation is noted in its docstring.

Overview of the pipeline (see the paper for full derivations):

  1. Observation loading & peak detection            -> Sect. 2.1, Appendix B
  2. Geometry helpers (distance / bearing)            -> Appendix E
  3. Averaging-kernel / vertical weighting            -> Appendix C, D
  4. Trajectory loading & hi-res interpolation        -> Sect. 2.3
  5. Upwind-domain segmentation (radial/angular bins) -> Sect. 2.3.1, Appendix E, F
  6. Transport-kernel construction & fitting           -> Sect. 2.4-2.6, Eq. 2-8

Historical note: this module was formerly named `lsp_functions.py` ("Local
Source Projection", the working title used during development). The method
is now released under the name "Transfer-Kernel Ranging" (TKR), matching the
terminology used in the paper. Function names and behaviour are unchanged
from the original research code -- only names, structure, and comments were
adapted for this release.
"""
import os
import json
import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================================
# 1. Observation loading and peak detection (paper Sect. 2.1, Appendix B)
# =========================================================================
def read_obs_proffast_parquet(path, date, species='CH4', quantile=0.1, roll_time='60min',
                               location_id=None, quality_flag_value=0, utc_offset_hours=-7):
    """Load total-column observations from a modern EM27/SUN retrieval-bundle parquet
    (GGG2020 / PROFFAST 2.4 style: clean column names, tz-aware 'utc' timestamp) and
    compute the rolling-quantile background and enhancement (paper Sect. 2.1 / Appendix B,
    Eq. B1).

    This is the parquet-native counterpart of the original notebook's older
    `read_obs_proffit` helper, which expected a raw PROFFAST `comb_invparms_*.csv`
    file with space-prefixed column names (' XCH4', ' appSZA', ...) instead.

    Differences from that older helper (both confirmed with the author):
      - No XAIR/0.99775 empirical correction is applied here: this retrieval bundle
        is already corrected, unlike the older comb_invparms CSV format.
      - `quality_flag_value`: rows are kept where `event_data_quality_flag == quality_flag_value`.
        The convention (whether 0 means "good" or "bad") was NOT independently confirmed
        for this bundle format -- verify before relying on this filter for data where the
        flag actually varies (in the reference file used during development, all rows had
        flag == 0, so the filter was a no-op either way).

    IMPORTANT: `date` is matched against the LOCAL date (`utc_offset_hours`), not the UTC
    calendar date. A multi-day bundle's UTC-day boundary falls in the middle of the local
    afternoon/evening, so naive UTC-date filtering pulls in the tail of the *previous*
    local day's measurement session (observed directly during development: filtering by
    UTC date included ~40 extra minutes of the prior evening, appearing as spurious peaks
    at 00:00-00:40 UTC). Filtering by local date avoids this.

    Parameters
    ----------
    path : str
        Path to the retrieval-bundle parquet file.
    date : str or datetime-like
        Local date to select (the bundle may span multiple days).
    species : str
        Target gas column prefix, e.g. 'CH4' -> uses columns 'XCH4', 'CH4'.
    quantile : float
        Quantile used to define the rolling background (paper: 0.1, i.e. the 10th
        percentile; see the sensitivity analysis in Appendix B, Table B1).
    roll_time : str
        Width of the centered rolling window used for the background, as a pandas
        offset string (paper: '60min'; see Appendix B for the sensitivity to this choice).
    location_id : str, optional
        If given, additionally filters to this location_id. Note: a physical site
        may be recorded under different location_id labels across a campaign (e.g.
        after a database migration) -- check the unique `location_id` values per date
        before assuming they match a site label used elsewhere in your pipeline.
    quality_flag_value : int
        Value of `event_data_quality_flag` to keep (see the note above).
    utc_offset_hours : float
        Local time zone offset from UTC, used only to determine the local calendar
        date boundary (e.g. -7 for PDT). Does not affect the returned timestamps,
        which remain UTC.

    Returns
    -------
    DataFrame indexed by 'minutes' (minutes since local midnight UTC), with columns
    including 'Enh_ppm' (the background-subtracted enhancement used throughout the
    pipeline), 'quantile' (the rolling background itself), and auxiliary columns
    (XAIR, H2O, XCO2, sza, azi, ...) used for the co-emitted-species / data-quality
    checks in Appendix B.
    """
    obs = pd.read_parquet(path)

    target_date = pd.to_datetime(date).date()
    local_date = (obs['utc'] + pd.Timedelta(hours=utc_offset_hours)).dt.date
    obs = obs[local_date == target_date].copy()
    if location_id is not None:
        obs = obs[obs['location_id'] == location_id].copy()
    if len(obs) == 0:
        raise ValueError(f"No rows found for local date={date}, location_id={location_id} in {path}")

    n_before = len(obs)
    obs = obs[obs['event_data_quality_flag'] == quality_flag_value].copy()
    if len(obs) < n_before:
        print(f"Quality flag filter: kept {len(obs)}/{n_before} rows (event_data_quality_flag == {quality_flag_value}).")

    obs = obs.sort_values('utc').set_index('utc')

    dmin = pd.Timestamp(target_date, tz='UTC')
    obs['minutes'] = (obs.index - dmin).total_seconds() / 60

    # Already-corrected retrieval bundle: no XAIR-based correction is applied here
    # (see docstring above for the difference to the older comb_invparms CSV format).
    obs['XCH4_c'] = obs['XCH4']
    obs['XCO2_c'] = obs['XCO2']

    # Rolling-quantile background (paper Appendix B, Eq. B1): for every time step, take
    # the given quantile of all observations within a centered window of width `roll_time`.
    xb = obs[['XCH4', 'CH4', 'XAIR', 'H2O', 'XCO2', 'XCH4_c', 'XCO2_c']].rolling(
        roll_time, min_periods=1, center=True
    ).quantile(quantile, interpolation='linear')

    # Enhancement = observation minus rolling background, in several unit/variant flavours.
    obs['Enh_ppm'] = obs[f'X{species}'] - xb[f'X{species}']
    obs['Enh_c_ppm'] = obs[f'X{species}_c'] - xb[f'X{species}_c']
    obs['Enh_molec_m2'] = obs[species] - xb[species]
    obs['quantile'] = xb[f'X{species}']
    obs['quantile_molec_m2'] = xb[species]
    obs['obs_raw'] = obs[f'X{species}']
    obs['xair'] = obs['XAIR']
    obs['h2o_molec'] = obs['H2O']
    obs['h2o_molec_quantile'] = xb['H2O']
    obs['molec_m2'] = obs[species]
    obs['XCO2_quantile'] = xb['XCO2']
    obs['XCO2_c_quantile'] = xb['XCO2_c']
    obs['quantile_xair'] = xb['XAIR']

    # Track contiguous "chunks" of non-NaN data (data gaps, e.g. clouds, split the day
    # into segments). Peak finding (see `find_group_peaks` below) is run per chunk so
    # that a data gap is never mistaken for part of a peak.
    nan_cont_index = np.cumsum(~np.isnan(obs['Enh_ppm'])) - 1
    chunk_index = np.concatenate(([0], np.cumsum(np.diff(np.isnan(obs['Enh_ppm'])))))
    obs['nan_cont_index'] = nan_cont_index
    obs['nan_chunk_index'] = chunk_index

    cols = ['Enh_ppm', 'Enh_c_ppm', 'obs_raw', 'minutes', 'quantile', 'nan_cont_index', 'nan_chunk_index',
            'xair', 'quantile_xair', 'h2o_molec', 'h2o_molec_quantile', 'molec_m2', 'quantile_molec_m2',
            'Enh_molec_m2', 'XCO2', 'XCO2_c', 'XCO2_quantile', 'XCO2_c_quantile', 'sza', 'azi']
    return obs[cols]


def find_group_peaks(group, key, prominence):
    """Find enhancement peaks within one contiguous (gap-free) chunk of observations,
    using `scipy.signal.find_peaks`. `prominence` is the minimum peak prominence
    (in the same units as `key`, i.e. ppm for 'Enh_ppm'); see `config.json`'s
    `prominence` field and the paper's Fig. 1 for the resulting peak set.

    Returns a DataFrame with one row per detected peak: its time, height, prominence,
    and the left/right "base" times used by `find_peaks` to define the peak extent.
    """
    from scipy.signal import find_peaks
    pks, pkd = find_peaks(group[key].dropna(), prominence=prominence)
    tidx = group.index[pks]
    return pd.DataFrame({
        'peak_time': tidx,
        'peak_height': group[key].loc[tidx],
        'prominence': pkd['prominences'],
        'left_bases': group.index[pkd['left_bases']],
        'right_bases': group.index[pkd['right_bases']],
    })


def find_nearest(array, value):
    """Return the element of a sorted `array` closest to `value` (binary search).
    Used to snap an observed peak time onto the discrete time grid on which
    trajectories were pre-computed (`df_dens.minutes`)."""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx - 1]) < np.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit, least_squares
from pvlib.solarposition import get_solarposition


# =========================================================================
# 2. Geometry helpers (paper Appendix E: discretisation of the upwind domain)
# =========================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon coordinates, in meters."""
    R = 6371000
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    """Initial compass bearing in degrees from point 1 to point 2 (north=0, east=90)."""
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.degrees(np.arctan2(y, x))
    return (initial_bearing + 360) % 360


def circular_mean(deg_arr, phase=np.pi, what='mean'):
    """Circular mean (or median) of an array of angles given in degrees, robust to the
    0/360 degree wraparound (used e.g. to find the representative wind direction for
    the angular/bearing segmentation, paper Sect. 2.3.1 / Appendix E)."""
    x = np.sin(np.pi / 180 * deg_arr + phase)
    y = np.cos(np.pi / 180 * deg_arr + phase)
    if what == 'mean':
        X, Y = np.nanmean(x), np.nanmean(y)
    elif what == 'median':
        X, Y = np.nanmedian(x), np.nanmedian(y)
    return 180 / np.pi * (np.arctan2(X, Y) + phase)


# =========================================================================
# 3. Atmosphere / averaging-kernel helpers (paper Appendix C, D)
# =========================================================================
def pressure_std_atm(alt, p0=1013.15, z0=0, H=7800):
    """Barometric-formula pressure (hPa) at altitude `alt` (m) above `z0`, given a
    surface pressure `p0` and scale height `H`. Used to map particle altitude to
    pressure for the averaging-kernel lookup (paper Appendix D)."""
    return p0 * np.exp(-(alt - z0) / H)


def molec_column(alt, p0=1013.15, z0=0, H=7800, A=1, g=9.81, m_molar=28.97 / 1000):
    """Total dry-air column (mol/m^2, for A=1 m^2) above altitude `alt`, from the
    hydrostatic relation. Used as the normalisation w_infinity in the vertical
    weighting (paper Appendix D, Eq. D1-D2)."""
    return pressure_std_atm(alt, p0, z0, H) * A / g / m_molar


def load_ak(filepath: str, as_fun=True):
    """Load the instrument's column averaging kernel A(sza, pressure) from a JSON file
    (paper Appendix D). The JSON is expected to have keys 'szas', 'pressures', 'aks'
    (a 2D array indexed [sza, pressure]).

    If `as_fun=True` (default), returns a `RegularGridInterpolator` so the kernel can
    be evaluated at arbitrary (sza, pressure) pairs, e.g. `ak((sza, pressure))`.
    If `as_fun=False`, returns the raw grid as a DataFrame instead.
    """
    assert filepath.endswith(".json"), "Filepath must end with .json"
    with open(filepath, "r") as f:
        d = json.load(f)
    if as_fun:
        return RegularGridInterpolator(
            (d['szas'], d['pressures']), d['aks'], bounds_error=False, fill_value=None
        )
    return pd.DataFrame(data=d['aks'], index=d['szas'], columns=d['pressures'])


# =========================================================================
# 4. Trajectory loading and hi-res interpolation (paper Sect. 2.3)
# =========================================================================
def calculate_wind_vectors(df, group_id=None):
    """Given one backward-trajectory (particle path, 'indx' group) sorted by time,
    compute the wind speed and direction implied by consecutive particle positions
    (used for the per-segment `wind_speed` / `wind_dir_deg` diagnostics, e.g. the
    wind-speed correction discussed in paper Sect. 2.7.2 / Appendix H)."""
    df = df.sort_values(by='time', ascending=False).reset_index(drop=True)
    lat_prev = df['lati'].shift(-1)
    lon_prev = df['long'].shift(-1)
    time_prev = df['date'].shift(-1)
    df['distance_m'] = haversine_distance(lat_prev, lon_prev, df['lati'], df['long'])
    df['dt_sec'] = (df['date'] - time_prev).dt.total_seconds()
    df['wind_speed'] = df['distance_m'] / df['dt_sec']
    dlon = np.radians(df['long'] - lon_prev)
    y = np.sin(dlon) * np.cos(np.radians(df['lati']))
    x = (np.cos(np.radians(lat_prev)) * np.sin(np.radians(df['lati'])) -
         np.sin(np.radians(lat_prev)) * np.cos(np.radians(df['lati'])) * np.cos(dlon))
    bearing = np.degrees(np.arctan2(y, x))
    # Trajectory direction -> meteorological wind direction (the direction the wind
    # is blowing FROM) via the +180 degree convention.
    df['wind_dir_deg'] = (bearing + 180) % 360
    if group_id is not None:
        df['indx'] = group_id
    return df


def interpolate_particle_trajectories(x, hi_res_time_resolution, plot=False):
    """Interpolate each particle trajectory (grouped by 'indx') from STILT's native
    (coarser) time step onto a finer, regular time grid of spacing
    `hi_res_time_resolution` (minutes; see `config.json`'s `hi_res_time_resolution`,
    used to build a smooth transport kernel in `make_histogram` below).

    The per-timestep footprint ('foot') column is rescaled by the ratio of the new to
    the old time step so that its time-integral is preserved under interpolation.
    """
    all_out = []
    for cidx, grp in x.groupby("indx"):
        df = grp.copy()
        df['time'] = df['time'].astype(float)
        low_res_time = np.nanmedian(np.abs(np.diff(df['time'])))
        hi_res_time_to = np.min(df['time'])
        hi_res_time = -np.arange(
            -np.max(df['time']), -hi_res_time_to + hi_res_time_resolution, hi_res_time_resolution
        )[::-1]
        foot_corrector = hi_res_time_resolution / low_res_time
        df = df.sort_values(by='time', ascending=True)
        tx = df.reset_index().set_index('time')
        df_out = pd.DataFrame(index=hi_res_time, columns=tx.columns, dtype="float64")
        df_out.index.name = "time"
        df_out = df_out.astype({"date": tx['date'].dtype})
        df_out.update(tx)
        if 'foot' in df_out:
            df_out['foot'] = df_out['foot'] * foot_corrector
        df_out = df_out.interpolate(axis=0)
        rest = tx.drop(df_out.index.intersection(tx.index))
        df_out = pd.concat([df_out, rest]).sort_index().reset_index()
        df_out = df_out.sort_index(ascending=False)
        df_out['indx'] = cidx
        all_out.append(df_out)
    return pd.concat(all_out, ignore_index=True)


def read_traj_parq(traj_parq_path, cutoff_time=None, select_release_heights=None, hi_res_time=0,
                    averaging_kernel_path=None):
    """Read one exported STILT trajectory product for a single peak time
    ('<time>-<site_id>-total-column/' folder: an 'about.json' metadata file plus
    one 'traj/particle_stilt.<i>.parquet' per receptor release height, paper Sect. 2.3).

    For every release height `i` in `select_release_heights`:
      - loads the particle trajectories from parquet,
      - inserts a synthetic t=0 row at the receptor's own position (release point),
      - computes wind vectors along each trajectory (`calculate_wind_vectors`),
      - optionally interpolates onto a finer time grid (`hi_res_time`),
      - attaches the receptor-relative distance/bearing of every particle position
        (`recep_dist_km`, `recep_bearing_deg`, and their east/north components),
      - attaches the per-trajectory averaging-kernel weight (paper Appendix D) via
        `full_weight_per_trajectory = density_weight / particles_per_release_point * A(sza, pressure)`.

    Adapted from the original research notebook: the averaging-kernel path is now a
    function argument (`averaging_kernel_path`) instead of a hardcoded local path, so
    it can be supplied via `config.json`.

    Parameters
    ----------
    traj_parq_path : str
        Path to the peak's trajectory export folder (trailing slash expected).
    cutoff_time : float, optional
        Drop particle positions with `time` (minutes, negative = backward in time)
        less than this cutoff, i.e. keep only the most recent part of the backward
        trajectory (see `config.json`'s `trajectory_cutoff_minutes`).
    select_release_heights : iterable of int, optional
        Which release-height indices to load (default: all). See `config.json`'s
        `release_heights` (number of receptor points along the line of sight,
        paper Sect. 2.3 / Appendix C).
    hi_res_time : float
        If nonzero, interpolate trajectories onto this time resolution (minutes)
        via `interpolate_particle_trajectories`.
    averaging_kernel_path : str
        Path to the averaging-kernel JSON (see `load_ak`).

    Returns
    -------
    (trajectories, meta) : (DataFrame, dict)
        `trajectories` concatenates all requested release heights; `meta` holds
        site-level metadata (receptor lat/lon, measurement time, particles per
        release point).
    """
    about_path = traj_parq_path + 'about.json'
    if not os.path.exists(about_path):
        raise FileNotFoundError(f"{about_path} does not exist.")

    with open(about_path) as f:
        jsn = json.load(f)

    ak = load_ak(averaging_kernel_path)

    if len(jsn['config']['jobs']) > 1:
        print('Warning! about.json contains multiple jobs. Only first one is used.')
        release_heights = jsn['config']['jobs'][0]['release']['release_heights']
        particles_per_release_point = jsn['config']['jobs'][0]['release']['particles_per_release_point']
    elif len(jsn['config']['jobs']) == 1:
        release_heights = jsn['config']['jobs'][0]['release']['release_heights']
        particles_per_release_point = jsn['config']['jobs'][0]['release']['particles_per_release_point']
    else:
        release_heights = jsn['release']['release_heights']
        particles_per_release_point = jsn['release']['particles_per_release_point']

    site_sza = 90 - jsn['receptor']['sun_elevation']
    release_lats = [x['lat'] for x in jsn['receptor']['release_points']]
    release_lons = [x['lon'] for x in jsn['receptor']['release_points']]
    release_density_weight = [x['density_weight'] for x in jsn['receptor']['release_points']]
    release_alt_agl = [x['alt_agl'] for x in jsn['receptor']['release_points']]
    release_alt_asl = [x['alt_asl'] for x in jsn['receptor']['release_points']]
    release_pressures = [x['pressure'] for x in jsn['receptor']['release_points']]

    daytime = jsn['receptor']['dt']

    meta = {
        'site_loc_lon': jsn['receptor']['location']['lon'],
        'site_loc_lat': jsn['receptor']['location']['lat'],
        'daytime': daytime,
        'particles_per_release_point': particles_per_release_point,
    }

    trj = []
    if not select_release_heights:
        select_release_heights = range(len(release_heights))

    for i in select_release_heights:
        traj = pd.read_parquet(traj_parq_path + '/traj/particle_stilt.%i.parquet' % i)
        if cutoff_time is not None:
            traj = traj[traj['time'] > cutoff_time]

        # Insert a synthetic t=0 row at the receptor's own release position, so every
        # trajectory explicitly starts at the receptor (needed for the delta-function
        # argument in Sect. 2.5 / Appendix A: at t=0 all particles coincide at a point).
        tmax = np.max(traj['time'])
        insert = traj[traj['time'] == tmax].copy()
        insert.loc[:, 'time'] = -0
        insert.loc[:, 'lati'] = release_lats[i]
        insert.loc[:, 'long'] = release_lons[i]
        insert.loc[:, 'zasl'] = release_alt_asl[i]
        insert.loc[:, 'zagl'] = release_alt_agl[i]

        traj = pd.concat([insert, traj], ignore_index=True)
        traj['date'] = pd.to_datetime(daytime) + pd.to_timedelta(traj['time'], unit='m')
        traj = traj.groupby('indx', group_keys=False).apply(
            lambda g: calculate_wind_vectors(g, g.name), include_groups=False
        )

        if hi_res_time != 0:
            traj = interpolate_particle_trajectories(traj, hi_res_time, plot=False)

        traj['alt'] = release_heights[i]
        traj['scf'] = release_density_weight[i]
        traj['numpar'] = particles_per_release_point
        traj['recep'] = int(pd.to_datetime(daytime).strftime('%Y%m%d'))
        traj['ak'] = ak((site_sza, release_pressures[i]))
        # Per-trajectory averaging-kernel weight (paper Appendix D, Eq. D1): combines the
        # relative weight of this receptor point (density_weight / particles per point)
        # with the instrument's column averaging kernel at this receptor's SZA/pressure.
        traj['full_weight_per_trajectory'] = (
            release_density_weight[i] / particles_per_release_point * ak((site_sza, release_pressures[i]))
        )

        traj.sort_values(by=['indx', 'time'], ascending=[True, False], inplace=True)
        traj.index = range(len(traj))

        # Receptor-relative polar coordinates of every particle position (used for the
        # radial/angular segmentation below, paper Sect. 2.3.1 / Appendix E).
        traj['recep_dist_km'] = haversine_distance(
            [release_lats[i]] * len(traj), [release_lons[i]] * len(traj), traj['lati'], traj['long']
        ) / 1000
        traj['recep_bearing_deg'] = calculate_initial_bearing(
            [release_lats[i]] * len(traj), [release_lons[i]] * len(traj), traj['lati'], traj['long']
        )
        traj['recep_dist_north_km'] = traj['recep_dist_km'] * np.cos(np.radians(traj['recep_bearing_deg']))
        traj['recep_dist_east_km'] = traj['recep_dist_km'] * np.sin(np.radians(traj['recep_bearing_deg']))

        trj.append(traj)

    return pd.concat(trj), meta


# =========================================================================
# 5. Radial / angular segmentation of the upwind domain (paper Sect. 2.3.1,
#    Appendix E: discretisation; Appendix F: choice of bin size)
# =========================================================================
def create_radius(drf, dr=0.001, rad_max=100, max_steps=50, d_alpha=False):
    """Create exponentially increasing radial bin edges (km) for the upwind-distance
    segmentation (paper Appendix E, radial index `i`). Starting step size `dr` grows
    by a factor `(1 + drf)` at every step, giving high resolution near the receptor
    (small `i`) and efficient coverage of large radii (paper Appendix F motivates this
    non-uniform choice; Fig. F1/F2 show the method is insensitive to the exact bin size).

    Parameters
    ----------
    drf : float
        Fractional growth rate of the radial step size per bin (config: `drf`).
    dr : float
        Initial radial step size in km (config: `dr`).
    rad_max : float
        Maximum radius in km at which to stop (config: `rad_max_km`).
    max_steps : int
        Hard cap on the number of radial bins (config: `max_radial_steps`).
    d_alpha : bool
        If True, also return the corresponding angular bin width (degrees) at each
        radius, chosen so that segments remain approximately square (isotropic) in
        the local Cartesian domain (see `bearing_segmentation` below).

    Returns
    -------
    radii : list of float
        Radial bin edges (km), starting at 0.
    dalpha : list of float, only if d_alpha=True
        Angular bin width (degrees) associated with each radius.
    """
    radii = [0]
    dalpha = [60]
    for x in range(max_steps):
        radii.append(radii[-1] + dr)
        dr += drf * dr
        if d_alpha:
            dalpha.append(360. * np.abs(dr) / (np.pi * (radii[x] + radii[x - 1])))
        if radii[-1] > rad_max:
            break
    if d_alpha:
        return (radii, dalpha)
    return radii


def bearing_segmentation(x):
    """Assign each particle position in a (source-altitude, upwind-distance) group to
    an angular (wind-relative bearing) bin (paper Appendix E, angular index `j`).

    The bin grid is centered on the group's circular-median bearing so that the
    dominant wind direction falls in the middle of a bin rather than straddling a
    bin edge, and the bin width is the mean angular step `dalpha` precomputed by
    `create_radius` for this radial distance.
    """
    deg_step = x['dalpha'].mean()
    x['deg_step'] = deg_step
    med = circular_mean(x.recep_bearing_deg, what='median')
    delta = med - np.floor(med / deg_step) * deg_step + deg_step / 2
    segmentation = np.arange(start=-deg_step * 2, stop=360 + deg_step * 2, step=deg_step) + delta
    x['recep_bearing_segmentation'] = pd.cut(x['recep_bearing_deg'], segmentation, right=False)
    return x[['recep_dist_km', 'recep_bearing_deg', 'recep_bearing_segmentation', 'deg_step']]


def ring_area(dist_km, bearing_steps):
    """Ground-surface area (m^2) of each (radial, angular) segment: an annular sector
    between the inner/outer radius (`dist_km`, a pandas Interval per row) spanning the
    angular bin width (`bearing_steps`, an Interval per row). Used as the emission
    footprint's source area, `segment_area_m2`, in the emission-strength inversion
    (paper Sect. 2.6, Eq. 7-8)."""
    steps = np.array([b.right - b.left for b in bearing_steps])
    dmax = np.array([d.right for d in dist_km])
    dmin = np.array([d.left for d in dist_km])
    ring_area_km_2 = np.pi * ((dmax) ** 2 - (dmin) ** 2)
    return ring_area_km_2 * steps / 360 * 1000 * 1000


# =========================================================================
# 6. Step 1 + Step 2: transport-kernel fit & emission-strength inversion
#    (paper Sect. 2.4-2.6, Eq. 2-8)
# =========================================================================
def make_histogram(group, observation, peak_time,
                    std_dev=1, weight=True, plot=False, max_plot_radius=3.0,
                    color='orange', target='Enh_ppm', debug=False,
                    emission_duration=False, check_group=True,
                    hi_res_time_resolution=1 / 30):
    """Core fitting routine for one upwind segment (a single (altitude, distance,
    bearing) group of particles): builds the segment's transport kernel from particle
    arrival times, fits it to the observed peak (Step 1, paper Eq. 5-6), and,
    if `emission_duration=True`, additionally deconvolves the emission time series
    for this segment (Step 2, paper Eq. 7-8).

    Called once per segment via a groupby().apply() in the main pipeline notebook, so
    `group` here is already restricted to particles belonging to one (source-altitude,
    upwind-distance, bearing) bin.

    Step 0: build the transport kernel k(t) for this segment (paper Eq. 2-3)
    --------------------------------------------------------------------------
    `foot_source` implements the surface-sensitivity footprint of Eq. (2),
    `f = dt_sim * m_air / (dz * rho)`, and `foot_recep` multiplies it by the
    per-trajectory averaging-kernel weight (`full_weight_per_trajectory`, from
    `read_traj_parq`) to give the receptor-weighted footprint of Eq. (3). A weighted
    histogram of particle arrival times (relative to the segment's median transit
    time) then gives the discretised transport kernel k_i,j,k(t_m); this histogram is
    linearly interpolated (`kernel = lambda minutes: ...`) so it can be evaluated
    (and later convolved) at arbitrary times.

    Step 1: kernel fit to find r_max (paper Eq. 5-6)
    --------------------------------------------------------------------------
    The kernel is fit to the observed enhancement `observation[target]` with three
    free parameters -- amplitude `s`, offset `o`, and time shift `p` (`fun = lambda x,
    s, o, p: kernel(x + p) * s + o`) via `scipy.optimize.curve_fit`. The standard
    deviation of the fit residual (`residual_std_ppm`) is the quantity `rho_s` in
    Eq. (6); across all segments at a given upwind distance, its minimum identifies
    `r_max` (done downstream in the notebook, not in this function).

    Step 2: emission-strength inversion (paper Eq. 7-8), only if `emission_duration=True`
    --------------------------------------------------------------------------
    A non-negative least-squares deconvolution recovers the time-resolved emission
    rate `e(t)` such that convolving it with the (shifted) kernel best reproduces the
    observation (Eq. 7). The emission is scaled by an arbitrary factor `fit_scf` for
    numerical conditioning and rescaled back afterwards. Integrating `e(t)` over time
    and multiplying by the segment's ground area gives the total emitted mass for this
    segment, `total_emission_mol` (Eq. 8).

    Parameters
    ----------
    group : DataFrame
        Particle-timestep rows for one upwind segment (must include 'dens', 'dz_source',
        'time', 'segment_area_m2', 'foot', 'full_weight_per_trajectory').
    observation : DataFrame
        Observed enhancement time series around the peak, indexed by minutes (see
        `peak_window_minutes` in config.json for how wide a window is used).
    peak_time : float
        Time (minutes) of the observed peak maximum, used to center the kernel.
    std_dev : float
        Width (minutes) of the Gaussian weighting applied to the fit residual when
        `weight=True`, giving more influence to points near the peak.
    weight : bool
        Whether to apply the Gaussian weighting above during the curve_fit.
    target : str
        Column of `observation` to fit against (default 'Enh_ppm').
    emission_duration : bool
        If True, also run the Step 2 emission-strength inversion (slower).
    check_group : bool
        If True, warn if `group` unexpectedly spans more than one segment area.
    hi_res_time_resolution : float
        Half-bin width (minutes) used when histogramming particle arrival times.

    Returns
    -------
    Series with (among others): 'residual_std_ppm' (rho_s, Eq. 6), 'kernel_fit_params'
    (amplitude/offset/shift), 'kernel_fitted' (fitted curve on the observation grid),
    and, if `emission_duration=True`, 'emission_mol' (total mass, Eq. 8),
    'emission_duration_min', 'duration_minutes' (the fitted e(t)), and
    'duration_residual' (the Step 2 fit residual).
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    m_air = 28.97 / 1000

    # --- Build the transport kernel for this segment (paper Eq. 2-3) ---
    group['foot_source'] = hi_res_time_resolution * 60 * m_air / (group['dens'] * group['dz_source'])
    group['foot_recep'] = group['foot_source'] * group['full_weight_per_trajectory']
    group['group_vertical_weight'] = group['foot_recep']
    group.loc[(np.isnan(group['group_vertical_weight']), 'group_vertical_weight')] = 0

    out = pd.concat((group.mean(numeric_only=True), group.std(numeric_only=True).add_suffix('_std')), axis=0)
    if check_group and np.sum(group['segment_area_m2'] != group['segment_area_m2'].iloc[0]) > 0:
        print('Warning: multiple segments in group!')

    group_area_m2 = group['segment_area_m2'].iloc[0]
    group_sensitivity = group['group_vertical_weight'].sum()

    # Weighted histogram of particle arrival times -> discretised kernel k_i,j,k(t_m)
    # (paper Eq. 3). Bin edges extend to +/-inf so out-of-range particles are still
    # counted (in `outside_before` / `outside_after`) without biasing the in-range bins.
    c_bins = np.concatenate(([-np.inf], np.arange(-30, 30, hi_res_time_resolution * 2), [np.inf]))
    counts, t_bins = np.histogram(group['time'] - group['time'].median(), bins=c_bins,
                                   weights=group['group_vertical_weight'])
    counts = counts / 2
    outside_before, outside_after = counts[0], counts[-1]
    t_bins, counts = t_bins[1:-1], counts[1:-1]
    midx = np.argmax(counts)
    tbins = t_bins[:-1] + np.diff(t_bins) / 2
    # Reverse and re-center the histogram time axis onto the observation's peak_time
    # (backward-trajectory time -> forward/observation time), then make it a
    # continuous, interpolatable kernel function.
    tbins_obs = -(tbins - tbins[midx]) + peak_time
    kernel = lambda minutes: np.interp(minutes, tbins_obs[::-1], counts[::-1])

    # --- Step 1: fit amplitude/offset/shift of the kernel to the observed peak (Eq. 5) ---
    fun = lambda x, s, o, p: kernel(x + p) * s + o
    A0 = observation[target].max() / np.max(counts)
    p0 = [A0, 0, 0]
    bounds = [(1e-10, -A0 / 4, -1), (A0 * 100, A0 / 4, 1)]
    popt = p0

    try:
        if weight:
            # Gaussian weighting centered on the peak: down-weights the fit residual
            # far from the peak so the optimizer focuses on reproducing the peak shape.
            w = 1.1 - np.exp(-((observation.index - peak_time) / std_dev) ** 2)
            popt, pcov = curve_fit(fun, observation.index, observation[target], p0=p0, sigma=w,
                                    bounds=bounds, check_finite=True)
        else:
            popt, pcov = curve_fit(fun, observation.index, observation[target], p0=p0,
                                    bounds=bounds, check_finite=True)

        emission_puffs = np.zeros_like(observation.index, dtype=float)
        emission_duration_min = np.nan  # default if emission_duration=False or the peak is zero

        if emission_duration:
            # --- Step 2: non-negative least-squares deconvolution (Eq. 7-8) ---
            t = observation.index
            xx = observation[target].to_numpy() - popt[1]  # remove the fitted offset
            fit_scf = 1e5  # scaling factor for numerical conditioning of the NNLS problem
            h = kernel(observation.index + popt[2]) * fit_scf

            def fwd_fun(e):
                return np.convolve(h, e, mode='same')

            def residual_fun(e):
                return np.abs(xx - fwd_fun(e))

            p0e = np.zeros_like(h)
            bounds_e = [-1e-15, np.inf]  # enforce e(t) >= 0 (non-negativity, Eq. 7-8)
            E = least_squares(residual_fun, p0e, bounds=bounds_e)
            emission_puffs = E.x.copy() * fit_scf  # undo the numerical scaling
            duration_residual = residual_fun(E.x)
            enhancement_fwd = fwd_fun(E.x)
            # Total emitted mass for this segment (Eq. 8): time-integral of e(t),
            # converted from concentration units via the segment's ground area.
            total_emission_mol = np.trapezoid(emission_puffs, t * 60) * 1e-6 * group_area_m2

            peak_puff = np.max(emission_puffs)
            if peak_puff > 0:
                peak_rate_mol_per_s = peak_puff * 1e-6 * group_area_m2
                emission_duration_min = (total_emission_mol / peak_rate_mol_per_s) / 60.0
            else:
                emission_duration_min = np.nan

    except Exception as e:
        # curve_fit failed for this segment (e.g. too few particles / degenerate kernel):
        # return NaNs/zeros for everything downstream so the segment is simply excluded
        # from later aggregation rather than crashing the whole peak's processing.
        out['residual_std_ppm'] = np.nan
        out['residual_sum_ppm'] = np.nan
        out['outside_before'] = outside_before
        out['outside_after'] = outside_after
        out['hist_points'] = len(group)
        out['foot_intensity'] = group.foot.sum()
        out['emission_area'] = group_area_m2
        out['group_sensitivity'] = group_sensitivity
        out['kernel_fit_params'] = popt
        out['time_minutes'] = observation.index.to_numpy()
        out['observations_ppm'] = observation[target].to_numpy()
        out['kernel_fitted'] = np.zeros_like(observation.index)
        out['duration_minutes'] = np.zeros_like(observation.index)
        out['duration_residual'] = np.zeros_like(observation.index)
        out['duration_residual_std_ppm'] = np.nan
        out['emission_mol'] = np.nan
        out['emission_duration_min'] = np.nan
        return out

    # Step 1 fit residual and its standard deviation -> rho_s (Eq. 6), the quantity
    # whose minimum across segments identifies r_max (done downstream in the notebook).
    residual = (fun(observation.index, *popt) - observation[target]).to_numpy()
    stdout = np.nanstd(residual)
    res_sum = np.nansum(np.abs(residual))

    out['residual_std_ppm'] = stdout
    out['residual_sum_ppm'] = res_sum
    out['outside_before'] = outside_before
    out['outside_after'] = outside_after
    out['hist_points'] = len(group)
    out['foot_intensity'] = group.foot.sum()
    out['emission_area'] = group_area_m2
    out['group_sensitivity'] = group_sensitivity
    out['kernel_fit_params'] = popt
    out['time_minutes'] = observation.index.to_numpy()
    out['observations_ppm'] = observation[target].to_numpy()
    out['kernel_fitted'] = fun(observation.index, *popt)
    if emission_duration:
        out['duration_minutes'] = emission_puffs
        out['duration_residual'] = duration_residual
        out['duration_residual_std_ppm'] = np.nanstd(duration_residual)
        out['emission_mol'] = total_emission_mol
        out['emission_duration_min'] = emission_duration_min
    else:
        out['duration_minutes'] = np.zeros_like(observation.index)
        out['duration_residual'] = np.zeros_like(observation.index)
        out['duration_residual_std_ppm'] = np.nan
        out['emission_mol'] = np.nan
        out['emission_duration_min'] = np.nan

    return out


# =========================================================================
# 7. Run I/O: caching the (slow) Step 1/2 results to disk
# =========================================================================
def save_results(
    filepath,
    all_fits,
    fitted_by_peak,
    trajectories,
    fitted_to_obs,
    radius_steps_km,
    delta_alpha_deg,
    source_altitude_segmentation,
    selected_peaktime_grid,
    config=None,
):
    """Pickle every result of one notebook run (all processed peaks) to `filepath`,
    so Part 1 (the slow trajectory/fitting step) doesn't need to be re-run every time
    Part 2 (figure generation) is edited. See `load_results` for the counterpart.

    `config` is stored alongside the results purely for provenance/reproducibility
    (so a saved run can later be traced back to the `config.json` that produced it).
    """
    payload = {
        'all_fits': all_fits,
        'fitted_by_peak': fitted_by_peak,
        'trajectories': trajectories,
        'fitted_to_obs': fitted_to_obs,
        'radius_steps_km': radius_steps_km,
        'delta_alpha_deg': delta_alpha_deg,
        'source_altitude_segmentation': source_altitude_segmentation,
        'selected_peaktime_grid': selected_peaktime_grid,
        'config': config,
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Results saved: {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")


def load_results(filepath):
    """Load a pickle previously written by `save_results` and return it as a dict
    with keys 'all_fits', 'fitted_by_peak', 'trajectories', 'fitted_to_obs',
    'radius_steps_km', 'delta_alpha_deg', 'source_altitude_segmentation',
    'selected_peaktime_grid', 'config'."""
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        payload = pickle.load(f)

    print(f"Results loaded: {filepath}")
    print(f"  all_fits shape: {payload['all_fits'].shape}")
    print(f"  fitted_by_peak: {len(payload['fitted_by_peak'])} peaks")
    print(f"  selected_peaktime_grid: {payload['selected_peaktime_grid']}")

    return payload


def dual_time_formatter_factory(utc_offset_hours):
    """Return a matplotlib tick formatter showing 'HH:MM (local HH:MM)' labels, e.g.
    for x-axes in UTC where a local-time reference is also useful (paper's site was
    on Pacific Daylight Time, UTC-7). Usage:
        ax.xaxis.set_major_formatter(FuncFormatter(dual_time_formatter_factory(-7)))
    """
    import matplotlib.dates as mdates

    def _formatter(x, pos):
        dt_utc = mdates.num2date(x).replace(tzinfo=None)
        dt_local = dt_utc + pd.Timedelta(hours=utc_offset_hours)
        return f"{dt_utc.strftime('%H:%M')}\n({dt_local.strftime('%H:%M')} local)"

    return _formatter


# =========================================================================
# 8. Wind-speed correction and unit conversion (paper Sect. 2.7.2, Appendix H)
# =========================================================================
def read_berkeley_met(path, plot=False):
    """Read a local meteorological station CSV (paper's LBNL1 station, used to
    correct the modelled wind speed, Sect. 2.7.2 / Appendix H). Expects the
    two-row header format of the Berkeley Lab MesoWest export (columns
    'wind_speed_set_1', 'wind_direction_set_1', 'pressure_set_1', 'Date_Time', ...).

    This file is site-specific meteorological station data and is NOT part of the
    published demo dataset; supply your own station CSV (or skip the wind-speed
    correction entirely) if applying this to a different site.

    Returns a DataFrame with an added 'minutes' column (minutes since the first
    timestamp in the file) for easy interpolation against trajectory time.
    """
    met = pd.read_csv(path, header=[6, 7])
    met.columns = met.columns.droplevel(1)
    met['pressure_hpa'] = met.pressure_set_1 / 0.029529983071445

    met['date'] = [datetime.datetime.strptime(x, '%m/%d/%Y %H:%M UTC') for x in np.array(met['Date_Time'])]
    ref_date = met['date'][0]
    met['hour'] = [(x - ref_date).total_seconds() / 60 / 60 for x in met.date]
    met['minutes'] = met['hour'] * 60

    return met


def apply_wind_speed_correction(
    df,
    met_path,
    reference_candidate='LBNL1',
    candidate_col='candidate',
    peak_time_col='peak_time',
    modeled_wind_col='wind_speed',
    emission_cols=None,
    suffix='_wcorr',
    drop_reference=True,
    inplace=False,
):
    """Correct retrieved emission strengths for the mismatch between modelled
    (HRRR/STILT) and observed wind speed (paper Sect. 2.7.2, Eq. 9; Appendix H).

    Emission strength scales approximately inversely with wind speed at fixed
    transport geometry (Eq. 9), so a per-peak correction factor
    `observed_wind_speed / modelled_wind_speed`, measured at a reference site with a
    real meteorological station (`reference_candidate`, e.g. 'LBNL1'), is applied
    multiplicatively to every candidate's emission columns for that peak.

    Parameters
    ----------
    df : DataFrame
        Long-format table with one row per (candidate, peak_time), containing
        `candidate_col`, `peak_time_col`, `modeled_wind_col`, and the emission
        columns to be corrected.
    met_path : str
        Path to the local meteorological station CSV (see `read_berkeley_met`).
    reference_candidate : str
        Value of `candidate_col` that represents the meteorological reference site
        itself (not a real emission candidate), whose modelled-vs-observed wind
        speed ratio defines the correction factor applied to all other candidates.
    drop_reference : bool
        If True (default), drop the reference site's own row(s) from the output,
        since it is not an emission candidate.
    emission_cols : list of str, optional
        Columns to correct. Defaults to any column named 'emission_mol',
        'emission_mean', 'emission_median', 'emission_min', 'emission_max', or
        ending in '_g_s', '_kg_peak', or '_tCH4_yr'.

    Returns
    -------
    df (or a copy) with added columns '<col><suffix>' for each corrected emission
    column, plus 'wind_speed_measured' and 'wind_speed_corr_factor'.
    """
    met = read_berkeley_met(path=met_path)
    met_spd = lambda x: np.interp(x, met.minutes, met.wind_speed_set_1)

    out = df if inplace else df.copy()

    ref = out[out[candidate_col] == reference_candidate].copy()
    if ref.empty:
        raise ValueError(f"Reference candidate '{reference_candidate}' not found in '{candidate_col}'.")

    ref['wind_speed_measured'] = met_spd(ref[peak_time_col])
    ref['wind_speed_corr_factor'] = ref['wind_speed_measured'] / ref[modeled_wind_col]

    # Correction factor per peak_time (from the reference site), broadcast to all candidates.
    corr_map = ref.set_index(peak_time_col)['wind_speed_corr_factor']
    out['wind_speed_corr_factor'] = out[peak_time_col].map(corr_map)

    if emission_cols is None:
        emission_cols = [
            c for c in out.columns
            if c in ['emission_mol', 'emission_mean', 'emission_median', 'emission_min', 'emission_max']
            or c.endswith(('_g_s', '_kg_peak', '_tCH4_yr'))
        ]

    for col in emission_cols:
        out[f'{col}{suffix}'] = out[col] * out['wind_speed_corr_factor']

    if drop_reference:
        out = out[out[candidate_col] != reference_candidate].reset_index(drop=True)

    return out


def convert_emissions(
    df,
    mol_cols=('emission_median', 'emission_min', 'emission_max'),
    duration_col='emission_duration_min',
    duration_minutes=None,
    m_ch4=16.04,
    inplace=False,
):
    """Convert CH4 emission estimates from mol per peak event to g/s, kg per peak,
    and t CH4/yr (the units used in the paper's Table 1 / Fig. 5).

    The t CH4/yr figure assumes continuous emission at the retrieved peak rate and
    is used only for order-of-magnitude comparison with literature source strengths
    (see the paper's caution about this assumption in Sect. 3 / Table 1 caption) --
    it is not a claim that the source emits continuously.

    Parameters
    ----------
    df : DataFrame
        Must contain the columns in `mol_cols` (emitted amount in mol per peak event).
    duration_col : str, optional
        Column with the peak duration in minutes, used row-wise to convert mol/peak
        to g/s. If None, `duration_minutes` (a single fixed value) is used for every row.
    duration_minutes : float, optional
        Fixed peak duration in minutes, used only if `duration_col` is None.
    m_ch4 : float
        Molar mass of CH4 in g/mol (default: 16.04).

    Returns
    -------
    df (or a copy) with added columns '<col>_g_s', '<col>_kg_peak', '<col>_tCH4_yr'
    for each column in `mol_cols`.
    """
    SECONDS_PER_YEAR = 365.25 * 24 * 3600

    if duration_col is None and duration_minutes is None:
        raise ValueError("Provide either duration_col or duration_minutes (needed to convert to g/s).")

    out = df if inplace else df.copy()

    if duration_col is not None:
        dur_s = out[duration_col].astype(float) * 60.0
    else:
        dur_s = pd.Series(duration_minutes * 60.0, index=out.index)

    for col in mol_cols:
        mass_g = out[col] * m_ch4  # mol -> g (total mass per peak)
        out[f'{col}_kg_peak'] = mass_g / 1000.0
        out[f'{col}_g_s'] = mass_g / dur_s
        out[f'{col}_tCH4_yr'] = out[f'{col}_g_s'] * SECONDS_PER_YEAR / 1e6  # g/s -> t/yr

    return out


# =========================================================================
# 9. Candidate-source matching (paper Sect. 3, Table 1 / Fig. 5)
# =========================================================================
def match_candidate(fitted, lat, lon, la_0, lo_0, hist_points_min=500, bearing_delta=0):
    """Find the upwind segment(s) in `fitted` (one peak's per-segment fit results)
    that geometrically match a candidate source location (lat, lon), given the
    receptor position (la_0, lo_0).

    A candidate matches a segment if its distance from the receptor falls inside
    the segment's radial bin AND its bearing falls inside the segment's angular bin
    (optionally widened by `bearing_delta` degrees on each side, to tolerate some
    wind-direction variability). Segments with fewer than `hist_points_min` particle
    timesteps are discarded as too poorly sampled to trust (consistent with the
    `hist_points` filtering used throughout the pipeline, e.g. in Figure 4).

    Parameters
    ----------
    fitted : DataFrame
        Per-segment fit results for one peak (a subset of `all_fits`/`fitted_by_peak`),
        with columns 'recep_dist_segmentation' and 'recep_bearing_segmentation'
        (pandas Interval columns from the upwind-domain segmentation).
    lat, lon : float
        Candidate source location.
    la_0, lo_0 : float
        Receptor location.
    hist_points_min : int
        Minimum number of particle timesteps for a segment to be considered
        well-sampled enough to use.
    bearing_delta : float
        Half-width (degrees) of the bearing acceptance window around the
        candidate's exact bearing. 0 = strict match to the segment containing the
        exact bearing; 180 = accept any bearing (only distance is checked).

    Returns
    -------
    (matched, d_km, bearing) : (DataFrame, float, float)
        `matched` is the subset of `fitted` whose segment contains the candidate;
        `d_km` and `bearing` are the candidate's distance (km) and bearing (deg)
        from the receptor.
    """
    def _to_segments(lo, hi):
        """Split a circular interval [lo, hi) (mod 360) into 1-2 linear segments."""
        lo, hi = lo % 360, hi % 360
        if lo < hi:
            return [(lo, hi)]
        elif lo > hi:
            return [(lo, 360.0), (0.0, hi)]
        else:
            # lo == hi: treat as either empty or the full circle -> here, full circle.
            return [(0.0, 360.0)]

    def _linear_overlap(a, b):
        return a[0] < b[1] and b[0] < a[1]

    def circular_overlap(lo1, hi1, lo2, hi2):
        segs1 = _to_segments(lo1, hi1)
        segs2 = _to_segments(lo2, hi2)
        return any(_linear_overlap(s1, s2) for s1 in segs1 for s2 in segs2)

    d_km = haversine_distance(la_0, lo_0, lat, lon) / 1000
    bearing = calculate_initial_bearing(la_0, lo_0, lat, lon)

    def in_dist(iv):
        return pd.notna(iv) and iv.left <= d_km < iv.right

    full_circle = (2 * bearing_delta) >= 360  # special case: window covers the whole circle

    def in_bearing_window(iv):
        if pd.isna(iv):
            return False
        if full_circle:
            return True
        lo_w = bearing - bearing_delta
        hi_w = bearing + bearing_delta
        return circular_overlap(lo_w, hi_w, iv.left, iv.right)

    mask = fitted['recep_dist_segmentation'].apply(in_dist) & fitted['recep_bearing_segmentation'].apply(in_bearing_window)
    matched = fitted[mask]
    matched = matched[matched['hist_points'] > hist_points_min]
    return matched, d_km, bearing


# =========================================================================
# 10. KMZ export for Google Earth (paper Fig. 6)
# =========================================================================
def rgba_to_kml_color(rgba):
    """Convert an RGBA color (0..1 floats, matplotlib convention) to a KML color
    string in 'aabbggrr' hex format."""
    r, g, b, a = [int(255 * x) for x in rgba]
    return f'{a:02x}{b:02x}{g:02x}{r:02x}'


def make_colorbar_png(path, cmap_name='cool', label='normalised fit residual\n(0 = best fit, 1 = worst)'):
    """Render a standalone colorbar image (white background, for good contrast against
    both satellite imagery and the point cloud) to embed as a KML ScreenOverlay legend."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(1.4, 4.2), dpi=200)
    cmap = plt.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label(label, fontsize=9, color='black')
    cb.ax.tick_params(labelsize=8, color='black', labelcolor='black')
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(1.2)
    fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                      facecolor='white', alpha=0.9, zorder=-1))
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    return path


def add_colorbar_overlay(kml, png_path):
    """Attach the colorbar PNG as a fixed-position KML ScreenOverlay (top-right of
    the viewport, independent of the 3D camera angle)."""
    import simplekml

    overlay = kml.newscreenoverlay(name='CH4 fit residual scale')
    overlay.icon.href = kml.addfile(png_path)
    overlay.overlayxy = simplekml.OverlayXY(x=1, y=1, xunits=simplekml.Units.fraction, yunits=simplekml.Units.fraction)
    overlay.screenxy = simplekml.ScreenXY(x=0.98, y=0.98, xunits=simplekml.Units.fraction, yunits=simplekml.Units.fraction)
    overlay.size = simplekml.Size(x=0.14, y=0, xunits=simplekml.Units.fraction, yunits=simplekml.Units.fraction)
    return overlay


def add_candidate_markers(kml, locations, keys, color, icon_href, la_0, lo_0):
    """Add candidate source locations as their own labelled KML placemarks (distinct
    from the fit-residual point cloud), scaled up slightly with distance from the
    receptor so far-away candidates remain easy to spot."""
    folder = kml.newfolder(name='Candidate source locations')
    for key in keys:
        lat, lon, desc = locations[key]
        distance = haversine_distance(lat, lon, la_0, lo_0)
        size_factor = max([1, distance / 400])
        pnt = folder.newpoint(name=key, coords=[(lon, lat, 2)])
        pnt.description = desc
        pnt.altitudemode = 'relativeToGround'
        pnt.style.iconstyle.scale = 1.8 * size_factor
        pnt.style.iconstyle.icon.href = icon_href
        pnt.style.iconstyle.color = color
        pnt.style.labelstyle.scale = 1.5 * size_factor
        pnt.style.labelstyle.color = 'ffffffff'
    return folder


def build_kmz(fitted_by_peak, base, lo_0, la_0, source_locations, candidates_to_plot,
              quantile_target='duration_residual_std_ppm',
              rnorm=2, loff=0, laff=0,
              hist_points_min=None, out_dir='.'):
    """Build one KMZ file per peak (Google Earth overlay of the paper's Fig. 6): a
    point cloud of upwind segments colored by normalised fit residual (cyan = best
    fit / most likely source region, magenta = worst), plus labelled markers for the
    receptor and the candidate source locations.

    Parameters
    ----------
    fitted_by_peak : dict[float, DataFrame]
        Per-peak fit results, as produced in the main pipeline loop (keyed by
        `peaktime_grid`).
    base : pd.Timestamp
        Reference date, used to turn each peak's `peak_time` (minutes) into a
        real timestamp for the output filename.
    lo_0, la_0 : float
        Receptor longitude/latitude.
    source_locations : dict[str, (lat, lon, description)]
        Candidate source locations to plot as markers (paper's SOURCE_LOCATIONS).
    candidates_to_plot : list of str
        Subset of `source_locations` keys to actually render as markers.
    quantile_target : str
        Column used for the point-cloud color scale (paper Fig. 6 uses the Step 1
        kernel-only fit residual, 'residual_std_ppm'; 'duration_residual_std_ppm'
        uses the Step 2 kernel*emission residual instead).
    rnorm : float
        Only segments within `rnorm` km of the receptor are used to set the
        color-scale's min/max (keeps the color scale meaningful near the source,
        instead of being stretched by far-away, poorly-constrained segments).
    hist_points_min : int, optional
        Minimum particle count per segment to include (see `match_candidate`).
    out_dir : str
        Output directory for the .kmz files and the colorbar legend PNG.

    Returns
    -------
    List of paths to the saved .kmz files (one per peak).
    """
    import matplotlib.pyplot as plt
    import simplekml

    os.makedirs(out_dir, exist_ok=True)
    colorbar_png = make_colorbar_png(os.path.join(out_dir, 'colorbar_legend.png'))
    cmap = plt.colormaps['cool']

    CANDIDATE_ICON = 'http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png'
    FIT_POINT_ICON = 'http://maps.google.com/mapfiles/kml/shapes/dot.png'
    RECEPTOR_ICON = 'http://maps.google.com/mapfiles/kml/shapes/star.png'
    CANDIDATE_COLOR = simplekml.Color.yellow
    RECEPTOR_COLOR = simplekml.Color.yellow

    saved_paths = []
    for key, mes_df in fitted_by_peak.items():
        mes_df = mes_df.copy()
        if hist_points_min is not None:
            mes_df = mes_df[mes_df['hist_points'] > hist_points_min]

        ctime = base + pd.Timedelta(minutes=float(mes_df['peak_time'].iloc[0]))

        mn = np.nanmin(mes_df[mes_df.recep_dist_km < rnorm][quantile_target])
        mx = np.nanmax(mes_df[mes_df.recep_dist_km < rnorm][quantile_target])
        mes_df['color_norm'] = (mes_df[quantile_target] - mn) / (mx - mn)
        mes_df.loc[(mes_df.color_norm > 1), 'color_norm'] = 1

        kml = simplekml.Kml()
        add_colorbar_overlay(kml, colorbar_png)

        seg = kml.newpoint(name='Receptor', coords=[(lo_0 + loff, la_0 + laff, 2)])
        seg.altitudemode = 'relativeToGround'
        seg.style.iconstyle.scale = 3
        seg.style.iconstyle.icon.href = RECEPTOR_ICON
        seg.style.iconstyle.color = RECEPTOR_COLOR

        fit_folder = kml.newfolder(name='Fit residual points')
        for i in range(len(mes_df)):
            coords = [(mes_df.iloc[i].long + loff, mes_df.iloc[i].lati + laff, mes_df.iloc[i].zagl)]
            rgba = cmap(mes_df.color_norm.iloc[i])
            kml_color = rgba_to_kml_color(rgba)

            seg = fit_folder.newpoint(coords=coords)
            seg.altitudemode = 'relativeToGround'
            seg.style.iconstyle.scale = 3
            seg.style.iconstyle.icon.href = FIT_POINT_ICON
            seg.style.iconstyle.color = kml_color

        add_candidate_markers(kml, source_locations, candidates_to_plot,
                               CANDIDATE_COLOR, CANDIDATE_ICON, la_0, lo_0)

        del mes_df
        kmz_path = os.path.join(out_dir, "Source_fit_residual_%s.kmz" % (f"{ctime:%H:%M}_UTC".replace(':', '_')))
        kml.savekmz(kmz_path)
        saved_paths.append(kmz_path)
        print(f"saved: {kmz_path}")

    return saved_paths
