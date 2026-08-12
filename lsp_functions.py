"""
Consolidated function library for the Local Source Projection (LSP) demo.
Functions extracted verbatim (or with minor, explicitly marked adaptations)
from the authors' original research notebook (LSP.ipynb, cell 0), keeping
the trajectory / intermediate data structures unchanged as requested.
"""
import os
import json
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Observation loading (modern parquet retrieval bundle) + peak finding
# -----------------------------------------------------------------------
def read_obs_proffast_parquet(path, date, species='CH4', quantile=0.1, roll_time='60min',
                               location_id=None, quality_flag_value=0, utc_offset_hours=-7):
    """Reads observations from a modern EM27 retrieval-bundle parquet (GGG2020/PROFFAST 2.4
    style: clean column names, tz-aware 'utc' timestamp), and computes the rolling-quantile
    background and enhancement -- the parquet-native counterpart of the original notebook's
    `read_obs_proffit`, which expected a raw PROFFAST `comb_invparms_*.csv` with
    space-prefixed column names (' XCH4', ' appSZA', ...) instead.

    Differences from the original `read_obs_proffit`, both confirmed with the author:
      - No XAIR/0.99775 empirical correction is applied here: this retrieval bundle is
        already corrected, unlike the older comb_invparms CSV format.
      - `quality_flag_value`: rows are kept where `event_data_quality_flag == quality_flag_value`.
        The convention (0 = good vs. 0 = bad) was NOT independently confirmed for this bundle
        format -- verify before relying on this filter for data where the flag actually varies
        (in the reference file used during development, all rows had flag == 0, so the filter
        was a no-op either way).

    IMPORTANT: `date` is matched against the LOCAL date (`utc_offset_hours`), not the UTC
    calendar date. A multi-day bundle's UTC-day boundary falls in the middle of local
    afternoon/evening, so naive UTC-date filtering pulls in the tail of the *previous*
    local day's measurement session (observed directly during development: filtering by
    UTC date included ~40 extra minutes of the prior evening, appearing as spurious peaks
    at 00:00-00:40 UTC). Filtering by local date removes this.

    Parameters
    ----------
    path : str
        Path to the retrieval-bundle parquet file.
    date : str or datetime-like
        Local date to select (the bundle may span multiple days).
    location_id : str, optional
        If given, additionally filters to this location_id (a physical site may be recorded
        under different location_id labels across the campaign, e.g. after a database
        migration -- check the unique `location_id` values per date before assuming they
        match a site label used elsewhere in your pipeline).
    utc_offset_hours : float
        Local time zone offset from UTC, used only to determine the local calendar date
        boundary (e.g. -7 for PDT). Does not affect the returned timestamps, which remain UTC.
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

    # already-corrected retrieval: no XAIR-based correction applied (see docstring)
    obs['XCH4_c'] = obs['XCH4']
    obs['XCO2_c'] = obs['XCO2']

    xb = obs[['XCH4', 'CH4', 'XAIR', 'H2O', 'XCO2', 'XCH4_c', 'XCO2_c']].rolling(
        roll_time, min_periods=1, center=True
    ).quantile(quantile, interpolation='linear')

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

    nan_cont_index = np.cumsum(~np.isnan(obs['Enh_ppm'])) - 1
    chunk_index = np.concatenate(([0], np.cumsum(np.diff(np.isnan(obs['Enh_ppm'])))))
    obs['nan_cont_index'] = nan_cont_index
    obs['nan_chunk_index'] = chunk_index

    cols = ['Enh_ppm', 'Enh_c_ppm', 'obs_raw', 'minutes', 'quantile', 'nan_cont_index', 'nan_chunk_index',
            'xair', 'quantile_xair', 'h2o_molec', 'h2o_molec_quantile', 'molec_m2', 'quantile_molec_m2',
            'Enh_molec_m2', 'XCO2', 'XCO2_c', 'XCO2_quantile', 'XCO2_c_quantile', 'sza', 'azi']
    return obs[cols]


def find_group_peaks(group, key, prominence):
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
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx - 1]) < np.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit, least_squares
from pvlib.solarposition import get_solarposition


# -----------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon coordinates in meters."""
    R = 6371000
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    """Bearing in degrees between two lat/lon coordinates (north=0, east=90)."""
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.degrees(np.arctan2(y, x))
    return (initial_bearing + 360) % 360


def circular_mean(deg_arr, phase=np.pi, what='mean'):
    x = np.sin(np.pi / 180 * deg_arr + phase)
    y = np.cos(np.pi / 180 * deg_arr + phase)
    if what == 'mean':
        X, Y = np.nanmean(x), np.nanmean(y)
    elif what == 'median':
        X, Y = np.nanmedian(x), np.nanmedian(y)
    return 180 / np.pi * (np.arctan2(X, Y) + phase)


# -----------------------------------------------------------------------
# Atmosphere / averaging kernel helpers
# -----------------------------------------------------------------------
def pressure_std_atm(alt, p0=1013.15, z0=0, H=7800):
    return p0 * np.exp(-(alt - z0) / H)


def molec_column(alt, p0=1013.15, z0=0, H=7800, A=1, g=9.81, m_molar=28.97 / 1000):
    return pressure_std_atm(alt, p0, z0, H) * A / g / m_molar


def load_ak(filepath: str, as_fun=True):
    assert filepath.endswith(".json"), "Filepath must end with .json"
    with open(filepath, "r") as f:
        d = json.load(f)
    if as_fun:
        return RegularGridInterpolator(
            (d['szas'], d['pressures']), d['aks'], bounds_error=False, fill_value=None
        )
    return pd.DataFrame(data=d['aks'], index=d['szas'], columns=d['pressures'])


# -----------------------------------------------------------------------
# Trajectory loading / interpolation
# -----------------------------------------------------------------------
def calculate_wind_vectors(df, group_id=None):
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
    df['wind_dir_deg'] = (bearing + 180) % 360
    if group_id is not None:
        df['indx'] = group_id
    return df


def interpolate_particle_trajectories(x, hi_res_time_resolution, plot=False):
    """Interpolates particle trajectories (grouped by 'indx') onto a finer time grid."""
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
    """Reads one 'about.json' + 'traj/particle_stilt.<i>.parquet' trajectory product.

    Adapted from the original notebook: the averaging-kernel path is now a
    function argument (`averaging_kernel_path`) instead of a hardcoded
    Windows path, so it can be supplied via config.
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
        traj['full_weight_per_trajectory'] = (
            release_density_weight[i] / particles_per_release_point * ak((site_sza, release_pressures[i]))
        )

        traj.sort_values(by=['indx', 'time'], ascending=[True, False], inplace=True)
        traj.index = range(len(traj))

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


# -----------------------------------------------------------------------
# Radial / angular segmentation
# -----------------------------------------------------------------------
def create_radius(drf, dr=0.001, rad_max=100, max_steps=50, d_alpha=False):
    """Creates exponentially increasing radial steps."""
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
    deg_step = x['dalpha'].mean()
    x['deg_step'] = deg_step
    med = circular_mean(x.recep_bearing_deg, what='median')
    delta = med - np.floor(med / deg_step) * deg_step + deg_step / 2
    segmentation = np.arange(start=-deg_step * 2, stop=360 + deg_step * 2, step=deg_step) + delta
    x['recep_bearing_segmentation'] = pd.cut(x['recep_bearing_deg'], segmentation, right=False)
    return x[['recep_dist_km', 'recep_bearing_deg', 'recep_bearing_segmentation', 'deg_step']]


def ring_area(dist_km, bearing_steps):
    steps = np.array([b.right - b.left for b in bearing_steps])
    dmax = np.array([d.right for d in dist_km])
    dmin = np.array([d.left for d in dist_km])
    ring_area_km_2 = np.pi * ((dmax) ** 2 - (dmin) ** 2)
    return ring_area_km_2 * steps / 360 * 1000 * 1000


# -----------------------------------------------------------------------
# Step 1 + Step 2: kernel fit & emission retrieval (paper Eq. 5-8)
# -----------------------------------------------------------------------
def make_histogram(group, observation, peak_time,
                    std_dev=1, weight=True, plot=False, max_plot_radius=3.0,
                    color='orange', target='Enh_ppm', debug=False,
                    emission_duration=False, check_group=True,
                    hi_res_time_resolution=1 / 30):
    """Fits the transport kernel of one upwind segment to the observed peak
    (Step 1, paper Eq. 5-6) and, if emission_duration=True, deconvolves the
    emission time series (Step 2, paper Eq. 7-8)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    m_air = 28.97 / 1000
    group['foot_source'] = hi_res_time_resolution * 60 * m_air / (group['dens'] * group['dz_source'])
    group['foot_recep'] = group['foot_source'] * group['full_weight_per_trajectory']
    group['group_vertical_weight'] = group['foot_recep']
    group.loc[(np.isnan(group['group_vertical_weight']), 'group_vertical_weight')] = 0

    out = pd.concat((group.mean(numeric_only=True), group.std(numeric_only=True).add_suffix('_std')), axis=0)
    if check_group and np.sum(group['segment_area_m2'] != group['segment_area_m2'].iloc[0]) > 0:
        print('Warning: multiple segments in group!')

    group_area_m2 = group['segment_area_m2'].iloc[0]
    group_sensitivity = group['group_vertical_weight'].sum()

    c_bins = np.concatenate(([-np.inf], np.arange(-30, 30, hi_res_time_resolution * 2), [np.inf]))
    counts, t_bins = np.histogram(group['time'] - group['time'].median(), bins=c_bins,
                                   weights=group['group_vertical_weight'])
    counts = counts / 2
    outside_before, outside_after = counts[0], counts[-1]
    t_bins, counts = t_bins[1:-1], counts[1:-1]
    midx = np.argmax(counts)
    tbins = t_bins[:-1] + np.diff(t_bins) / 2
    tbins_obs = -(tbins - tbins[midx]) + peak_time
    kernel = lambda minutes: np.interp(minutes, tbins_obs[::-1], counts[::-1])

    fun = lambda x, s, o, p: kernel(x + p) * s + o
    A0 = observation[target].max() / np.max(counts)
    p0 = [A0, 0, 0]
    bounds = [(1e-10, -A0 / 4, -1), (A0 * 100, A0 / 4, 1)]
    popt = p0

    try:
        if weight:
            w = 1.1 - np.exp(-((observation.index - peak_time) / std_dev) ** 2)
            popt, pcov = curve_fit(fun, observation.index, observation[target], p0=p0, sigma=w,
                                    bounds=bounds, check_finite=True)
        else:
            popt, pcov = curve_fit(fun, observation.index, observation[target], p0=p0,
                                    bounds=bounds, check_finite=True)

        emission_puffs = np.zeros_like(observation.index, dtype=float)
        emission_duration_min = np.nan  # NEU: default falls emission_duration=False oder Peak=0

        if emission_duration:
            t = observation.index
            xx = observation[target].to_numpy() - popt[1]
            fit_scf = 1e5
            h = kernel(observation.index + popt[2]) * fit_scf

            def fwd_fun(e):
                return np.convolve(h, e, mode='same')

            def residual_fun(e):
                return np.abs(xx - fwd_fun(e))

            p0e = np.zeros_like(h)
            bounds_e = [-1e-15, np.inf]
            E = least_squares(residual_fun, p0e, bounds=bounds_e)
            emission_puffs = E.x.copy() * fit_scf
            duration_residual = residual_fun(E.x)
            enhancement_fwd = fwd_fun(E.x)
            total_emission_mol = np.trapezoid(emission_puffs, t * 60) * 1e-6 * group_area_m2
            
            peak_puff = np.max(emission_puffs)
            if peak_puff > 0:
                peak_rate_mol_per_s = peak_puff * 1e-6 * group_area_m2
                emission_duration_min = (total_emission_mol / peak_rate_mol_per_s) / 60.0
            else:
                emission_duration_min = np.nan

    except Exception as e:
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
