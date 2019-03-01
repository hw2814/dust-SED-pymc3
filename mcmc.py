#!/usr/bin/python3
"""
Run MCMC fitting MBB spectrum to flux measurements.
Run on the command line to produce results files of temperature, mass and emissivity, or import
functions to run in a notebook.
"""
from argparse import ArgumentParser
from math import ceil
import numpy as np
from os.path import join
from typing import Tuple

import pandas as pd
import pymc3 as pm
from pymc3.backends.base import MultiTrace
import theano.tensor as tt
import uncertainties.unumpy as unp

TEMP_BOUNDS = (5, 80)
BETA_BOUNDS = (0, 4)
MASS_BOUNDS = (3, 12)

FLUX_COLS = ['F60', 'F70', 'F100', 'F160', 'F250', 'F350', 'F500']
ERR_COLS = ['sigma60', 'sigma70', 'sigma100', 'sigma160', 'sigma250', 'sigma350', 'sigma500']


class Constants:
    h = 6.62607e-34
    c = 2.998e8
    k_b = 1.38065e-23
    knu0 = 0.192
    nu0 = 856.5e9
    m_sun = 1.989e30
    H0 = 73


constants = Constants()


def get_measurement_wavelengths(df: pd.DataFrame) -> np.array:
    """From column names of non-empty flux columns, determine the measurement wavelengths"""
    if isinstance(df, pd.Series):
        cols = [int(col[1:]) for col in df[FLUX_COLS].dropna().index]
    else:
        cols = [int(col[1:]) for col in df[FLUX_COLS].dropna(how='any', axis=1).columns]
    return 1e-6 * np.array(cols)


def get_measurement_frequencies(df: pd.DataFrame) -> np.array:
    """Determines measurement frequencies from non-empty flux columns, see get_measurement_wavelengths"""
    wls = get_measurement_wavelengths(df)
    return constants.c / wls


def identify_flux_measurements(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From a dataframe, identify flux measurement and error columns.
    Returns dataframe slices of the flux and error data
    """
    if isinstance(df, pd.Series):
        fluxes = df[FLUX_COLS].dropna()
        errors = df[ERR_COLS].dropna()
    else:
        fluxes = df[FLUX_COLS].dropna(how='any', axis=1)
        errors = df[ERR_COLS].dropna(how='any', axis=1)
    return fluxes, errors


def flux_func(nu_obs, temp, beta, mass, d, z):
    """
    Return spectral energy distribution of a galaxy as a modified black body in units of mJy parameterised by:
    :param nu_obs: the observed frequency of flux measurements
    :param temp: dust temperature
    :param beta: the emissivity parameter
    :param mass: the mass of dust in the galaxy
    :param d: the distance to the galaxy
    :param z: the redshift of the galaxy
    """
    h = constants.h
    c = constants.c
    k_b = constants.k_b
    knu0 = constants.knu0
    nu0 = constants.nu0
    m_sun = constants.m_sun
    if isinstance(temp, pm.model.TransformedRV):
        nu = (1 + z)[:, None] * nu_obs[None, :]
        return 1e29 * (m_sun * 10 ** mass[:, None] / (d ** 2)[:, None]) * knu0 * (nu / nu0) ** beta[:, None] * (
                2 * h * nu ** 3 / c ** 2) * 1 / (tt.exp(h * nu / (k_b * temp[:, None])) - 1)
    elif isinstance(temp, np.ndarray):
        if not isinstance(z, np.ndarray):
            z = np.array([z])
        if not isinstance(d, np.ndarray):
            d = np.array([d])
        nu = (1 + z)[:, None] * nu_obs[None, :]
        return 1e29 * (m_sun * 10 ** mass[:, None] / (d ** 2)[:, None]) * knu0 * (nu / nu0) ** beta[:, None] * (
                2 * h * nu ** 3 / c ** 2) * 1 / (unp.exp(h * nu / (k_b * temp[:, None])) - 1)
    else:
        nu = (1 + z) * nu_obs
        return 1e29 * (m_sun * 10 ** mass / (d ** 2)) * knu0 * (nu / nu0) ** beta * (2 * h * nu ** 3 / c ** 2) * 1 / (
                unp.exp(h * nu / (k_b * temp)) - 1)


def create_args():
    """Create command line arguments"""
    options = ArgumentParser(description=__doc__, usage="python mcmc.py FILENAME [options]")
    options.add_argument('filename', action='store', nargs='?', default=join('data', 'flux_measurements.csv'), type=str,
                         help="filename of input data. Default 'data/flux_measurements.csv'")
    options.add_argument('-n', '--n-sample', action='store', default=15000, type=int, dest='n_sample',
                         help='Number of samples to take. Default 15000')
    options.add_argument('-c', '--chains', action='store', default=3, type=int, dest='chains',
                         help='Number of concurrent Markov Chains. Default: 3')
    options.add_argument('-b', '--burn', action='store', default=500, type=int, dest='burn',
                         help='Number of initial samples to burn. Default 500')
    options.add_argument('-t', '--thin', action='store', default=5, type=int, dest='thin',
                         help='Thinning level of sample (keep every n sample only). Default 5')
    options.add_argument('-ch', '--chunks', action='store', default=250, type=int, dest='chunk',
                         help='Size of data chunks to collect traces. Default 250')
    return options.parse_args()


def make_model(flux, err, nu, d, z):
    """Construct pymc3 model from parameters as in flux_func"""
    with pm.Model() as model:
        # Priors
        temp = pm.Uniform('temp', *TEMP_BOUNDS, shape=flux.shape[0])
        beta = pm.Uniform('beta', *BETA_BOUNDS, shape=flux.shape[0])
        mass = pm.Uniform('mass', *MASS_BOUNDS, shape=flux.shape[0])

        # Expected value
        f_exp = flux_func(nu, temp, beta, mass, d, z)

        # Sampling distribution
        f_obs = pm.Normal('F_obs', mu=f_exp, sd=err, observed=flux)
    return model


def get_trace(flux, err, nu, d, z, n_sample, chains, **kwargs) -> MultiTrace:
    """
    Run MCMC for flux measurements, kwargs passed to pymc3.sample
    :param flux: observed flux measurements in mJy, shape [batch x nu]
    :param err: observed uncertainty in mJy, shape [batch x nu]
    :param nu: observed frequencies, 1D array of values
    :param d: distances to objects in metres
    :param z: redshifts of objects
    :param n_sample: number of samples to take
    :param chains: number of concurrent chains
    :return: results of MCMC
    """
    model = make_model(flux, err, nu, d, z)
    with model:
        trace = pm.sample(n_sample, chains=chains, njobs=chains, init='adapt_diag', **kwargs)
        return trace


def extract_results_from_trace(index: int, trace: MultiTrace, burn: int, thin: int) -> pd.DataFrame:
    """
    Add results of MCMC to data_set
    :param index: iterable of values for index
    :param trace: pymc3.trace object, MCMC results for data set
    :param burn: number of initial samples to burn
    :param thin: multiple of samples to keep
    :return: pandas.DataFrame of input data with MCMC results
    """
    trace_results = pd.DataFrame(index=index)
    for parameter in ['temp', 'beta', 'mass']:
        values = trace.get_values(parameter, burn=burn, thin=thin)
        trace_results[f'{parameter} mean'] = values.mean(axis=0)
        trace_results[f'{parameter} std'] = values.std(axis=0)
    return trace_results


def run_mcmc(data_set: pd.DataFrame, n_sample: int, chains: int, burn: int, thin: int, chunk: int,
             **kwargs) -> pd.DataFrame:
    """
    Identifies relevant flux measurements and runs the MCMC. The dataframe is chunked since the trace
    can get very large for a large number of samples and objects. kwargs passed to pymc3.sample
    """
    fluxes, errors = identify_flux_measurements(data_set)
    nu = get_measurement_frequencies(data_set)
    n_chunks = ceil(len(data_set) / chunk)
    trace_results = []
    for n_chunk, i in enumerate(range(0, len(data_set), chunk)):
        print(f'\nRunning MCMC for {data_set["Origin"].iloc[0]}: chunk {n_chunk + 1} of {n_chunks}')
        trace = get_trace(
            flux=fluxes.iloc[i: i + chunk],
            err=errors.iloc[i: i + chunk],
            nu=nu,
            d=data_set.iloc[i: i + chunk]['D'],
            z=data_set.iloc[i: i + chunk]['z'],
            n_sample=n_sample,
            chains=chains,
            **kwargs
        )
        trace_results.append(extract_results_from_trace(data_set.iloc[i: i + chunk].index, trace, burn, thin))
    return pd.concat(trace_results)


def main_func(df: pd.DataFrame, n_sample: int, chains: int, burn: int, thin: int, chunk: int) -> dict:
    """
    Returns the results as a pandas.DataFrame
    :param df: pandas.DataFrame of objects' measurements
    :param n_sample: int, number of samples for MCMC
    :param chains: int, number of concurrent chains to run
    :param burn: number of initial samples to burn
    :param thin: multiple of samples to keep
    :param chunk: int, max size of dataframe chunks to run mcmc at a time
    :return: dict of results like data source: pd.DataFrame
    """
    # the different data sets have different frequency measurements, so must be run separately
    data_sets = {data_source: df[df['Origin'] == data_source] for data_source in df['Origin'].unique()}
    log_text = '\n'.join([f'\t{data_source}: {len(data_set)}' for data_source, data_set in data_sets.items()])
    print(f'Processing {len(df)} objects:\n{log_text}')
    print(f'MCMC will take {n_sample} samples for each source in {chains} chain(s)')
    traces = {source: run_mcmc(data_set, n_sample, chains, burn, thin, chunk) for source, data_set in data_sets.items()}
    return traces


if __name__ == '__main__':
    args = create_args().__dict__
    input_data = pd.read_csv(args.pop('filename'))
    results = main_func(input_data, **args)
    for source, data in results.items():
        filename = "-".join([source, str(args["n_sample"]), str(args["chains"]), str(args["burn"]), str(args["thin"])])
        data.to_csv(join('results', 'raw', filename + '.csv'))
