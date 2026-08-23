"""
This is the Main file for implementing the Poisson likelihood lightcurve matching algorithm locally,
without using a csv configuration file.

MAIN FILE of the workflow:
ops/NewTimeSeries.py ==> Empty Time Series
gen/GenTimeDist.py ==> Generate realistic Time Series according to models
gen/GenTimeDist_IceCube.py ==> Generate realistic histogram according to models
TimeLagGenerator ==> Generate a list of possible time lags
PairTimeSeriesToHist_Smoothing ==> Apply smoothing function to the time series / histogram
PoissonLagLikelihood_v8 ==> Applied interpolation to the lnJ table, with c++ plugin, and correction term
LikelihoodScanCollector ==> Collect all the likelihoods from different histogram pairs, make a summary

Different error-determination methods: 
sigma1: Fisher Information (second-derivative)
sigma2: 0.5 method (approximation on Gaussian fisher)
sigma3: Godambe info (can only be calculated after providing the correct J and H)

Analysing the impact of the yield ratio.
Fixing SuperK yield (7800 / 4000), then let alpha to be the ratio between the yield of det2 / SuperK's
"""

# Fit the inverse polynomial to the 1/I(\sen \lambda + \bg) with respect to \lambda 
# Introduce the "0.5 method", and compare it with the second-derivative method. 

# Make things more user-friendly on terminal
import argparse

# Make csv file to store results
import csv

import re
import sys
from pathlib import Path

import numpy as np
import scipy.stats as sc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parents[2]))

from snewpdag.dag import Node

import importlib

# Some useful tools for us to run nodes in this file: 
def load_plugin(filename):
  module_name = 'snewpdag.plugins.' + filename[:-3].replace('/', '.')
  return importlib.import_module(module_name)
def run_node(node, data):
  node.update(data)
  return node.last_data
# We need to specify the arguments in node as well. 
# If we don't use run_node, then we will run them in a chained manner
# .update() --> notify() --> save the output in .last_data
# Feed the .last_data into the next node


# Nodes:
NewTimeSeries = load_plugin('ops/NewTimeSeries.py').NewTimeSeries
NewHist1D = load_plugin('ops/NewHist1D.py').NewHist1D
Uniform = load_plugin('gen/Uniform.py').Uniform
GenTimeDist = load_plugin('gen/GenTimeDist.py').GenTimeDist
GenTimeDist_IceCube = load_plugin('gen/GenTimeDist_IceCube.py').GenTimeDist_IceCube
TimeLagGenerator = load_plugin('TimeLagGenerator.py').TimeLagGenerator
PairTimeSeriesToHist = load_plugin('PairTimeSeriesToHist.py').PairTimeSeriesToHist
PairTimeSeriesToHist_Smoothing = load_plugin('PairTimeSeriesToHist_Smoothing.py').PairTimeSeriesToHist_Smoothing
PairTimeSeriesToHist_Smoothing_Toy = load_plugin('PairTimeSeriesToHist_Smoothing_Toy.py').PairTimeSeriesToHist_Smoothing_Toy
PoissonLagLikelihood = load_plugin('PoissonLagLikelihood_v8.py').PoissonLagLikelihood_v8
LikelihoodScanCollector = load_plugin('LikelihoodScanCollector.py').LikelihoodScanCollector


# Flexible for mac / cluster
PROJECT_ROOT = Path(__file__).parents[2]

OUTPUT_DIRECTORY = PROJECT_ROOT / 'output'
MC_OUTPUT_DIRECTORY = PROJECT_ROOT / 'output' / 'mc_v10_test'

# DEFAULT PARAMETERS
TRUE_LAG = 0.022
SCAN_LOW = -0.1 
SCAN_HIGH = 0.1 
COARSE_TIME_LAG_STEP_SIZE = 0.005 
FINE_TIME_LAG_STEP_SIZE = 0.0001 
HISTOGRAM_BIN_WIDTH = 0.002 
WINDOW_START = -0.6 
WINDOW_SIZE = 9.2 
SEED = 1000

# DETECTOR RATIO:
SUPERK_YIELD = 7800 # 4000 if using s-11
SUPERK_BG = 0.1 
SUPERK_MODEL_DIREC = str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-wc.data')
SUPERK_RESOL = 1e-6
ALPHA = 0.1 # det2's yield / det1's yield 

# Godambe Information: (which will vary for different detector pairs / other parameters)
J = np.nan # ms^-2
H = np.nan # ms^-2

SMOOTHING = True
# Generate histogram for IceCube directly --> Speed up the algo A LOT
HIST1_IC = False 

# Gaussian Normal Distribution (GND) parameters:
TOY = True # Use Gaussian as our kernel
# around 5ms to 15 ms is a good choice to catch up the rising edge
SIGMA_GND = 0.005 # second 

# Convolution Smoothing parameters:
RISE_CONSTANT = 0.02
FALL_CONSTANT = 2.0
FRAC_RISE = 0.8
FRAC_FALL = 0.2
if TOY == True: 
  IMPACT_RANGE = SIGMA_GND * 5.0 
else: 
  IMPACT_RANGE = 2.0

MEAN_CORRECTION = False


def summary_builder(coarse_data, fine_data):
  """Combine coarse scan and fine scan, prepare the data for plotting the likelihood

  Coarse points inside the fine-scan interval are replaced by the denser fine
  points.  This avoids duplicate lag values and keeps every lag paired with
  the correct log-likelihood.
  """

  coarse_lags = np.asarray(coarse_data['possible_time_lag_list'], dtype=float)
  coarse_like = np.asarray(coarse_data['log_likelihood_list'], dtype=float)
  coarse_lnJ = np.asarray(coarse_data['sum_lnJ_list'], dtype=float)
  coarse_ln_fac = np.asarray(coarse_data['sum_ln_fac_list'], dtype=float)

  fine_lags = np.asarray(fine_data['possible_time_lag_list'], dtype=float)
  fine_like = np.asarray(fine_data['log_likelihood_list'], dtype=float)
  fine_lnJ = np.asarray(fine_data['sum_lnJ_list'], dtype=float)
  fine_ln_fac = np.asarray(fine_data['sum_ln_fac_list'], dtype=float)

  # Keep coarse points only outside the interval covered by the fine scan.
  coarse_outside_roi = ((coarse_lags < np.min(fine_lags)) | (coarse_lags > np.max(fine_lags)))

  # Combine everything:
  total_lags = np.concatenate((coarse_lags[coarse_outside_roi], fine_lags))
  total_like = np.concatenate((coarse_like[coarse_outside_roi], fine_like))
  total_lnJ = np.concatenate((coarse_lnJ[coarse_outside_roi], fine_lnJ))
  total_ln_fac = np.concatenate(
      (coarse_ln_fac[coarse_outside_roi], fine_ln_fac))

  sort_indices = np.argsort(total_lags)
  total_lags = total_lags[sort_indices]
  total_like = total_like[sort_indices]
  total_lnJ = total_lnJ[sort_indices]
  total_ln_fac = total_ln_fac[sort_indices]

  # Make the table: 
  total_table = np.column_stack((total_lags, total_like))

  return {
      'possible_time_lag_list': total_lags,
      'log_likelihood_list': total_like,
      'sum_lnJ_list': total_lnJ,
      'sum_ln_fac_list': total_ln_fac,
      'table': total_table,
      'scan_type': 'combined',
  }

# build an empty time series --> adding background noise --> adding model lightcurve:
def build_timeseries(true_lag, det1_yield, det1_bg, alpha, model_directory, background_window_1=(-1.0, 9.0), background_window_2=(-1.0, 9.0)):

  bg1_start, bg1_stop = background_window_1
  bg2_start, bg2_stop = background_window_2

  payload_data = {'action': 'alert'}

  if HIST1_IC == True: 
    # IceCube's hist: 
    start_empty = -1.0
    stop_empty = 9.0
    nbins_empty = int((stop_empty - start_empty)/ HISTOGRAM_BIN_WIDTH)
    payload_data = run_node(NewHist1D('hist1',
                                    nbins_empty, 
                                    start_empty,
                                    stop_empty,
                                    name = 'new-hist1'), payload_data)
    payload_data = run_node(Uniform('hist1',
                                    det1_bg,
                                    bg1_start,
                                    bg1_stop,
                                    name='bg-hist1'), payload_data)
    payload_data = run_node(GenTimeDist_IceCube('hist1',
                                                sig_filename=model_directory,
                                                sig_filetype='tn',
                                                sig_delimiter=',',
                                                sig_mean=det1_yield,
                                                sig_once=False,
                                                sig_t0=0.0,
                                                name='model-hist1'),payload_data)
  else: 
    payload_data = run_node(NewTimeSeries('ts1', name='new-ts1'), payload_data)
    payload_data = run_node(Uniform('ts1',
                                    det1_bg,
                                    bg1_start,
                                    bg1_stop,
                                    name='bg-ts1'), payload_data)
    payload_data = run_node(GenTimeDist('ts1',
                                sig_filename=model_directory,
                                sig_filetype='tn',
                                sig_delimiter=',',
                                sig_mean=det1_yield,
                                sig_once=False,
                                sig_t0=0.0,
                                name='model-ts1'), payload_data)
  
  payload_data = run_node(NewTimeSeries('ts2', name='new-ts2'), payload_data)
  payload_data = run_node(Uniform('ts2',
                                  det1_bg,
                                  bg2_start,
                                  bg2_stop,
                                  name='bg-ts2'), payload_data)
  payload_data = run_node(GenTimeDist('ts2',
                              sig_filename=model_directory,
                              sig_filetype='tn',
                              sig_delimiter=',',
                              sig_mean=det1_yield * alpha,
                              sig_once=False,
                              sig_t0=true_lag,
                              name='model-ts2'), payload_data)
  return payload_data

# THE MOST IMPORTANT FUNCTION!!!
def calculations(payload_data, det1_yield, det1_bg, alpha,
                 coarse_time_lag_step_size, fine_time_lag_step_size, histogram_bin_width,
                 window_start, window_size,coarse_scan_low,coarse_scan_high,smoothing,toy):
    
    a = det1_yield * histogram_bin_width / 8.69
    p = alpha * a 
    b = det1_bg * histogram_bin_width
    q = b

    # Searching the values of the integral computed by Wolfram Alpha from a csv file:
    csv_file = Path(__file__).with_name("continuous_poisson_integral.csv")
    mu_values = []
    integral_values = []

    with csv_file.open() as file:
            reader = csv.DictReader(file)
            for row in reader:
                mu_values.append(float(row["mu"]))
                integral_values.append(float(row["integral"]))
    
    mu_values = np.array(mu_values)
    integral_values = np.array(integral_values)
    y_values = 1 / integral_values -1 


    lambda1_values = (mu_values - b) / a
    lambda2_values = (mu_values - q) / p

    # Ignoring the division by zero and invalid operation: 
    with np.errstate(divide='ignore', invalid='ignore'):
        x1 = 1 / lambda1_values
        x2 = 1 / lambda2_values

    # Fit 1 / I(a*lambda + b) - 1 = c / lambda.  Only positive
    # lambda values belong to the physical integration domain.
    valid_indices_1 = (np.isfinite(x1)
        & np.isfinite(y_values)
        & (lambda1_values > 0.0)     
    )
    valid_indices_2 = (
        np.isfinite(x2)
        & np.isfinite(y_values)
        & (lambda2_values > 0.0)
    )

    if not np.any(valid_indices_1):
        raise ValueError('No positive-lambda calibration points are available for det1')
    if not np.any(valid_indices_2):
        raise ValueError('No positive-lambda calibration points are available for det2')

    x1_valid = x1[valid_indices_1]
    x2_valid = x2[valid_indices_2]
    y1_values_valid = y_values[valid_indices_1]
    y2_values_valid = y_values[valid_indices_2]

    M1 = np.column_stack([x1_valid])
    M2 = np.column_stack([x2_valid])

    coeffs1, *_ = np.linalg.lstsq(M1, y1_values_valid, rcond=None)
    coeffs2, *_ = np.linalg.lstsq(M2, y2_values_valid, rcond=None)

    c1_cal = coeffs1[0]
    c2_cal = coeffs2[0]
  
    print('coefficient of 1/lambda of det1 = {:.4f}'.format(c1_cal))
    print('coefficient of 1/lambda of det2 = {:.4f}'.format(c2_cal))

    # Coarse-scan's lags:
    payload_data = run_node(TimeLagGenerator('possible_time_lag_list',
                                    scan_low=coarse_scan_low,
                                    scan_high=coarse_scan_high,
                                    step_size=coarse_time_lag_step_size,
                                    name='possible_time_lag_list'),
                                    payload_data)

    # Select the histogram builder from the command-line smoothing settings.
    if not smoothing:
        histogram_pair = PairTimeSeriesToHist('ts1',
                                            'ts2',
                                            'hist_pair',
                                            hist_bin_width=histogram_bin_width,
                                            window_start=window_start,
                                            window_size=float(window_size),
                                            lag_field='lag_field',
                                            hist1_ic=HIST1_IC,
                                            in_hist1_field='hist1',
                                            name='pair')
    elif toy:
        histogram_pair = PairTimeSeriesToHist_Smoothing_Toy('ts1',
                                                            'ts2',
                                                            'hist_pair',
                                                            bin_width=histogram_bin_width,
                                                            window_start=window_start,
                                                            window_size=float(window_size),
                                                            lag_field='lag_field',
                                                            sigma_gnd=SIGMA_GND,
                                                            impact_range=IMPACT_RANGE,
                                                            cache_hist1=True,
                                                            hist1_ic=HIST1_IC,
                                                            in_hist1_field='hist1',
                                                            name='pair_smoothed_toy')
    else:
        histogram_pair = PairTimeSeriesToHist_Smoothing('ts1',
                                                        'ts2',
                                                        'hist_pair',
                                                        hist_bin_width=histogram_bin_width,
                                                        window_start=window_start,
                                                        window_size=float(window_size),
                                                        lag_field='lag_field',
                                                        rise_constant=RISE_CONSTANT,
                                                        fall_constant=FALL_CONSTANT,
                                                        sigma1=SUPERK_RESOL,
                                                        sigma2=SUPERK_RESOL,
                                                        frac_fall=FRAC_FALL,
                                                        frac_rise=FRAC_RISE,
                                                        impact_range=IMPACT_RANGE,
                                                        cache_hist1=True,
                                                        in_hist1_field='hist1',
                                                        hist1_ic=HIST1_IC,
                                                        mean_correction=MEAN_CORRECTION,
                                                        name='pair_smoothed')
    
    # new function with correction terms:
    like_cal = PoissonLagLikelihood('hist_pair',
                                    'likelihood_data',
                                    sen_1=det1_yield/8.69, # keep per second
                                    sen_2=det1_yield/8.69 * alpha,
                                    bg_1=det1_bg, # keep per second! 
                                    bg_2=det1_bg,
                                    cutoff=3.0, 
                                    c1=c1_cal,
                                    c2=c2_cal,
                                    name='like')
    
    coarse_collector = LikelihoodScanCollector('likelihood_data',
                                        'scan',
                                        scan_type='coarse',
                                        name='coarse_collector')
    fine_collector = LikelihoodScanCollector('likelihood_data',
                                        'scan',
                                        scan_type='fine',
                                        name='fine_collector')


    # Coarse-scan:
    for lag in payload_data['possible_time_lag_list']:
        temp_data = payload_data.copy()
        temp_data['lag_field'] = float(lag)
        histogram_pair.update(temp_data)
        like_cal.update(histogram_pair.last_data)
        coarse_collector.update(like_cal.last_data)

    # Feed the output from coarse_collector into TimeLagGenerator again.
    # Fine-scan's lags:
    fine_payload_data = run_node(TimeLagGenerator('possible_time_lag_list',
                                                    scan_type='fine',
                                                    fine_step=fine_time_lag_step_size,
                                                    in_roi_field='scan/roi',
                                                    name='fine_time_lag_list'),
                                                    coarse_collector.last_data) 

    # Fine-scan:
    for lag in fine_payload_data['possible_time_lag_list']:
        temp_data = fine_payload_data.copy()
        temp_data['lag_field'] = float(lag)
        histogram_pair.update(temp_data)
        like_cal.update(histogram_pair.last_data)
        fine_collector.update(like_cal.last_data)

    coarse_scan_data = coarse_collector.last_data['scan']
    fine_scan_data = fine_collector.last_data['scan']

    total_scan = summary_builder(coarse_scan_data, fine_scan_data)

    # Keep the original scans for plotting/debugging and also return one sorted
    # table spanning the complete scan range.
    return {
        'coarse': coarse_scan_data,
        'fine': fine_scan_data,
        'total': total_scan
    }

# FILE_NAME_MODIFICATION:
def format_float_for_filename(value):
  return '{:.6g}'.format(value).replace('-', 'm').replace('.', 'p')

def format_text_for_filename(value):
  text = str(value).strip()
  text = re.sub(r'[^A-Za-z0-9._-]+', '-', text)
  return text.strip('-') or 'unknown'

def build_plot_filename(alpha, seed):
  alpha = format_float_for_filename(alpha)
  return (
    'LogL_against_tau'
    '_alpha-{}'
    '_seed-{}.png'
  ).format(alpha, seed)

# Make our csv files collected in a more systematic way
def build_run_name(alpha, true_lag, coarse_time_lag_step_size,
                   fine_time_lag_step_size, histogram_bin_width, window_size,
                   scan_low, scan_high):
  return (
      'alpha-{}'
      '_true-lag-{}s'
      '_hist-bin-{}s'
      '_coarse-step-{}s'
      '_fine-step-{}s'
      '_window-{}s'
      '_scan-{}s-to-{}s'
  ).format(
      format_float_for_filename(alpha),
      format_float_for_filename(true_lag),
      format_float_for_filename(histogram_bin_width),
      format_float_for_filename(coarse_time_lag_step_size),
      format_float_for_filename(fine_time_lag_step_size),
      format_float_for_filename(window_size),
      format_float_for_filename(scan_low),
      format_float_for_filename(scan_high)
  )

def next_available_path(directory, filename):
  path = directory / filename
  if not path.exists():
    return path

  stem = path.stem
  suffix = path.suffix
  run_number = 2
  while True:
    candidate = directory / '{}_run-{:03d}{}'.format(stem, run_number, suffix)
    if not candidate.exists():
      return candidate
    run_number += 1

# LIKELIHOOD PLOT:
def plot_scan_and_find_best_lag(scan_data, true_lag, filename, alpha,
                                histogram_bin_width, window_start, window_size,
                                smoothing, toy):
  # Retrieving the data:
  coarse_lags = np.asarray(scan_data['coarse']['possible_time_lag_list'])
  coarse_log_likelihoods = np.asarray(scan_data['coarse']['log_likelihood_list'])
  fine_lags = np.asarray(scan_data['fine']['possible_time_lag_list'])
  fine_log_likelihoods = np.asarray(scan_data['fine']['log_likelihood_list'])
  total_lags = np.asarray(scan_data['total']['possible_time_lag_list'])
  total_log_likelihoods = np.asarray(scan_data['total']['log_likelihood_list'])

  sigma_1 = SUPERK_RESOL
  sigma_2 = SUPERK_RESOL

  if len(total_lags) < 3:
    raise ValueError('combined scan needs at least three points for the likelihood fit')

  log_likelihood_max = np.nanmax(total_log_likelihoods)
  shifted_coarse_log_likelihoods = coarse_log_likelihoods - log_likelihood_max
  shifted_fine_log_likelihoods = fine_log_likelihoods - log_likelihood_max
  shifted_total_log_likelihoods = total_log_likelihoods - log_likelihood_max

  # Use polynomial to fit the likelihood plot:
  degree_total = min(4, len(total_lags) - 1)
  degree_fine = 2
  fit_curve_total = np.polynomial.Polynomial.fit(total_lags,
                                           shifted_total_log_likelihoods,
                                           degree_total)
  fit_curve_fine = np.polynomial.Polynomial.fit(fine_lags,
                                           shifted_fine_log_likelihoods,
                                           degree_fine)

  number_of_grid_total = 1000
  number_of_grid_fine = 100

  x_fit_total = np.linspace(total_lags.min(), total_lags.max(), number_of_grid_total)
  x_fit_fine = np.linspace(fine_lags.min(), fine_lags.max(), number_of_grid_fine)
  y_fit_total = fit_curve_total(x_fit_total)
  y_fit_fine = fit_curve_fine(x_fit_fine)

  # Determine the best lag directly from the largest raw combined-scan
  # log-likelihood.  The fitted curve is not used to find the best lag.
  if not np.any(np.isfinite(total_log_likelihoods)):
    raise ValueError('combined scan has no finite log-likelihood values')

  best_total_index = int(np.nanargmax(total_log_likelihoods))
  best_lag = float(total_lags[best_total_index])

  # Keep using the fine-scan fit curvature to estimate the standard error,
  # but evaluate that curvature at the raw best lag.
  second_derivative_obs = fit_curve_fine.deriv(2)(best_lag)
  H_obs = float(- second_derivative_obs)
  H_ref = float(- fit_curve_fine.deriv(2)(true_lag))
  standard_error_1 = H_obs**(-0.5) if np.isfinite(H_obs) and H_obs > 0.0 else np.nan

  left_error_bound_1 = best_lag - standard_error_1
  right_error_bound_1 = best_lag + standard_error_1


  # "0.5 Method" on the combined scan.  This allows a crossing to fall
  # outside the fine-scan interval and continue into the coarse scan.
  left_index = best_total_index - 1
  right_index = best_total_index + 1

  while left_index >= 0:
    if shifted_total_log_likelihoods[left_index] > -0.5: 
      left_index -= 1
    else: 
      break 
  
  while right_index < np.size(total_lags):
    if shifted_total_log_likelihoods[right_index] > -0.5:
      right_index += 1 
    else: 
      break 

  if left_index < 0 or right_index >= np.size(total_lags):
    raise ValueError('combined scan does not contain both -0.5 crossings')

  # for coverage, we will use this as the 1-sigma region:
  left_error_bound_2 = total_lags[left_index]
  right_error_bound_2 = total_lags[right_index]

  # for pulls, we will use this (assume Gaussian <==> Symmetrical)
  standard_error_2 = max(right_error_bound_2 - best_lag, best_lag - left_error_bound_2)

  # "Godambe" Information: 
  # J = Var(score), and needs to be found by doing Monte Carlo:
  # score should be computed at the same point for different trials in the Monte Carlo
  score_ref = float(fit_curve_fine.deriv(1)(true_lag))
  if np.isfinite(J) and J > 0.0 and np.isfinite(H) and H > 0.0: 
    godambe = (H**2 / J) * 1000**2 # back to second 
    standard_error_3 = godambe**(-0.5) # back to second
    left_error_bound_3 = best_lag - standard_error_3 
    right_error_bound_3 = best_lag + standard_error_3
  else: 
    godambe = np.nan 
    standard_error_3 = np.nan
    left_error_bound_3 = np.nan
    right_error_bound_3 = np.nan

  print('Plotting Info:')
  print('raw combined-scan best lag: {:.4f} s'.format(best_lag))
  print('standard error from 2nd derivative method: {:.4f} s'.format(standard_error_1))
  print('standard error from 0.5 method: {:.4f} s'.format(standard_error_2))
  print('standard error from Godambe information: {:.4f} s'.format(standard_error_3))
  print('H_obs (- second derivative): {:.4f}'.format(H_obs))
  print('score_ref (slope at true-lag): {:.4f}'.format(score_ref))

  fig = Figure(figsize=(10, 4))
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)
  ax.plot(coarse_lags, shifted_coarse_log_likelihoods,
          marker='o', linestyle='None', markersize=2,
          color='tab:blue', label='coarse scan')
  ax.plot(fine_lags, shifted_fine_log_likelihoods,
          marker='x', linestyle='None', markersize=4,
          color='tab:red', label='fine scan')
  ax.plot(x_fit_total, y_fit_total, color='tab:orange', linestyle='-', linewidth = 1.0,
          label='{}-deg full-scan fit'.format(degree_total))
  ax.plot(x_fit_fine, y_fit_fine, color='tab:green', linestyle='-', linewidth = 1.0,
          label='{}-deg fine-scan fit'.format(degree_fine))
  ax.axvline(true_lag, color='tab:cyan', linestyle='--', linewidth = 2.5, label='true lag')
  ax.axvline(best_lag, color='tab:red', linestyle=':', linewidth = 2.5,
             label='raw combined-scan best lag')

  # shading the 1-sigma region, just for better visualisation:
  # Second-derivative region
  ax.axvspan(
    left_error_bound_1,
    right_error_bound_1,
    facecolor='tab:green',
    edgecolor='tab:green',
    alpha=0.20,
    linewidth=1.2,
    label='Second-derivative method'
)

# 0.5-method region
  ax.axvspan(
      left_error_bound_2,
      right_error_bound_2,
      facecolor='none',
      edgecolor='tab:blue',
      hatch='\\\\',
      linewidth=1.2,
      label='0.5 method'
  )
  if np.isfinite(J) and J > 0.0 and np.isfinite(H) and H > 0.0: 
    ax.axvspan(
      left_error_bound_3,
      right_error_bound_3, 
      facecolor='tab:orange',
      edgecolor='tab:orange',
      alpha=0.20,
      linewidth=1.2,
      label='Godambe Information'
    )
    
  ax.set_xlabel('Time Lag (sec)')
  ax.set_ylabel('ln(L) - ln(L_max)')

  window_stop = window_start + window_size
  title_prefix = (
      'alpha = {}, binwidth={} s, window=[{:.3f}, {:.3f}] s'.format(
          alpha, histogram_bin_width, window_start, window_stop)
          
  )

  if not smoothing:
    ax.set_title(title_prefix + '\nNO SMOOTHING')
    kernel_legend_title = None
  elif toy:
    ax.set_title(title_prefix + '\nkernel-type = Gaussian')
    kernel_legend_title = 'sigma={}'.format(SIGMA_GND)
  else:
    mean_correction_text = ('with mean correction' if MEAN_CORRECTION else 'without mean correction')
    ax.set_title(
        title_prefix + '\nkernel-type = Convolution ({})'.format(
            mean_correction_text))
    kernel_legend_title = ('sigma1={}, sigma2={}\n''rise={}, fall={}\n''frac_rise={}, frac_fall={}').format(sigma_1,
                                                                                                            sigma_2,
                                                                                                            RISE_CONSTANT,
                                                                                                            FALL_CONSTANT,
                                                                                                            FRAC_RISE,
                                                                                                            FRAC_FALL,
                                                                                                           )

  ax.legend(title=kernel_legend_title, title_fontsize=9)
  fig.tight_layout()
  canvas.print_png(filename)

  return best_lag, standard_error_1, standard_error_2, left_error_bound_2, right_error_bound_2, standard_error_3, H_obs, H_ref, score_ref

# use the terminal to change parameters:
# if no extra parameters are provided in the terminal, then we will just use the default parameters that I have typed inside this file.
def add_boolean_argument(parser, name, default, help_text):
  """Add matching --<name> and --no-<name> command-line flags."""
  dest = name.replace('-', '_')
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--' + name, dest=dest, action='store_true',
                     help=help_text)
  group.add_argument('--no-' + name, dest=dest, action='store_false',
                     help='Disable: ' + help_text)
  parser.set_defaults(**{dest: default})

def parse_args():
  parser = argparse.ArgumentParser(
    description='Run a Poisson likelihood scan for one detector pair.'
  )
  parser.add_argument('--superk_yield', type=float, default=SUPERK_YIELD, help='The estimated yield of SuperK in 8.69 seconds')
  parser.add_argument('--superk_bg', type=float, default=SUPERK_BG, help='Background rate of SuperK per second')
  parser.add_argument('--alpha', type=float, default=ALPHA, help='Ratio of det2 yield / superk yield')                    
  parser.add_argument('--true-lag', type=float, default=TRUE_LAG,
                      help='True time lag in seconds.')
  parser.add_argument('--scan-low', type=float, default=SCAN_LOW,
                      help='Lowest lag to scan in seconds.')
  parser.add_argument('--scan-high', type=float, default=SCAN_HIGH,
                      help='Highest lag to scan in seconds.')
  parser.add_argument('--coarse-lag-step', type=float, default=COARSE_TIME_LAG_STEP_SIZE,
                      help='Coarse lag scan step size in seconds.')
  parser.add_argument('--fine-lag-step', type=float, default=FINE_TIME_LAG_STEP_SIZE,
                      help='Fine Lag scan step size in seconds.')
  parser.add_argument('--hist-bin-width', type=float, default=HISTOGRAM_BIN_WIDTH,
                      help='Histogram bin width in seconds.')
  parser.add_argument('--window-size', type=float, default=WINDOW_SIZE,
                      help='Histogram time window size in seconds.')
  parser.add_argument('--window-start', type=float, default=WINDOW_START,
                      help='Histogram time window start in seconds.')
  parser.add_argument('--seed', type=int, default=SEED,
                      help='Random seed for reproducible generated events.')
  parser.add_argument('--model_directory', type=str, default=SUPERK_MODEL_DIREC,
                      help='Model directory that you want to use.')
  smoothing = parser.add_argument_group('smoothing')
  add_boolean_argument(
      smoothing, 'smoothing', SMOOTHING,
      'Apply smoothing when building histogram pairs.'
  )
  add_boolean_argument(
      smoothing, 'hist1-ic', HIST1_IC, 
      'Generate detector 1 directly as a histogram. By default this is '
      'enabled when --det1 is IceCube.'
  )
  add_boolean_argument(
      smoothing, 'toy', TOY,
      'Use the Gaussian toy smoothing kernel.'
  )
  smoothing.add_argument('--sigma-gnd', type=float, default=SIGMA_GND,
                         help='Gaussian toy-kernel standard deviation in seconds.')
  smoothing.add_argument('--rise-constant', type=float, default=RISE_CONSTANT,
                         help='Convolution-kernel rise time constant in seconds.')
  smoothing.add_argument('--fall-constant', type=float, default=FALL_CONSTANT,
                         help='Convolution-kernel fall time constant in seconds.')
  smoothing.add_argument('--frac-rise', type=float, default=FRAC_RISE,
                         help='Convolution-kernel rising-edge fraction.')
  smoothing.add_argument('--frac-fall', type=float, default=FRAC_FALL,
                         help='Convolution-kernel falling-tail fraction.')
  smoothing.add_argument('--impact-range', type=float, default=IMPACT_RANGE,
                         help='Half-width of the smoothing impact range in seconds.')
  add_boolean_argument(
      smoothing, 'mean-correction', MEAN_CORRECTION,
      'Correct the mean of the asymmetric convolution kernel.'
  )
  parser.add_argument('--run-name', default=None,
                      help='Monte Carlo campaign folder name under output/mc_v5.')
  parser.add_argument('--output-dir', type=Path, default=None,
                      help='Directory for output plots.')
  parser.add_argument('--results-dir', type=Path, default=None,
                      help='Directory for one-row Monte Carlo result CSV files.')
  return parser.parse_args()

def apply_smoothing_args(args):
  """Make parsed smoothing settings available to the workflow functions."""
  global HIST1_IC, SMOOTHING, TOY, SIGMA_GND
  global RISE_CONSTANT, FALL_CONSTANT, FRAC_RISE, FRAC_FALL
  global IMPACT_RANGE, MEAN_CORRECTION

  HIST1_IC = args.hist1_ic
  SMOOTHING = args.smoothing
  TOY = args.toy
  SIGMA_GND = args.sigma_gnd
  RISE_CONSTANT = args.rise_constant
  FALL_CONSTANT = args.fall_constant
  FRAC_RISE = args.frac_rise
  FRAC_FALL = args.frac_fall
  IMPACT_RANGE = args.impact_range
  MEAN_CORRECTION = args.mean_correction

def build_result_filename(alpha, seed):
  return 'trial_alpha-{}_seed-{}.csv'.format(alpha, seed)

# Systematic place to save our result:
def save_trial_result(args,
                      best_lag,
                      standard_error_1,
                      standard_error_2,
                      output_filename,
                      left_error,
                      right_error,
                      standard_error_3,
                      H_obs, H_ref, score_ref):

  args.results_dir.mkdir(parents=True, exist_ok=True)
  result_filename = args.results_dir / build_result_filename(args.alpha, args.seed)

  if standard_error_1 > 0.0 and np.isfinite(standard_error_1):
    pull_1 = (best_lag - args.true_lag) / standard_error_1
  else:
    pull_1 = np.nan

  if standard_error_2 > 0.0 and np.isfinite(standard_error_2):
    pull_2 = (best_lag - args.true_lag) / standard_error_2
  else:
    pull_2 = np.nan

  if standard_error_3 > 0.0 and np.isfinite(standard_error_3):
    pull_3 = (best_lag - args.true_lag) / standard_error_3
  else:
    pull_3 = np.nan

  row = {
      'seed': args.seed,
      'alpha': args.alpha,
      'true_lag': args.true_lag,
      'best_lag': best_lag,
      'sigma1': standard_error_1,
      'sigma2': standard_error_2,
      'pull1': pull_1,
      'pull2': pull_2,
      'left_error': left_error,
      'right_error': right_error,
      'sigma3':standard_error_3,
      'pull3': pull_3,
      'H_obs': H_obs,
      'H_ref': H_ref,
      'score_ref': score_ref,
      'hist_bin_width': args.hist_bin_width,
      'coarse_lag_step': args.coarse_lag_step,
      'fine_lag_step': args.fine_lag_step,
      'scan_low': args.scan_low,
      'scan_high': args.scan_high,
      'window_size': args.window_size,
      'hist1_ic': args.hist1_ic,
      'smoothing': args.smoothing,
      'toy': args.toy,
      'sigma_gnd': args.sigma_gnd,
      'rise_constant': args.rise_constant,
      'fall_constant': args.fall_constant,
      'frac_rise': args.frac_rise,
      'frac_fall': args.frac_fall,
      'impact_range': args.impact_range,
      'mean_correction': args.mean_correction,
      'plot_file': str(output_filename),
  }

  with result_filename.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)

  return result_filename, pull_1, pull_2


def main():
  args = parse_args()
  apply_smoothing_args(args)
  run_name = args.run_name
  if run_name is None:
    run_name = build_run_name(
        args.alpha,
        args.true_lag,
        args.coarse_lag_step,
        args.fine_lag_step,
        args.hist_bin_width,
        args.window_size,
        args.scan_low,
        args.scan_high
    )

  # Specify our run_directory in terms of our parameters
  run_dir = MC_OUTPUT_DIRECTORY / format_text_for_filename(run_name)

  model_directory = args.model_directory

  # The sub-folder that stores all the likelihood plots
  if args.output_dir is None:
    args.output_dir = run_dir / 'plots'

  # The sub-folder that stores all the one-row CSV files
  if args.results_dir is None:
    args.results_dir = run_dir / 'results'

  Node.rng = np.random.default_rng(args.seed)

  data = build_timeseries(args.true_lag, args.superk_yield, args.superk_bg, 
                          args.alpha, model_directory,
                          background_window_1=(-1.0,9.0),
                          background_window_2=(-1.0,9.0))

  # Be careful, that now scan_data is a dictionary, saving scanning data of coarse, fine and total scan.
  scan_data = calculations(data, args.superk_yield, args.superk_bg, args.alpha,
                        args.coarse_lag_step,
                        args.fine_lag_step,
                        args.hist_bin_width,
                        args.window_start,
                        args.window_size,
                        args.scan_low,
                        args.scan_high,
                        args.smoothing,
                        args.toy)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  output_filename = args.output_dir / build_plot_filename(args.alpha, args.seed)
                                                  
  best_lag, standard_error_1, standard_error_2, left_error_bound, right_error_bound, standard_error_3, H_obs, H_ref, score_ref = plot_scan_and_find_best_lag(scan_data, 
                                                                                                                                                                 args.true_lag, 
                                                                                                                                                                 output_filename, 
                                                                                                                                                                 args.alpha, 
                                                                                                                                                                 args.hist_bin_width, 
                                                                                                                                                                 args.window_start, 
                                                                                                                                                                 args.window_size, 
                                                                                                                                                                 args.smoothing, 
                                                                                                                                                                 args.toy)
                                                                                                                  


  # Generate a csv file to save stuff!
  result_filename, pull_1, pull_2 = save_trial_result(args, best_lag, standard_error_1, standard_error_2, output_filename, left_error_bound, right_error_bound, standard_error_3, H_obs, H_ref, score_ref)

  print('---------------------------------------')
  print('Summary:')
  print('alpha = {}'.format(args.alpha))
  print('histogram bin width: {:.6f} s'.format(args.hist_bin_width))
  print('coarse time lag step size: {:.6f} s'.format(args.coarse_lag_step))
  print('fine time lag step size: {:.6f} s'.format(args.fine_lag_step))
  print('scan range: {:.6f} s to {:.6f} s'.format(args.scan_low, args.scan_high))
  print('detector 1 generated as histogram: {}'.format(args.hist1_ic))
  if not args.smoothing:
    print('smoothing: disabled')
  elif args.toy:
    print('smoothing kernel: Gaussian (sigma={:.6f} s, impact range={:.6f} s)'.format(
        args.sigma_gnd, args.impact_range))
  else:
    print('smoothing kernel: convolution '
          '(rise={:.6f} s, fall={:.6f} s, frac rise={:.6f}, '
          'frac fall={:.6f}, impact range={:.6f} s, mean correction={})'.format(
              args.rise_constant, args.fall_constant,
              args.frac_rise, args.frac_fall,
              args.impact_range, args.mean_correction))
  print('seed: {}'.format(args.seed))
  print('true lag: {:.6f} s'.format(args.true_lag))
  print('best lag: {:.6f} s'.format(best_lag))
  print('sigma (second derivative): {:.6f} s'.format(standard_error_1))
  print('sigma (0.5 method): {:.6f} s'.format(standard_error_2))
  print('pull (second derivative): {:.6f}'.format(pull_1))
  print('pull (0.5 method): {:.6f}'.format(pull_2))
  print('plot: {}'.format(output_filename))
  print('result csv: {}'.format(result_filename))


if __name__ == '__main__':
  main()
