"""
MAIN FILE of the workflow:
ops/NewTimeSeries.py ==> Empty Time Series
gen/GenTimeDist.py ==> Generate realistic Time Series according to models
gen/GenTimeDist_IceCube.py ==> Generate realistic histogram according to models
TimeLagGenerator ==> Generate a list of possible time lags
PairTimeSeriesToHist_Smoothing ==> Apply smoothing function to the time series / histogram
PoissonLagLikelihood_v7 ==> Applied interpolation to the lnJ table, with c++ plugin
LikelihoodScanCollector ==> Collect all the likelihoods from different histogram pairs, make a summary
"""

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
PoissonLagLikelihood = load_plugin('PoissonLagLikelihood_v7.py').PoissonLagLikelihood_v7
LikelihoodScanCollector = load_plugin('LikelihoodScanCollector.py').LikelihoodScanCollector


# Flexible for mac / cluster
PROJECT_ROOT = Path(__file__).parents[2]

OUTPUT_DIRECTORY = PROJECT_ROOT / 'output'
MC_OUTPUT_DIRECTORY = PROJECT_ROOT / 'output' / 'mc_v7_test'

# DEFAULT PARAMETERS
TRUE_LAG = 0.0
SCAN_LOW = -0.1 # s    # Remove physical wall: -0.042
SCAN_HIGH = 0.1 # s    # Remove physical wall: 0.042
COARSE_TIME_LAG_STEP_SIZE = 0.005 # s
FINE_TIME_LAG_STEP_SIZE = 0.0001 #s
HISTOGRAM_BIN_WIDTH = 0.002 # s # IceCube default histogram bin-width = 2ms
DETECTOR_PAIR = ('IceCube', 'LVD') # (det1, det2)
WINDOW_START = -0.6
WINDOW_SIZE = 9.2 #s
SEED = 1001

SMOOTHING = True
# Generate histogram for IceCube directly --> Speed up the algo A LOT
HIST1_IC = True 
if DETECTOR_PAIR[0] != 'IceCube': 
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

DETECTOR_TOTAL_EVENT_SIGNALS = { # per 7 seconds
  'SuperK': 7800,
  'JUNO': 7200,
  'SNOPLUS': 280,
  'LVD': 360,
  'IceCube': 660000 # Using full version now
}
DETECTOR_BACKGROUND_RATES = { # per 1 second
  'SuperK': 0.1,
  'JUNO': 0.0015,
  'SNOPLUS': 0.001,
  'LVD': 0.03,
  'IceCube': 1476000 # Using full version now
}
MODEL_DIRECTORY = {
  'SuperK': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-wc.data'),
  'JUNO': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-scint.data'),
  'SNOPLUS': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-scint.data'),
  'LVD': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-scint.data'),
  'IceCube': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-wc.data')
}
DETECTOR_RESOLUTION = {
  'SuperK': 1e-6, #/s
  'JUNO': 1e-6, #/s
  'SNOPLUS': 1e-6, #/s
  'LVD': 1e-6, #/s
  'IceCube': 1e-6 #/s
}

# ONLY FOR SAFETY, WHILE RETREIVING THE VALUES:
def detector_yield(detector_name):
  if detector_name not in DETECTOR_TOTAL_EVENT_SIGNALS:
    valid_names = ', '.join(sorted(DETECTOR_TOTAL_EVENT_SIGNALS))
    raise ValueError(
      "Unknown detector '{}'. Choose one of: {}".format(detector_name, valid_names)
    )
  return DETECTOR_TOTAL_EVENT_SIGNALS[detector_name]

def detector_background_rate(detector_name):
  if detector_name not in DETECTOR_BACKGROUND_RATES:
    valid_names = ', '.join(sorted(DETECTOR_BACKGROUND_RATES))
    raise ValueError(
      "Unknown detector '{}'. Choose one of: {}".format(detector_name, valid_names)
    )
  return DETECTOR_BACKGROUND_RATES[detector_name]

def detector_model_directory(detector_name):
  if detector_name not in MODEL_DIRECTORY:
    valid_names = ', '.join(sorted(DETECTOR_BACKGROUND_RATES))
    raise ValueError(
      "Unknown detector '{}'. Choose one of: {}".format(detector_name, valid_names)
    )
  return MODEL_DIRECTORY[detector_name]

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
def build_timeseries(true_lag, detector_pair, model_directory_1, model_directory_2,
                     background_window_1=(-1.0, 9.0),
                     background_window_2=(-1.0, 9.0)):

  det1, det2 = detector_pair
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
                                    detector_background_rate(det1),
                                    bg1_start,
                                    bg1_stop,
                                    name='bg-hist1'), payload_data)
    payload_data = run_node(GenTimeDist_IceCube('hist1',
                                                sig_filename=model_directory_1,
                                                sig_filetype='tn',
                                                sig_delimiter=',',
                                                sig_mean=detector_yield(det1),
                                                sig_once=False,
                                                sig_t0=0.0,
                                                name='model-hist1'),payload_data)
  else: 
    payload_data = run_node(NewTimeSeries('ts1', name='new-ts1'), payload_data)
    payload_data = run_node(Uniform('ts1',
                                    detector_background_rate(det1),
                                    bg1_start,
                                    bg1_stop,
                                    name='bg-ts1'), payload_data)
    payload_data = run_node(GenTimeDist('ts1',
                                sig_filename=model_directory_1,
                                sig_filetype='tn',
                                sig_delimiter=',',
                                sig_mean=detector_yield(det1),
                                sig_once=False,
                                sig_t0=0.0,
                                name='model-ts1'), payload_data)
  
  payload_data = run_node(NewTimeSeries('ts2', name='new-ts2'), payload_data)
  payload_data = run_node(Uniform('ts2',
                                  detector_background_rate(det2),
                                  bg2_start,
                                  bg2_stop,
                                  name='bg-ts2'), payload_data)
  payload_data = run_node(GenTimeDist('ts2',
                              sig_filename=model_directory_2,
                              sig_filetype='tn',
                              sig_delimiter=',',
                              sig_mean=detector_yield(det2),
                              sig_once=False,
                              sig_t0=true_lag,
                              name='model-ts2'), payload_data)
  return payload_data

# THE MOST IMPORTANT FUNCTION!!!
def calculations(payload_data,
                 coarse_time_lag_step_size,
                 fine_time_lag_step_size,
                 histogram_bin_width,
                 detector_pair,
                 window_size,
                 coarse_scan_low,
                 coarse_scan_high,
                 smoothing,
                 toy):

  det1, det2 = detector_pair

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
                                          window_start=WINDOW_START,
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
                                                         window_start=WINDOW_START,
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
                                                     window_start=WINDOW_START,
                                                     window_size=float(window_size),
                                                     lag_field='lag_field',
                                                     rise_constant=RISE_CONSTANT,
                                                     fall_constant=FALL_CONSTANT,
                                                     sigma1=DETECTOR_RESOLUTION[det1],
                                                     sigma2=DETECTOR_RESOLUTION[det2],
                                                     frac_fall=FRAC_FALL,
                                                     frac_rise=FRAC_RISE,
                                                     impact_range=IMPACT_RANGE,
                                                     cache_hist1=True,
                                                     in_hist1_field='hist1',
                                                     hist1_ic=HIST1_IC,
                                                     mean_correction=MEAN_CORRECTION,
                                                     name='pair_smoothed')
  like_cal = PoissonLagLikelihood('hist_pair',
                                  'likelihood_data',
                                  sen_1=detector_yield(det1)/8.0, # unit matters this time, as we are not dealing with ratio anymore.
                                  sen_2=detector_yield(det2)/8.0,
                                  bg_1=detector_background_rate(det1),
                                  bg_2=detector_background_rate(det2),
                                  name='like')
  coarse_collector = LikelihoodScanCollector('likelihood_data',
                                      'scan',
                                      scan_type='coarse',
                                      name='coarse_collector')
  fine_collector = LikelihoodScanCollector('likelihood_data',
                                      'scan',
                                      scan_type='fine',
                                      name='fine_collector')


  # True-running for coarse-scan:
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

  # True-running for fine-scan:
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

def build_plot_filename(detector_pair, true_lag, coarse_time_lag_step_size,
                        fine_time_lag_step_size, histogram_bin_width,
                        window_size, scan_low, scan_high, seed):
  window_size = format_float_for_filename(window_size)
  det1 = format_text_for_filename(detector_pair[0])
  det2 = format_text_for_filename(detector_pair[1])
  return (
    'LogL_against_tau'
    '_DETS-{}-{}'
    '_seed-{}.png'
  ).format(det1, det2, seed)

# Make our csv files collected in a more systematic way
def build_run_name(detector_pair, true_lag, coarse_time_lag_step_size,
                   fine_time_lag_step_size, histogram_bin_width, window_size,
                   scan_low, scan_high):
  det1 = format_text_for_filename(detector_pair[0])
  det2 = format_text_for_filename(detector_pair[1])
  return (
      '{}-{}'
      '_true-lag-{}s'
      '_hist-bin-{}s'
      '_coarse-step-{}s'
      '_fine-step-{}s'
      '_window-{}s'
      '_scan-{}s-to-{}s'
  ).format(
      det1,
      det2,
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
def plot_scan_and_find_best_lag(scan_data, true_lag, filename, detector_pair,
                                histogram_bin_width, window_start, window_size,
                                smoothing, toy):
  # Retrieving the data:
  coarse_lags = np.asarray(scan_data['coarse']['possible_time_lag_list'])
  coarse_log_likelihoods = np.asarray(scan_data['coarse']['log_likelihood_list'])
  fine_lags = np.asarray(scan_data['fine']['possible_time_lag_list'])
  fine_log_likelihoods = np.asarray(scan_data['fine']['log_likelihood_list'])
  total_lags = np.asarray(scan_data['total']['possible_time_lag_list'])
  total_log_likelihoods = np.asarray(scan_data['total']['log_likelihood_list'])
  total_sum_lnJs = np.asarray(scan_data['total']['sum_lnJ_list'])
  total_sum_ln_facs = np.asarray(scan_data['total']['sum_ln_fac_list'])

  sigma_1 = DETECTOR_RESOLUTION[detector_pair[0]]
  sigma_2 = DETECTOR_RESOLUTION[detector_pair[1]]


  if len(total_lags) < 3:
    raise ValueError('combined scan needs at least three points for the likelihood fit')

  # Use one common maximum so the coarse and fine likelihoods are shown on
  # the same vertical scale.
  log_likelihood_max = np.nanmax(total_log_likelihoods)
  sum_lnJ_max = np.nanmax(total_sum_lnJs)
  sum_ln_fac_max = np.nanmax(total_sum_ln_facs)
  shifted_coarse_log_likelihoods = coarse_log_likelihoods - log_likelihood_max
  shifted_fine_log_likelihoods = fine_log_likelihoods - log_likelihood_max
  shifted_total_log_likelihoods = total_log_likelihoods - log_likelihood_max
  shift_total_sum_lnJs = total_sum_lnJs - sum_lnJ_max
  shift_total_sum_ln_facs = total_sum_ln_facs - sum_ln_fac_max


  # Use polynomial to fit the likelihood plot:
  degree_total = min(4, len(total_lags) - 1)
  degree_fine = 2
  fit_curve_total = np.polynomial.Polynomial.fit(total_lags,
                                           shifted_total_log_likelihoods,
                                           degree_total)
  fit_curve_fine = np.polynomial.Polynomial.fit(fine_lags,
                                           shifted_fine_log_likelihoods,
                                           degree_fine)
  if (not np.all(np.isfinite(total_lags))
      or not np.all(np.isfinite(shift_total_sum_ln_facs))):
    raise ValueError('factorial slope fit requires finite lag and ln(n!m!) values')

  # Subtracting a constant maximum changes only the intercept, not the slope
  # or its standard error.
  regression = sc.linregress(total_lags, shift_total_sum_ln_facs)
  slope = regression.slope
  slope_error = regression.stderr

  number_of_grid_total = 1000
  number_of_grid_fine = 100

  x_fit_total = np.linspace(total_lags.min(), total_lags.max(), number_of_grid_total)
  x_fit_fine = np.linspace(fine_lags.min(), fine_lags.max(), number_of_grid_fine)
  y_fit_total = fit_curve_total(x_fit_total)
  y_fit_fine = fit_curve_fine(x_fit_fine)
  y_fit_ln_fac = regression.intercept + slope * x_fit_total

  # Determine the best lag directly from the largest raw fine-scan
  # log-likelihood.  The fitted curve is not used to move the best lag.
  if not np.any(np.isfinite(fine_log_likelihoods)):
    raise ValueError('fine scan has no finite log-likelihood values')

  best_fine_index = int(np.nanargmax(fine_log_likelihoods))
  best_lag = float(fine_lags[best_fine_index])

  # Keep using the fine-scan fit curvature to estimate the standard error,
  # but evaluate that curvature at the raw best lag.
  second_derivative = fit_curve_fine.deriv(2)(best_lag)
  standard_error = (-second_derivative) ** (-0.5) if second_derivative <= 0.0 else np.nan

  left_error_bound = best_lag - standard_error
  right_error_bound = best_lag + standard_error

  print('Plotting Info:')
  print('raw fine-scan best lag: {:.4f} s'.format(best_lag))
  print('standard error from fit: {:.4f} s'.format(standard_error))
  print('slope of ln(n!m!) = {:.4g} ± {:.2g} s^-1'.format(
      slope, slope_error))

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
  ax.plot(total_lags, shift_total_sum_lnJs, marker = 'o', linestyle = 'None', markersize = 2, 
          color = 'black', label = 'sum_lnJ - sum_lnJ_max')
  ax.plot(total_lags, shift_total_sum_ln_facs, marker = 'o', linestyle = 'None', markersize = 2,
          color = 'tab:purple', label = 'sum_ln_fac - sum_ln_fac_max')
  ax.plot(x_fit_total, y_fit_ln_fac, color='tab:purple', linestyle='--',
          linewidth=1.0,
          label='ln(n!m!) slope = {:.4g} ± {:.2g} s$^{{-1}}$'.format(
              slope, slope_error))
  ax.axvline(true_lag, color='tab:cyan', linestyle='--', linewidth = 2.5, label='true lag')
  ax.axvline(best_lag, color='tab:red', linestyle=':', linewidth = 2.5,
             label='raw fine-scan best lag')

  # shading the 1-sigma region, just for better visualisation:
  ax.axvspan(left_error_bound,
             right_error_bound,
             facecolor = 'none',
             edgecolor='tab:green',
             hatch='///',
             linewidth=0.0,
             label='1-sigma region',
             zorder=0)

  ax.set_xlabel('Time Lag (sec)')
  ax.set_ylabel('ln(L) - ln(L_max)')

  window_stop = window_start + window_size
  title_prefix = (
      '{} vs {}, binwidth={} s, window=[{:.3f}, {:.3f}] s'.format(
          detector_pair[0], detector_pair[1], histogram_bin_width,
          window_start, window_stop)
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

  return best_lag, standard_error

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
  parser.add_argument('--det1', default=DETECTOR_PAIR[0],
                      choices=sorted(DETECTOR_TOTAL_EVENT_SIGNALS),
                      help='First detector name.')
  parser.add_argument('--det2', default=DETECTOR_PAIR[1],
                      choices=sorted(DETECTOR_TOTAL_EVENT_SIGNALS),
                      help='Second detector name.')
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
  smoothing = parser.add_argument_group('smoothing')
  add_boolean_argument(
      smoothing, 'smoothing', SMOOTHING,
      'Apply smoothing when building histogram pairs.'
  )
  add_boolean_argument(
      smoothing, 'hist1-ic', None,
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

  # Only IceCube may use the direct-histogram generation path.  Normalize this
  # after parsing so an explicit --hist1-ic cannot enable it for another
  # detector; --no-hist1-ic remains available for IceCube when needed.
  if args.det1 != 'IceCube':
    args.hist1_ic = False
  elif args.hist1_ic is None:
    args.hist1_ic = True

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


def build_result_filename(detector_pair, seed):
  det1 = format_text_for_filename(detector_pair[0])
  det2 = format_text_for_filename(detector_pair[1])
  return 'trial_{}_{}_seed-{}.csv'.format(det1, det2, seed)

# Systematic place to save our result:
def save_trial_result(args,
                      detector_pair,
                      best_lag,
                      standard_deviation,
                      output_filename):

  args.results_dir.mkdir(parents=True, exist_ok=True)
  result_filename = args.results_dir / build_result_filename(detector_pair, args.seed)

  if standard_deviation > 0.0 and np.isfinite(standard_deviation):
    pull = (best_lag - args.true_lag) / standard_deviation
  else:
    pull = np.nan

  row = {
      'seed': args.seed,
      'det1': detector_pair[0],
      'det2': detector_pair[1],
      'true_lag': args.true_lag,
      'best_lag': best_lag,
      'sigma': standard_deviation,
      'pull': pull,
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

  return result_filename, pull


def main():
  args = parse_args()
  apply_smoothing_args(args)
  detector_pair = (args.det1, args.det2)
  run_name = args.run_name
  if run_name is None:
    run_name = build_run_name(
        detector_pair,
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

  model1 = detector_model_directory(args.det1)
  model2 = detector_model_directory(args.det2)

  # The sub-folder that stores all the likelihood plots
  if args.output_dir is None:
    args.output_dir = run_dir / 'plots'

  # The sub-folder that stores all the one-row CSV files
  if args.results_dir is None:
    args.results_dir = run_dir / 'results'

  Node.rng = np.random.default_rng(args.seed)

  data = build_timeseries(args.true_lag, detector_pair, model1, model2,
                          background_window_1=(-1.0,9.0),
                          background_window_2=(-1.0,9.0))

  # Be careful, that now scan_data is a dictionary, saving scanning data of coarse, fine and total scan.
  scan_data = calculations(
    data,
    args.coarse_lag_step,
    args.fine_lag_step,
    args.hist_bin_width,
    detector_pair,
    args.window_size,
    args.scan_low,
    args.scan_high,
    args.smoothing,
    args.toy
  )

  args.output_dir.mkdir(parents=True, exist_ok=True)
  output_filename = args.output_dir / build_plot_filename(
    detector_pair, args.true_lag, args.coarse_lag_step,
    args.fine_lag_step, args.hist_bin_width, args.window_size,
    args.scan_low, args.scan_high, args.seed
  )
  best_lag, standard_deviation = plot_scan_and_find_best_lag(
      scan_data, args.true_lag, output_filename, detector_pair,
      args.hist_bin_width, WINDOW_START, args.window_size,
      args.smoothing, args.toy)


  # Generate a csv file to save stuff!
  result_filename, pull = save_trial_result(args, detector_pair, best_lag, standard_deviation, output_filename)

  print('---------------------------------------')
  print('Summary:')
  print('detectors: {} and {}'.format(detector_pair[0], detector_pair[1]))
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
  print('sigma: {:.6f} s'.format(standard_deviation))
  print('pull: {:.6f}'.format(pull))
  print('plot: {}'.format(output_filename))
  print('result csv: {}'.format(result_filename))


if __name__ == '__main__':
  main()
