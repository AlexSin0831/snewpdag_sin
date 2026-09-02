"""
This is the Main file for implementing the Poisson likelihood lightcurve matching algorithm locally,
without using a csv configuration file. 

MAIN FILE of the workflow:
ops/NewTimeSeries.py ==> Empty Time Series
ops/NewHist1D.py ==> Empty Histogram
gen/GenTimeDist.py ==> Generate realistic Time Series according to models
gen/GenTimeDist_IceCube.py ==> Generate realistic histogram according to models
TimeLagGenerator ==> Generate a list of possible time lags
PairTimeSeriesToHist ==> Shift the time series by a certain time lag + Generate the corresponding histogram pair
PoissonLagLikelihood_v5 ==> New integration by part approach ; Use c++ to calculate the lnJ table
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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parents[2]))

from snewpdag.dag import Node

import importlib


def load_plugin(filename):
  module_name = 'snewpdag.plugins.' + filename[:-3].replace('/', '.')
  return importlib.import_module(module_name)

def run_node(node, data):
  node.update(data)
  return node.last_data


NewTimeSeries = load_plugin('ops/NewTimeSeries.py').NewTimeSeries
NewHist1D = load_plugin('ops/NewHist1D.py').NewHist1D
Uniform = load_plugin('gen/Uniform.py').Uniform
GenTimeDist = load_plugin('gen/GenTimeDist.py').GenTimeDist
GenTimeDist_IceCube = load_plugin('gen/GenTimeDist_IceCube.py').GenTimeDist_IceCube
TimeLagGenerator = load_plugin('TimeLagGenerator.py').TimeLagGenerator
PairTimeSeriesToHist = load_plugin('PairTimeSeriesToHist.py').PairTimeSeriesToHist
PoissonLagLikelihood = load_plugin('PoissonLagLikelihood_v5.py').PoissonLagLikelihood_v5
LikelihoodScanCollector = load_plugin('LikelihoodScanCollector.py').LikelihoodScanCollector


# Flexible for mac / cluster
PROJECT_ROOT = Path(__file__).parents[2]

OUTPUT_DIRECTORY = PROJECT_ROOT / 'output'
MC_OUTPUT_DIRECTORY = PROJECT_ROOT / 'output' / 'mc_v5_test'

# DEFAULT PARAMETERS
TRUE_LAG = 0.022
SCAN_LOW = -0.1 # s    # Remove physical wall: -0.042
SCAN_HIGH = 0.1 # s    # Remove physical wall: 0.042
COARSE_TIME_LAG_STEP_SIZE = 0.005 # s
FINE_TIME_LAG_STEP_SIZE = 0.0001 #s

HISTOGRAM_BIN_WIDTH = 0.002 # s # IceCube default histogram bin-width = 2ms
DETECTOR_PAIR = ('SuperK', 'SNOPLUS') # (det1, det2)
WINDOW_START = -0.6
WINDOW_SIZE = 9.2 #s
SEED = 1000

# Generate histogram for IceCube directly --> Speed up the algo A LOT
# Set this to False if we want to test IceCube with a TimeSeries instead.
HIST1_IC = True


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


def build_total_likelihood_table(coarse_scan, fine_scan):
  """Combine coarse scan and fine scan, prepare the data for plotting the likelihood

  Coarse points inside the fine-scan interval are replaced by the denser fine
  points.  This avoids duplicate lag values and keeps every lag paired with
  the correct log-likelihood.
  """
  coarse_lags = np.asarray(coarse_scan['possible_time_lag_list'], dtype=float)
  coarse_log_likelihoods = np.asarray(coarse_scan['log_likelihood_list'], dtype=float)
  fine_lags = np.asarray(fine_scan['possible_time_lag_list'], dtype=float)
  fine_log_likelihoods = np.asarray(fine_scan['log_likelihood_list'], dtype=float)

  # Keep coarse points only outside the interval covered by the fine scan.
  coarse_outside_roi = ((coarse_lags < np.min(fine_lags)) | (coarse_lags > np.max(fine_lags)))

  total_lags = np.concatenate((coarse_lags[coarse_outside_roi], fine_lags))
  total_log_likelihoods = np.concatenate((coarse_log_likelihoods[coarse_outside_roi], fine_log_likelihoods))

  # Sort both arrays with the same indices so each (lag, likelihood) pair stays
  # together.  The table columns are: [lag, log_likelihood].
  sort_order = np.argsort(total_lags)
  total_lags = total_lags[sort_order]
  total_log_likelihoods = total_log_likelihoods[sort_order]
  total_table = np.column_stack((total_lags, total_log_likelihoods))

  return {
      'possible_time_lag_list': total_lags,
      'log_likelihood_list': total_log_likelihoods,
      'table': total_table,
      'scan_type': 'combined',
  }


def detector_background_windows(window_start, window_size, scan_low, scan_high):
  """Return raw detector time ranges needed for all trial-lag histograms."""
  if not np.isfinite(window_size) or window_size <= 0.0:
    raise ValueError('Histogram time window size must be positive and finite')
  window_stop = window_start + window_size
  return (
      (window_start, window_stop),
      (window_start + scan_low, window_stop + scan_high),
  )


# build an empty time series / histogram --> adding background noise --> adding model lightcurve:
def build_timeseries(true_lag, detector_pair, model_directory_1, model_directory_2,
                     background_window_1=(-3.0, 9.0),
                     background_window_2=(-3.0, 9.0),
                     histogram_bin_width=HISTOGRAM_BIN_WIDTH):
  GenTimeDist.one_series = ()
  GenTimeDist.one_mean = 0

  det1, det2 = detector_pair
  hist1_ic = HIST1_IC and det1 == 'IceCube'
  bg1_start, bg1_stop = background_window_1
  bg2_start, bg2_stop = background_window_2
  payload_data = {'action': 'alert'}

  if hist1_ic == True:
    # IceCube's hist:
    start_empty = min(-3.0, bg1_start)
    stop_empty = max(3.0, bg1_stop)
    nbins_empty = int(np.ceil((stop_empty - start_empty) / histogram_bin_width))
    stop_empty = start_empty + nbins_empty * histogram_bin_width
    payload_data = run_node(NewHist1D('hist1',
                                    nbins_empty,
                                    start_empty,
                                    stop_empty,
                                    name='new-hist1'), payload_data)
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
                                                name='model-hist1'), payload_data)
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
  
  # Do the 2 time series have the same relative separation for time stamps? 
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
                 coarse_scan_high):

  # Error-checking:
  if coarse_time_lag_step_size <= 0.0:
    raise ValueError('coarse lag step must be positive')
  if fine_time_lag_step_size <= 0.0:
    raise ValueError('fine lag step must be positive')
  if fine_time_lag_step_size >= coarse_time_lag_step_size:
    raise ValueError('fine lag step must be smaller than coarse lag step')

  det1, det2 = detector_pair
  hist1_ic = HIST1_IC and det1 == 'IceCube'

  # The first time we run TimeLagGenerator:
  # COARSE-SCAN:
  payload_data = run_node(TimeLagGenerator('possible_time_lag_list',
                                  scan_low=coarse_scan_low,
                                  scan_high=coarse_scan_high,
                                  step_size=coarse_time_lag_step_size,
                                  name='possible_time_lag_list'),
                                  payload_data)

  # Background has already been generated in the detector time series / histogram.
  histogram_pair = PairTimeSeriesToHist('ts1',
                                        'ts2',
                                        'hist_pair',
                                        hist_bin_width=histogram_bin_width,
                                        window_start=WINDOW_START,
                                        window=float(window_size),
                                        lag_field='lag_field',
                                        hist1_ic=hist1_ic,
                                        in_hist1_field='hist1',
                                        name='pair')

  # Calculate the log likelihood corresponding to each histogram pair:
  likelihood_calculator = PoissonLagLikelihood('hist_pair',
                                               'likelihood_data',
                                               sen_1=detector_yield(det1)/7.0, # unit matters this time, as we are not dealing with ratio anymore.
                                               sen_2=detector_yield(det2)/7.0,
                                               bg_1=detector_background_rate(det1),
                                               bg_2=detector_background_rate(det2),
                                               name='like')

  # Collect the payload data from the coarse scan:
  coarse_collector = LikelihoodScanCollector('likelihood_data',
                                      'scan',
                                      scan_type='coarse',
                                      name='coarse_collector')

  # Collect the payload data from the fine scan
  fine_collector = LikelihoodScanCollector('likelihood_data',
                                      'scan',
                                      scan_type='fine',
                                      name='fine_collector')


  # Looping the time lags for coarse scan:
  for lag in payload_data['possible_time_lag_list']:
    trial_payload_data = payload_data.copy()
    trial_payload_data['lag_field'] = float(lag)
    histogram_pair.update(trial_payload_data)
    likelihood_calculator.update(histogram_pair.last_data)
    coarse_collector.update(likelihood_calculator.last_data)

  # The coarse collector stores its result at 'scan' and puts the fine-scan region at 'scan/roi'.
  # Feed that payload into a second lag generator so it can read the ROI and replace the coarse grid with a finer one.
  # The second time we run TimeLagGenerator:
  # FINE-SCAN:
  fine_payload_data = run_node(TimeLagGenerator('possible_time_lag_list',
                                                scan_type='fine',
                                                fine_step=fine_time_lag_step_size,
                                                in_roi_field='scan/roi',
                                                name='fine_time_lag_list'),
                                                coarse_collector.last_data)

  # Looping the time lags for fine scan:
  for lag in fine_payload_data['possible_time_lag_list']:
    trial_payload_data = fine_payload_data.copy()
    trial_payload_data['lag_field'] = float(lag)
    histogram_pair.update(trial_payload_data)
    likelihood_calculator.update(histogram_pair.last_data)
    fine_collector.update(likelihood_calculator.last_data)

  coarse_scan_data = coarse_collector.last_data['scan']
  fine_scan_data = fine_collector.last_data['scan']


  total_scan = build_total_likelihood_table(coarse_scan_data, fine_scan_data)

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
  hist_bin_width = format_float_for_filename(histogram_bin_width)
  coarse_lag_step = format_float_for_filename(coarse_time_lag_step_size)
  fine_lag_step = format_float_for_filename(fine_time_lag_step_size)
  true_lag_text = format_float_for_filename(true_lag)
  window_size = format_float_for_filename(window_size)
  scan_low_text = format_float_for_filename(scan_low)
  scan_high_text = format_float_for_filename(scan_high)
  det1 = format_text_for_filename(detector_pair[0])
  det2 = format_text_for_filename(detector_pair[1])
  return (
    'LogL_against_tau'
    '_DETS-{}-{}'
    '_hist-bin-width-{}s'
    '_true-lag-{}s'
    '_window-size-{}s'
    '_scan-{}s-to-{}s'
    '_coarse-step-{}s'
    '_fine-step-{}s'
    '_seed-{}.png'
  ).format(det1, det2, hist_bin_width, true_lag_text, window_size,
           scan_low_text, scan_high_text, coarse_lag_step, fine_lag_step, seed)


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
def plot_scan_and_find_best_lag(scan_data, true_lag, filename, detector_pair, histogram_bin_width, window_size):
  # Retrieving the data:
  coarse_lags = np.asarray(scan_data['coarse']['possible_time_lag_list'])
  coarse_log_likelihoods = np.asarray(scan_data['coarse']['log_likelihood_list'])
  fine_lags = np.asarray(scan_data['fine']['possible_time_lag_list'])
  fine_log_likelihoods = np.asarray(scan_data['fine']['log_likelihood_list'])
  total_lags = np.asarray(scan_data['total']['possible_time_lag_list'])
  total_log_likelihoods = np.asarray(scan_data['total']['log_likelihood_list'])



  if len(total_lags) < 3:
    raise ValueError('combined scan needs at least three points for the likelihood fit')

  # Use one common maximum so the coarse and fine likelihoods are shown on
  # the same vertical scale.
  log_likelihood_max = np.nanmax(total_log_likelihoods)
  shifted_coarse_log_likelihoods = coarse_log_likelihoods - log_likelihood_max
  shifted_fine_log_likelihoods = fine_log_likelihoods - log_likelihood_max
  shifted_total_log_likelihoods = total_log_likelihoods - log_likelihood_max


  # Use polynomial to fit the likelihood plot:
  degree_total = min(10, len(total_lags) - 1)
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
  print('raw fine-scan best lag: {:.6f} s'.format(best_lag))
  print('standard error from fit: {:.6f} s'.format(standard_error))

  fig = Figure(figsize=(10, 4))
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)
  ax.plot(coarse_lags, shifted_coarse_log_likelihoods,
          marker='x', linestyle='None', linewidth=0.8,
          color='tab:blue', label='coarse scan')
  ax.plot(fine_lags, shifted_fine_log_likelihoods,
          marker='o', markersize=3, linestyle='None',
          color='tab:red', label='fine scan')
  ax.plot(x_fit_total, y_fit_total, color='tab:orange', linestyle=':', linewidth = 2,
          label='{}-deg full-scan fit'.format(degree_total))
  ax.plot(x_fit_fine, y_fit_fine, color='tab:green', linestyle='-', linewidth = 2.5,
          label='{}-deg fine-scan fit'.format(degree_fine))
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
  ax.set_title('Normalised ln(L) against time_lag, {} vs {}, binwidth={}, window-size={}'.format(detector_pair[0],
                                                                                                 detector_pair[1],
                                                                                                 histogram_bin_width,
                                                                                                 window_size))
  ax.legend()
  fig.tight_layout()
  canvas.print_png(filename)

  return best_lag, standard_error

# use the terminal to change parameters:
# if no extra parameters are provided in the terminal, then we will just use the default parameters that I have typed inside this file.
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
  parser.add_argument('--seed', type=int, default=SEED,
                      help='Random seed for reproducible generated events.')
  parser.add_argument('--run-name', default=None,
                      help='Monte Carlo campaign folder name under output/mc_v5.')
  parser.add_argument('--output-dir', type=Path, default=None,
                      help='Directory for output plots.')
  parser.add_argument('--results-dir', type=Path, default=None,
                      help='Directory for one-row Monte Carlo result CSV files.')
  return parser.parse_args()


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
      'plot_file': str(output_filename),
  }

  with result_filename.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)

  return result_filename, pull


def main():
  args = parse_args()
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
  bg_window_1, bg_window_2 = detector_background_windows(
      WINDOW_START, args.window_size, args.scan_low, args.scan_high)
  data = build_timeseries(args.true_lag, detector_pair, model1, model2,
                          background_window_1=bg_window_1,
                          background_window_2=bg_window_2,
                          histogram_bin_width=args.hist_bin_width)

  # Be careful, that now scan_data is a dictionary, saving scanning data of coarse, fine and total scan.
  scan_data = calculations(
    data,
    args.coarse_lag_step,
    args.fine_lag_step,
    args.hist_bin_width,
    detector_pair,
    args.window_size,
    args.scan_low,
    args.scan_high
  )

  args.output_dir.mkdir(parents=True, exist_ok=True)
  output_filename = args.output_dir / build_plot_filename(
    detector_pair, args.true_lag, args.coarse_lag_step,
    args.fine_lag_step, args.hist_bin_width, args.window_size,
    args.scan_low, args.scan_high, args.seed
  )
  best_lag, standard_deviation = plot_scan_and_find_best_lag(scan_data, args.true_lag, output_filename, detector_pair, args.hist_bin_width, args.window_size)


  # Generate a csv file to save stuff!
  result_filename, pull = save_trial_result(args, detector_pair, best_lag, standard_deviation, output_filename)

  print('---------------------------------------')
  print('Summary:')
  print('detectors: {} and {}'.format(detector_pair[0], detector_pair[1]))
  print('histogram bin width: {:.6f} s'.format(args.hist_bin_width))
  print('coarse time lag step size: {:.6f} s'.format(args.coarse_lag_step))
  print('fine time lag step size: {:.6f} s'.format(args.fine_lag_step))
  print('scan range: {:.6f} s to {:.6f} s'.format(args.scan_low, args.scan_high))
  print('seed: {}'.format(args.seed))
  print('true lag: {:.6f} s'.format(args.true_lag))
  print('best lag: {:.6f} s'.format(best_lag))
  print('sigma: {:.6f} s'.format(standard_deviation))
  print('pull: {:.6f}'.format(pull))
  print('plot: {}'.format(output_filename))
  print('result csv: {}'.format(result_filename))


if __name__ == '__main__':
  main()
