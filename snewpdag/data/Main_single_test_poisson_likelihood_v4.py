"""
MAIN FILE of the workflow: 
ops/NewTimeSeries.py ==> Empty Time Series
gen/GenTimeDist.py ==> Generate realistic Time Series according to models 
TimeLagGenerator ==> Generate a list of possible time lags
PairTimeSeriesToHist ==> Shift the time series by a certain time lag + Generate the corresponding histogram pair
PoissonLagLikelihood_v4 ==> New Approach
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
GenTimeDist = load_plugin('gen/GenTimeDist.py').GenTimeDist
TimeLagGenerator = load_plugin('TimeLagGenerator.py').TimeLagGenerator
PairTimeSeriesToHist = load_plugin('PairTimeSeriesToHist.py').PairTimeSeriesToHist
PoissonLagLikelihood = load_plugin('PoissonLagLikelihood_v4.py').PoissonLagLikelihood_v4
LikelihoodScanCollector = load_plugin('LikelihoodScanCollector.py').LikelihoodScanCollector

# Flexible for mac / cluster
PROJECT_ROOT = Path(__file__).parents[2]

MODEL_DIRECTORY_1 = str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-wc.data')
MODEL_DIRECTORY_2 = str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-wc.data')
OUTPUT_DIRECTORY = PROJECT_ROOT / 'output'
MC_OUTPUT_DIRECTORY = PROJECT_ROOT / 'output' / 'mc_v4'

# DEFAULT PARAMETERS
TRUE_LAG = 0.022
SCAN_LOW = -0.042 # s
SCAN_HIGH = 0.042 # s
TIME_LAG_STEP_SIZE = 0.001 # s
HISTOGRAM_BIN_WIDTH = 0.002 # s # IceCube default histogram bin-width = 2ms
DETECTOR_PAIR = ('IceCube', 'SuperK') # (det1, det2)
WINDOW_SIZE = 2.0 #s 
SEED = 12345
DETECTOR_TOTAL_EVENT_SIGNALS = { # per 7 seconds
  'SuperK': 7800, 
  'JUNO': 7200, 
  'SNOPLUS': 280,
  'LVD': 360,
  'IceCube': 660000 # Using full version now
}
DETECTOR_BACKGROUND_RATES = {
  'SuperK': 0.1, 
  'JUNO': 0.0015, 
  'SNOPLUS': 0.3,
  'LVD': 0.03,
  'IceCube': 1476000 # Using full version now 
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


# SETUP THE PAYLOAD:
# BUILD 2 TIMESERIES, WHICH WILL BE FED INTO OUR TRANSFORMER LATER:
# WITHOUT BACKGROUND NOISE HERE
def build_timeseries(true_lag, detector_pair, model_directory_1, model_directory_2):
  GenTimeDist.one_series = ()
  GenTimeDist.one_mean = 0

  det1, det2 = detector_pair
  payload_data = {'action': 'alert'}
  payload_data = run_node(NewTimeSeries('ts1', name='new-ts1'), payload_data)
  payload_data = run_node(GenTimeDist('ts1',
                              sig_filename=model_directory_1,
                              sig_filetype='tn',
                              sig_delimiter=',',
                              sig_mean=detector_yield(det1),
                              sig_once=False,
                              sig_t0=0.0,
                              name='model-ts1'), payload_data)
  payload_data = run_node(NewTimeSeries('ts2', name='new-ts2'), payload_data)
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
                 time_lag_step_size, 
                 histogram_bin_width,
                 detector_pair, 
                 window_size,
                 scan_low,
                 scan_high):
  
  det1, det2 = detector_pair
  payload_data = run_node(TimeLagGenerator('possible_time_lag_list',
                                  scan_low=scan_low,
                                  scan_high=scan_high,
                                  step_size=time_lag_step_size,
                                  name='possible_time_lag_list'), 
                                  payload_data)

  # We add the background noise directly into the histograms 
  # This should be easier than generating the time series with background nosie, as GenTimeDist is based on model
  histogram_pair = PairTimeSeriesToHist('ts1', 
                                        'ts2', 
                                        'hist_pair', 
                                        hist_bin_width=histogram_bin_width,
                                        background_rate_1=detector_background_rate(det1),
                                        background_rate_2=detector_background_rate(det2),
                                        start=0.0, 
                                        window=float(window_size),
                                        lag_field='lag_field', 
                                        name='pair')
  
  likelihood_calculator = PoissonLagLikelihood('hist_pair', 
                                               'likelihood_data', 
                                               sensitivity_1=detector_yield(det1)/7.0, # unit matters this time, as we are not dealing with ratio anymore. 
                                               sensitivity_2=detector_yield(det2)/7.0,
                                               background_rate_1=detector_background_rate(det1),
                                               background_rate_2=detector_background_rate(det2),
                                               name='like')
  
  collector = LikelihoodScanCollector('likelihood_data', 
                                      'scan', 
                                      scan_type='coarse', 
                                      name='collector')

  for lag in payload_data['possible_time_lag_list']:
    trial_payload_data = payload_data.copy()
    trial_payload_data['lag_field'] = float(lag)
    histogram_pair.update(trial_payload_data)
    likelihood_calculator.update(histogram_pair.last_data)
    collector.update(likelihood_calculator.last_data)

  return collector.last_data['scan'] # we only want the output by collector! 



# FILE_NAME_MODIFICATION: 
def format_float_for_filename(value):
  return '{:.6g}'.format(value).replace('-', 'm').replace('.', 'p')

def format_text_for_filename(value):
  text = str(value).strip()
  text = re.sub(r'[^A-Za-z0-9._-]+', '-', text)
  return text.strip('-') or 'unknown'

def build_plot_filename(detector_pair, true_lag, time_lag_step_size, histogram_bin_width, window_size, scan_low, scan_high):
  hist_bin_width = format_float_for_filename(histogram_bin_width)
  lag_step = format_float_for_filename(time_lag_step_size)
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
    '_lag-step-{}s.png'
  ).format(det1, det2, hist_bin_width, true_lag_text, window_size, scan_low_text, scan_high_text, lag_step)


# Make our csv files collected in a more systematic way
def build_run_name(detector_pair, true_lag, time_lag_step_size, histogram_bin_width, window_size, scan_low, scan_high):
  det1 = format_text_for_filename(detector_pair[0])
  det2 = format_text_for_filename(detector_pair[1])
  return (
      '{}-{}'
      '_true-lag-{}s'
      '_hist-bin-{}s'
      '_lag-step-{}s'
      '_window-{}s'
      '_scan-{}s-to-{}s'
  ).format(
      det1,
      det2,
      format_float_for_filename(true_lag),
      format_float_for_filename(histogram_bin_width),
      format_float_for_filename(time_lag_step_size),
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
  possible_time_lag_list = np.asarray(scan_data['possible_time_lag_list'])
  log_likelihood_list = np.asarray(scan_data['log_likelihood_list'])
  shifted_log_likelihood_list = log_likelihood_list - np.max(log_likelihood_list)


  # Use a 10-degree polynomial to fit the data points:
  degree = 10 
  fit_curve = np.polynomial.Polynomial.fit(possible_time_lag_list, 
                                           shifted_log_likelihood_list, 
                                           degree)

  number_of_grid = 1000
  x_fit = np.linspace(possible_time_lag_list.min(), possible_time_lag_list.max(), number_of_grid)
  y_fit = fit_curve(x_fit)

  # Compute the standard error by the second-derivative of the ln(L) with respect to tau:
  best_lag_fit = x_fit[np.argmax(y_fit)]
  second_derivative = fit_curve.deriv(2)(best_lag_fit)
  standard_error = (-second_derivative) ** (-0.5) if second_derivative < 0.0 else np.nan

  left_error_bound = best_lag_fit - standard_error
  right_error_bound = best_lag_fit + standard_error

  print('Plotting Info:')
  print('fit best lag: {:.6f} s'.format(best_lag_fit))
  print('standard error from fit: {:.6f} s'.format(standard_error))

  fig = Figure(figsize=(10, 4))
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)
  ax.plot(possible_time_lag_list, shifted_log_likelihood_list, marker='x',linestyle='None')
  ax.plot(x_fit, y_fit, color='tab:orange', linestyle='-', label='10-deg best fit polynomial')
  ax.axvline(true_lag, color='tab:blue', linestyle='--', label='true lag')
  ax.axvline(best_lag_fit, color='tab:red', linestyle=':', label='best lag')

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

  return best_lag_fit, standard_error

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
  parser.add_argument('--lag-step', type=float, default=TIME_LAG_STEP_SIZE,
                      help='Lag scan step size in seconds.')
  parser.add_argument('--hist-bin-width', type=float, default=HISTOGRAM_BIN_WIDTH,
                      help='Histogram bin width in seconds.')
  parser.add_argument('--window-size', type=float, default=WINDOW_SIZE,
                      help='Histogram time window size in seconds.')
  parser.add_argument('--seed', type=int, default=SEED,
                      help='Random seed for reproducible generated events.')
  parser.add_argument('--model1', default=MODEL_DIRECTORY_1,
                      help='Signal model file for detector 1.')
  parser.add_argument('--model2', default=MODEL_DIRECTORY_2,
                      help='Signal model file for detector 2.')
  parser.add_argument('--run-name', default=None,
                      help='Monte Carlo campaign folder name under output/mc_v4.')
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
      'lag_step': args.lag_step,
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
        args.lag_step,
        args.hist_bin_width,
        args.window_size,
        args.scan_low,
        args.scan_high
    )
  
  # Specify our run_directory in terms of our parameters
  run_dir = MC_OUTPUT_DIRECTORY / format_text_for_filename(run_name)


  # The sub-folder that stores all the likelihood plots
  if args.output_dir is None:
    args.output_dir = run_dir / 'plots'

  # The sub-folder that stores all the one-row CSV files 
  if args.results_dir is None:
    args.results_dir = run_dir / 'results'

  Node.rng = np.random.default_rng(args.seed)
  data = build_timeseries(args.true_lag, detector_pair, args.model1, args.model2)
  scan_data = calculations(
    data,
    args.lag_step,
    args.hist_bin_width,
    detector_pair,
    args.window_size,
    args.scan_low,
    args.scan_high
  )

  args.output_dir.mkdir(parents=True, exist_ok=True)
  output_filename = next_available_path(
    args.output_dir,
    build_plot_filename(detector_pair, args.true_lag, args.lag_step, args.hist_bin_width, args.window_size, args.scan_low, args.scan_high)
  )
  best_lag, standard_deviation = plot_scan_and_find_best_lag(scan_data, args.true_lag, output_filename, detector_pair, args.hist_bin_width, args.window_size)


  # Generate a csv file to save stuff!
  result_filename, pull = save_trial_result(args, detector_pair, best_lag, standard_deviation, output_filename)

  print('---------------------------------------')
  print('Summary:')
  print('detectors: {} and {}'.format(detector_pair[0], detector_pair[1]))
  print('histogram bin width: {:.6f} s'.format(args.hist_bin_width))
  print('time lag step size: {:.6f} s'.format(args.lag_step))
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
