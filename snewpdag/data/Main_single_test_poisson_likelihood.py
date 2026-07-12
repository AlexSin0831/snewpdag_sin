"""
MAIN FILE of the workflow: 
ops/NewTimeSeries.py ==> Empty Time Series
gen/GenTimeDist.py ==> Generate realistic Time Series according to models 
TimeLagGenerator ==> Generate a list of possible time lags
PairTimeSeriesToHist ==> Shift the time series by a certain time lag + Generate the corresponding histogram pair
PoissonLagLikelihood ==> Calculate the likelihood for a certain histogram pair 
LikelihoodScanCollector ==> Collect all the likelihoods from different histogram pairs, make a summary 
"""
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
PoissonLagLikelihood = load_plugin('PoissonLagLikelihood_v1.py').PoissonLagLikelihood_v1
LikelihoodScanCollector = load_plugin('LikelihoodScanCollector.py').LikelihoodScanCollector

MODEL_DIRECTORY_1 = '/Users/alexsin/Desktop/SURE/snewpdag/models/ibd-s27-nmo-wc.data'
MODEL_DIRECTORY_2 = '/Users/alexsin/Desktop/SURE/snewpdag/models/ibd-s27-nmo-scint.data'
OUTPUT_DIRECTORY = Path('output')

# PARAMETERS
TRUE_LAG = 0.022
TIME_LAG_STEP_SIZE = 0.001 # s
HISTOGRAM_BIN_WIDTH = 0.01 # s
DETECTOR_PAIR = ('SuperK', 'LVD') # (det1, det2)
WINDOW_SIZE = 2.0 #s 
DETECTOR_TOTAL_EVENT_SIGNALS = { # per 7 seconds
  'SuperK': 7800, 
  'JUNO': 7200, 
  'SNOPLUS': 280,
  'LVD': 360,
  'IceCube': 660000
}
DETECTOR_BACKGROUND_RATES = {
  'SuperK': 0.1, 
  'JUNO': 0.0015, 
  'SNOPLUS': 0.3,
  'LVD': 0.03,
  'IceCube': 1476000 #per second
}

# ONLY FOR SAFETY, WHILE RETREIVING THE VALUES:
def detector_yield(detector_name):
  if detector_name not in DETECTOR_TOTAL_EVENT_SIGNALS:
    valid_names = ', '.join(sorted(DETECTOR_TOTAL_EVENT_SIGNALS))
    raise ValueError(
      "Unknown detector '{}'. Choose one of: {}".format(detector_name, valid_names)
    )
  return DETECTOR_TOTAL_EVENT_SIGNALS[detector_name]

def detector_background(detector_name):
  if detector_name not in DETECTOR_BACKGROUND_RATES:
    valid_names = ', '.join(sorted(DETECTOR_BACKGROUND_RATES))
    raise ValueError(
      "Unknown detector '{}'. Choose one of: {}".format(detector_name, valid_names)
    )
  return DETECTOR_BACKGROUND_RATES[detector_name]


# SETUP THE PAYLOAD:
# BUILD 2 TIMESERIES, WHICH WILL BE FED INTO OUR TRANSFORMER LATER:
def build_timeseries(true_lag, detector_pair):
  GenTimeDist.one_series = ()
  GenTimeDist.one_mean = 0

  det1, det2 = detector_pair
  payload_data = {'action': 'alert'}
  payload_data = run_node(NewTimeSeries('ts1', name='new-ts1'), payload_data)
  payload_data = run_node(GenTimeDist('ts1',
                              sig_filename=MODEL_DIRECTORY_1,
                              sig_filetype='tn',
                              sig_delimiter=',',
                              sig_mean=detector_yield(det1),
                              sig_once=False,
                              sig_t0=0.0,
                              name='model-ts1'), payload_data)
  payload_data = run_node(NewTimeSeries('ts2', name='new-ts2'), payload_data)
  payload_data = run_node(GenTimeDist('ts2',
                              sig_filename=MODEL_DIRECTORY_2,
                              sig_filetype='tn',
                              sig_delimiter=',',
                              sig_mean=detector_yield(det2),
                              sig_once=False,
                              sig_t0=true_lag,
                              name='model-ts2'), payload_data)
  return payload_data


def calculations(payload_data, 
                 time_lag_step_size, 
                 histogram_bin_width,
                 detector_pair, 
                 window_size):
  det1, det2 = detector_pair
  payload_data = run_node(TimeLagGenerator('possible_time_lag_list',
                                  scan_low=-0.042, # -42 ms
                                  scan_high=0.042, # 42 ms
                                  step_size=time_lag_step_size,
                                  name='possible-time-lag-list'), 
                                  payload_data)


  histogram_pair = PairTimeSeriesToHist('ts1', 
                                        'ts2', 
                                        'hist_pair', 
                                        bin_width=histogram_bin_width,
                                        start=0.0, 
                                        window=float(window_size),
                                        lag_field='lag_field', 
                                        name='pair')
  
  likelihood_calculator = PoissonLagLikelihood('hist_pair', 
                                               'likelihood_data', 
                                               sensitivity_1=detector_yield(det1),
                                               sensitivity_2=detector_yield(det2),
                                               background_1 =detector_background(det1),
                                               background_2 =detector_background(det2),
                                               background_is_rate=True,
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

  return collector.last_data['scan'] # we only want the 'scan' part of the whole payload



# FILE_NAME_MODIFICATION: 
def format_float_for_filename(value):
  return '{:.6g}'.format(value).replace('-', 'm').replace('.', 'p')

def format_text_for_filename(value):
  text = str(value).strip()
  text = re.sub(r'[^A-Za-z0-9._-]+', '-', text)
  return text.strip('-') or 'unknown'

def build_plot_filename(detector_pair, true_lag, time_lag_step_size, histogram_bin_width, window_size):
  hist_bin_width = format_float_for_filename(histogram_bin_width)
  lag_step = format_float_for_filename(time_lag_step_size)
  true_lag_text = format_float_for_filename(true_lag)
  window_size = format_float_for_filename(window_size)
  det1 = format_text_for_filename(detector_pair[0])
  det2 = format_text_for_filename(detector_pair[1])
  return (
    'LogL_against_tau'
    '_DETS-{}-{}'
    '_hist-bin-width-{}s'
    '_true-lag-{}s'
    '_window-size-{}s'
    '_lag-step-{}s.png'
  ).format(det1, det2, hist_bin_width, true_lag_text, window_size, lag_step)

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
def plot_scan(scan_data, true_lag, filename):
  possible_time_lag_list = np.asarray(scan_data['possible_time_lag_list'])
  log_likelihood_list = np.asarray(scan_data['log_likelihood_list'])
  shifted_log_likelihood_list = log_likelihood_list - np.max(log_likelihood_list)


  # Use a 10-degree polynomial to fit the data points:
  degree = 10 
  fit_curve = np.polynomial.Polynomial.fit(possible_time_lag_list, 
                                           shifted_log_likelihood_list, 
                                           degree)

  x_fit = np.linspace(possible_time_lag_list.min(), possible_time_lag_list.max(), 500)
  y_fit = fit_curve(x_fit)

  # Compute the standard error by the second-derivative of the ln(L) with respect to tau:
  best_lag_fit = x_fit[np.argmax(y_fit)]
  second_derivative = fit_curve.deriv(2)(best_lag_fit)
  standard_error = (-second_derivative) ** (-0.5)

  left_error_bound = scan_data['best_lag'] - standard_error
  right_error_bound = scan_data['best_lag'] + standard_error

  print('fit best lag: {:.6f} s'.format(best_lag_fit))
  print('standard error from fit: {:.6f} s'.format(standard_error))

  fig = Figure(figsize=(8, 4))
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)
  ax.plot(possible_time_lag_list, shifted_log_likelihood_list, marker='x',linestyle='None')
  ax.plot(x_fit, y_fit, color='tab:orange', linestyle='-', label='10-deg best fit polynomial')
  ax.axvline(true_lag, color='tab:blue', linestyle='--', label='true lag')
  ax.axvline(scan_data['best_lag'], color='tab:red', linestyle=':', label='best lag')

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
  ax.set_title('Normalised ln(L) against time_lag, {} vs {}, binwidth={}, window-size={}'.format(DETECTOR_PAIR[0], 
                                                                                                 DETECTOR_PAIR[1], 
                                                                                                 HISTOGRAM_BIN_WIDTH,
                                                                                                 WINDOW_SIZE))
  ax.legend()
  fig.tight_layout()
  canvas.print_png(filename)


def main():
  Node.rng = np.random.default_rng(12345)
  data = build_timeseries(TRUE_LAG, DETECTOR_PAIR)
  scan_data = calculations(
    data,
    TIME_LAG_STEP_SIZE,
    HISTOGRAM_BIN_WIDTH,
    DETECTOR_PAIR,
    WINDOW_SIZE
  )

  OUTPUT_DIRECTORY.mkdir(exist_ok=True)
  output_filename = next_available_path(
    OUTPUT_DIRECTORY,
    build_plot_filename(DETECTOR_PAIR, TRUE_LAG, TIME_LAG_STEP_SIZE, HISTOGRAM_BIN_WIDTH, WINDOW_SIZE)
  )
  plot_scan(scan_data, TRUE_LAG, output_filename)

  print('detectors: {} and {}'.format(DETECTOR_PAIR[0], DETECTOR_PAIR[1]))
  print('histogram bin width: {:.6f} s'.format(HISTOGRAM_BIN_WIDTH))
  print('time lag step size: {:.6f} s'.format(TIME_LAG_STEP_SIZE))
  print('true lag: {:.6f} s'.format(TRUE_LAG))
  print('best lag: {:.6f} s'.format(scan_data['best_lag']))
  print('plot: {}'.format(output_filename))


if __name__ == '__main__':
  main()
