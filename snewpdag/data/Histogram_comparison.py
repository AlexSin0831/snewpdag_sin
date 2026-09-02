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

PROJECT_ROOT = Path(__file__).parents[2]

NewTimeSeries = load_plugin('ops/NewTimeSeries.py').NewTimeSeries
NewHist1D = load_plugin('ops/NewHist1D.py').NewHist1D
Uniform = load_plugin('gen/Uniform.py').Uniform
GenTimeDist = load_plugin('gen/GenTimeDist.py').GenTimeDist
GenTimeDist_IceCube = load_plugin('gen/GenTimeDist_IceCube.py').GenTimeDist_IceCube
PairTimeSeriesToHist = load_plugin('PairTimeSeriesToHist.py').PairTimeSeriesToHist
PairtimeSeriesToHist_Smoothing = load_plugin('PairTimeSeriesToHist_Smoothing.py').PairTimeSeriesToHist_Smoothing
PairtimeSeriesToHist_Smoothing_Toy = load_plugin('PairTimeSeriesToHist_Smoothing_Toy.py').PairTimeSeriesToHist_Smoothing_Toy


# DEFAULT PARAMETERS
TRUE_LAG = 0.022
HISTOGRAM_BIN_WIDTH = 0.002 # s # IceCube default histogram bin-width = 2ms
DETECTOR_PAIR = ('IceCube', 'LVD') # (det1, det2)
WINDOW_SIZE = 2.0 #s
# Generate histogram for IceCube directly --> Speed up the algo A LOT
HIST1_IC = True 
if DETECTOR_PAIR[0] != 'IceCube': 
  HIST1_IC = False

# Convolution Smoothing parameters:
RISE_CONSTANT = 0.02 # 20 ms is the time-scale of the rising edge of lightcurve
FALL_CONSTANT = 2.0
FRAC_RISE = 0.5
FRAC_FALL = 0.5
IMPACT_RANGE = 1.0
MEAN_CORRECTION = False

# Gaussian Normal Distribution (GND) parameters:
TOY = True # Use Gaussian as our kernel
SIGMA_GND = 0.01 # second 

SEED = 1234
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

# Main:
Node.rng = np.random.default_rng(SEED)

payload_data = {'action': 'alert'}

if HIST1_IC == False:
    # det1's time series:
    payload_data = run_node(NewTimeSeries('ts1', name = 'new-ts1'), payload_data)
    payload_data = run_node(Uniform('ts1', rate = detector_background_rate(DETECTOR_PAIR[0]), tmin = -3.0, tmax = 3.0, name = 'bg-ts1'), payload_data)
    payload_data = run_node(GenTimeDist('ts1', 
                                        sig_filename=detector_model_directory(DETECTOR_PAIR[0]),
                                        sig_filetype='tn',
                                        sig_delimiter=',',
                                        sig_mean=detector_yield(DETECTOR_PAIR[0]),
                                        sig_once=False,
                                        sig_t0=0.0,
                                        name = 'model-ts1'), payload_data)
else:
    # IceCube's hist: 
    start_empty = -3.0
    stop_empty = 3.0
    nbins_empty = int((stop_empty - start_empty)/ HISTOGRAM_BIN_WIDTH)
    payload_data = run_node(NewHist1D('hist_ic',
                                    nbins_empty, 
                                    start_empty,
                                    stop_empty,
                                    name = 'new-hist-ic'), payload_data)
    payload_data = run_node(Uniform('hist_ic', rate = detector_background_rate('IceCube'), tmin = -3.0, tmax = 3.0, name = 'bg-hist-ic'), payload_data)
    payload_data = run_node(GenTimeDist_IceCube('hist_ic',
                                                sig_filename=detector_model_directory('IceCube'),
                                                sig_filetype='tn',
                                                sig_delimiter=',',
                                                sig_mean=detector_yield('IceCube'),
                                                sig_once=False,
                                                sig_t0=0.0,
                                                name = 'model-hist_ic'), payload_data)  

# det2's time series:
payload_data = run_node(NewTimeSeries('ts2', name = 'new-ts2'), payload_data)
payload_data = run_node(Uniform('ts2', rate = detector_background_rate(DETECTOR_PAIR[1]), tmin = -3.0, tmax = 3.0, name = 'bg-ts2'), payload_data)
payload_data = run_node(GenTimeDist('ts2', 
                                    sig_filename=detector_model_directory(DETECTOR_PAIR[1]),
                                    sig_filetype='tn',
                                    sig_delimiter=',',
                                    sig_mean=detector_yield(DETECTOR_PAIR[1]),
                                    sig_once=False,
                                    sig_t0=TRUE_LAG,
                                    name = 'model-ts2'), payload_data)  


# Original Method: 
payload_data_1 = run_node(PairTimeSeriesToHist('ts1', 
                                             'ts2', 
                                             'hist_pair_original',
                                             bin_width = HISTOGRAM_BIN_WIDTH,
                                             window_start=-0.5,
                                             window=float(WINDOW_SIZE),
                                             lag=TRUE_LAG,
                                             hist1_ic = HIST1_IC,
                                             in_hist1_field = 'hist_ic',
                                             name='pair_original'), payload_data)

# Smoothing-Function: 
if TOY == True: 
  # Using Gaussian Normal Distribution as our kernel:
  payload_data_2 = run_node(PairtimeSeriesToHist_Smoothing_Toy('ts1',
                                                        'ts2',
                                                        'hist_pair_smoothed',
                                                        bin_width = HISTOGRAM_BIN_WIDTH,
                                                        window_start=-0.5,
                                                        window_size=float(WINDOW_SIZE),
                                                        lag=TRUE_LAG,
                                                        sigma_gnd = SIGMA_GND,
                                                        impact_range = IMPACT_RANGE,
                                                        cache_hist1 = False, 
                                                        hist1_ic = HIST1_IC,
                                                        in_hist1_field = 'hist_ic',
                                                        name='pair_smoothed_toy'), payload_data)
else: 
  # Using the convolution result as our kernel:
  payload_data_2 = run_node(PairtimeSeriesToHist_Smoothing('ts1',
                                                          'ts2',
                                                          'hist_pair_smoothed',
                                                          bin_width = HISTOGRAM_BIN_WIDTH,
                                                          window_start=-0.5,
                                                          window_size=float(WINDOW_SIZE),
                                                          lag=TRUE_LAG,
                                                          rise_constant = RISE_CONSTANT,
                                                          fall_constant = FALL_CONSTANT,
                                                          sigma1 = DETECTOR_RESOLUTION[DETECTOR_PAIR[0]],
                                                          sigma2 = DETECTOR_RESOLUTION[DETECTOR_PAIR[1]],
                                                          frac_fall = FRAC_FALL,
                                                          frac_rise = FRAC_RISE,
                                                          impact_range = IMPACT_RANGE,
                                                          cache_hist1 = False, 
                                                          hist1_ic = HIST1_IC,
                                                          in_hist1_field = 'hist_ic',
                                                          mean_correction = MEAN_CORRECTION,
                                                          name='pair_smoothed'), payload_data)


hist1_original = np.asarray(payload_data_1['hist_pair_original']['hist1'], dtype=np.float64)
hist2_original = np.asarray(payload_data_1['hist_pair_original']['hist2'], dtype=np.float64)

hist1_smoothed = np.asarray(payload_data_2['hist_pair_smoothed']['hist1'], dtype=np.float64)
hist2_smoothed = np.asarray(payload_data_2['hist_pair_smoothed']['hist2'], dtype=np.float64)

bin_edges = np.asarray(payload_data_1['hist_pair_original']['edges'], dtype=np.float64)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

OUTPUT_DIRECTORY = PROJECT_ROOT / 'output'
OUTPUT_DIRECTORY.mkdir(exist_ok=True)

# Plot the histgorams from the same detector (with and without smoothing)
def plot_histogram_comparison(original_hist, smoothed_hist, detector_name, filename):
  fig = Figure(figsize=(8, 5), tight_layout=True)
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)

  ax.stairs(original_hist, bin_edges, label='Original histogram', linewidth=1.2)
  ax.stairs(smoothed_hist, bin_edges, label='Smoothed histogram', linewidth=2.0)
  
  if TOY == True:
    ax.set_title('{} Lightcurve ; Smoothing Kernel : Gaussian (sigma = {}ms)'.format(detector_name, SIGMA_GND*1000))
  else: 
    ax.set_title('{} Lightcurve ; Smoothing Kernel : Convolution'.format(detector_name))

  ax.set_xlabel('Time after the core bounces (sec)')
  ax.set_ylabel('Events per {:.1f} ms bin'.format(HISTOGRAM_BIN_WIDTH * 1000))
  ax.grid(alpha=0.25)
  ax.legend()

  output_path = OUTPUT_DIRECTORY / filename
  canvas.print_figure(output_path, dpi=150)
  return output_path


# Plot the histogram: hist1_original vs hist1_smoothed in the same file.
hist1_plot = plot_histogram_comparison(
  hist1_original,
  hist1_smoothed,
  DETECTOR_PAIR[0],
  f'{DETECTOR_PAIR[0]}_original_vs_smoothed.png'
)

# Plot the histogram: hist2_original vs hist2_smoothed in the same file.
hist2_plot = plot_histogram_comparison(
  hist2_original,
  hist2_smoothed,
  DETECTOR_PAIR[1],
  f'{DETECTOR_PAIR[1]}_original_vs_smoothed.png'
)

print('Wrote {}'.format(hist1_plot))
print('Wrote {}'.format(hist2_plot))

