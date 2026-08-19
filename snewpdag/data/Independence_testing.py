import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parents[2]
MPLCONFIGDIR = PROJECT_ROOT / 'output' / '.matplotlib-cache'
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIGDIR))

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parents[2]))

from snewpdag.dag import Node


def load_plugin(filename):
  module_name = 'snewpdag.plugins.' + filename[:-3].replace('/', '.')
  return importlib.import_module(module_name)


def run_node(node, data):
  node.update(data)
  return node.last_data


NewTimeSeries = load_plugin('ops/NewTimeSeries.py').NewTimeSeries
Uniform = load_plugin('gen/Uniform.py').Uniform
GenTimeDist = load_plugin('gen/GenTimeDist.py').GenTimeDist


OUTPUT_DIRECTORY = PROJECT_ROOT / 'output'

TRUE_LAG = 0.022
HISTOGRAM_BIN_WIDTH = 0.01
DETECTOR_PAIR = ('SNOPLUS', 'SNOPLUS')
BACKGROUND_WINDOW = (-3.0, 3.0)
SEED = 1234

DETECTOR_TOTAL_EVENT_SIGNALS = {
  'SuperK': 7800,
  'JUNO': 7200,
  'SNOPLUS': 280,
  'LVD': 360,
  'IceCube': 660000,
}

DETECTOR_BACKGROUND_RATES = {
  'SuperK': 0.1,
  'JUNO': 0.0015,
  'SNOPLUS': 0.3,
  'LVD': 0.03,
  'IceCube': 1476000,
}

MODEL_DIRECTORY = {
  'SuperK': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-wc.data'),
  'JUNO': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-scint.data'),
  'SNOPLUS': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-scint.data'),
  'LVD': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-scint.data'),
  'IceCube': str(PROJECT_ROOT / 'models' / 'ibd-s27-nmo-wc.data'),
}


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
    valid_names = ', '.join(sorted(MODEL_DIRECTORY))
    raise ValueError(
      "Unknown detector '{}'. Choose one of: {}".format(detector_name, valid_names)
    )
  return MODEL_DIRECTORY[detector_name]


def add_background(payload_data, field, detector_name, background_window):
  bg_start, bg_stop = background_window
  return run_node(
    Uniform(field,
            detector_background_rate(detector_name),
            bg_start,
            bg_stop,
            name='bg-{}'.format(field)),
    payload_data,
  )


def add_signal(payload_data, field, detector_name, model_directory, signal_t0, same_signal):
  return run_node(
    GenTimeDist(field,
                sig_filename=model_directory,
                sig_filetype='tn',
                sig_delimiter=',',
                sig_mean=detector_yield(detector_name),
                sig_once=same_signal,
                sig_smear=not same_signal,
                sig_t0=signal_t0,
                name='model-{}'.format(field)),
    payload_data,
  )


def build_timeseries(true_lag, detector_pair, model_directory_1, model_directory_2,
                     background_window_1=BACKGROUND_WINDOW,
                     background_window_2=BACKGROUND_WINDOW,
                     include_background=False,
                     same_signal=True):
  """Build two time series.

  When same_signal is True, GenTimeDist reuses the same generated signal event
  offsets for both series; the only intended signal difference is sig_t0.
  """
  GenTimeDist.one_series = ()
  GenTimeDist.one_mean = 0

  det1, det2 = detector_pair
  payload_data = {'action': 'alert'}

  payload_data = run_node(NewTimeSeries('ts1', name='new-ts1'), payload_data)
  if include_background:
    payload_data = add_background(payload_data, 'ts1', det1, background_window_1)
  payload_data = add_signal(payload_data, 'ts1', det1, model_directory_1, 0.0, same_signal)

  payload_data = run_node(NewTimeSeries('ts2', name='new-ts2'), payload_data)
  if include_background:
    payload_data = add_background(payload_data, 'ts2', det2, background_window_2)
  payload_data = add_signal(payload_data, 'ts2', det2, model_directory_2, true_lag, same_signal)

  return payload_data


def sorted_times(ts):
  return np.sort(np.asarray(ts.times, dtype=np.float64))


def comparison_summary(ts1, ts2, true_lag):
  t1 = sorted_times(ts1)
  t2 = sorted_times(ts2)
  aligned_t2 = t2 - float(true_lag)

  summary = {
    'n_ts1': len(t1),
    'n_ts2': len(t2),
    'same_count': len(t1) == len(t2),
    'exact_after_shift': False,
    'max_abs_difference_after_shift': None,
  }

  if summary['same_count']:
    differences = aligned_t2 - t1
    summary['max_abs_difference_after_shift'] = float(np.max(np.abs(differences))) if len(differences) else 0.0
    summary['exact_after_shift'] = bool(np.allclose(aligned_t2, t1, rtol=0.0, atol=1.0e-12))

  return summary


def histogram_edges(*time_arrays, bin_width=HISTOGRAM_BIN_WIDTH):
  finite_arrays = [np.asarray(times, dtype=np.float64) for times in time_arrays if len(times)]
  if not finite_arrays:
    return np.arange(0.0, bin_width * 2.0, bin_width)

  all_times = np.concatenate(finite_arrays)
  low = np.floor(np.min(all_times) / bin_width) * bin_width
  high = np.ceil(np.max(all_times) / bin_width) * bin_width
  if high <= low:
    high = low + bin_width
  return np.arange(low, high + bin_width, bin_width)


def plot_timeseries(ts1, ts2, true_lag, detector_pair, bin_width, summary):
  t1 = sorted_times(ts1)
  t2 = sorted_times(ts2)
  t2_aligned = t2 - float(true_lag)

  fig = Figure(figsize=(11, 8), constrained_layout=True)
  FigureCanvas(fig)
  axes = fig.subplots(3, 1, height_ratios=(1.0, 1.5, 1.5), sharex=False)
  ax_raster, ax_raw, ax_aligned = axes

  ax_raster.eventplot([t1, t2],
                      lineoffsets=[1, 0],
                      linelengths=0.75,
                      colors=['tab:blue', 'tab:orange'])
  ax_raster.set_yticks([1, 0])
  ax_raster.set_yticklabels(['ts1 {}'.format(detector_pair[0]),
                             'ts2 {}'.format(detector_pair[1])])
  ax_raster.set_title('Generated event times')
  ax_raster.grid(True, axis='x', alpha=0.25)

  raw_edges = histogram_edges(t1, t2, bin_width=bin_width)
  ax_raw.hist(t1, bins=raw_edges, histtype='step', linewidth=1.6, label='ts1')
  ax_raw.hist(t2, bins=raw_edges, histtype='step', linewidth=1.6,
              label='ts2 raw, t0=+{:.6f}s'.format(true_lag))
  ax_raw.set_ylabel('Events / {:.4f}s'.format(bin_width))
  ax_raw.set_title('Raw histograms')
  ax_raw.legend(loc='upper right')
  ax_raw.grid(True, alpha=0.25)

  aligned_edges = histogram_edges(t1, t2_aligned, bin_width=bin_width)
  ax_aligned.hist(t1, bins=aligned_edges, histtype='step', linewidth=1.6, label='ts1')
  ax_aligned.hist(t2_aligned, bins=aligned_edges, histtype='step', linewidth=1.6,
                  label='ts2 shifted back by {:.6f}s'.format(true_lag))
  ax_aligned.set_xlabel('Time (s)')
  ax_aligned.set_ylabel('Events / {:.4f}s'.format(bin_width))
  ax_aligned.set_title('After removing the starting-point offset')
  ax_aligned.legend(loc='upper right')
  ax_aligned.grid(True, alpha=0.25)

  if summary['same_count']:
    message = 'same count; max |ts2 - lag - ts1| = {:.3e}s'.format(
      summary['max_abs_difference_after_shift']
    )
  else:
    message = 'different counts: ts1={}, ts2={}'.format(summary['n_ts1'], summary['n_ts2'])
  fig.suptitle('Time-series independence check ({})'.format(message))
  return fig


def build_plot_filename(detector_pair, seed, true_lag, include_background, same_signal):
  mode = 'same-signal' if same_signal else 'independent-draws'
  bg = 'with-bg' if include_background else 'signal-only'
  lag_ms = round(float(true_lag) * 1000.0, 3)
  return 'independence_{}_{}_seed-{}_lag-{}ms_{}_{}.png'.format(
    detector_pair[0], detector_pair[1], seed, lag_ms, mode, bg
  )


def parse_args():
  parser = argparse.ArgumentParser(
    description='Plot two generated time series and test whether one is just a shifted version of the other.'
  )
  parser.add_argument('--detector1', default=DETECTOR_PAIR[0], choices=sorted(DETECTOR_TOTAL_EVENT_SIGNALS))
  parser.add_argument('--detector2', default=DETECTOR_PAIR[1], choices=sorted(DETECTOR_TOTAL_EVENT_SIGNALS))
  parser.add_argument('--true-lag', type=float, default=TRUE_LAG,
                      help='Starting-time offset for ts2 in seconds.')
  parser.add_argument('--seed', type=int, default=SEED,
                      help='Random seed for reproducible generated events.')
  parser.add_argument('--bin-width', type=float, default=HISTOGRAM_BIN_WIDTH,
                      help='Histogram bin width in seconds.')
  parser.add_argument('--include-background', action='store_true',
                      help='Add independently generated detector background to both series.')
  parser.add_argument('--independent-draws', action='store_true',
                      help='Generate independent signal events instead of reusing the same signal offsets.')
  parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIRECTORY,
                      help='Directory where the plot PNG will be saved.')
  return parser.parse_args()


def main():
  args = parse_args()
  Node.rng = np.random.default_rng(args.seed)

  detector_pair = (args.detector1, args.detector2)
  same_signal = not args.independent_draws
  if same_signal and detector_yield(args.detector1) != detector_yield(args.detector2):
    raise ValueError(
      'same-signal mode needs detectors with the same signal count. '
      'Use the same detector twice, or pass --independent-draws.'
    )

  data = build_timeseries(args.true_lag,
                          detector_pair,
                          detector_model_directory(args.detector1),
                          detector_model_directory(args.detector2),
                          include_background=False,
                          same_signal=False)

  ts1 = data['ts1']
  ts2 = data['ts2']
  summary = comparison_summary(ts1, ts2, args.true_lag)
  fig = plot_timeseries(ts1, ts2, args.true_lag, detector_pair, args.bin_width, summary)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  plot_path = args.output_dir / build_plot_filename(
    detector_pair, args.seed, args.true_lag, args.include_background, same_signal
  )
  fig.savefig(plot_path, dpi=160)

  print('plot: {}'.format(plot_path))
  print('ts1 events: {}'.format(summary['n_ts1']))
  print('ts2 events: {}'.format(summary['n_ts2']))
  if summary['same_count']:
    print('exact after shifting ts2 back by true_lag: {}'.format(summary['exact_after_shift']))
    print('max abs difference after shift: {:.12g} s'.format(
      summary['max_abs_difference_after_shift']
    ))
  else:
    print('exact after shifting ts2 back by true_lag: False (different event counts)')


if __name__ == '__main__':
  main()
