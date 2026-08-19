"""
Read all the "SINGLE-ROW" files from each trial, combine them into one table,
and compare the normalised pull distributions from both error methods.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_MC_DIR = PROJECT_ROOT / 'output' / 'mc_v8'


def parse_args():
  parser = argparse.ArgumentParser(
      description='Combine Monte Carlo trial CSV files and compare pull methods.'
  )

  # run-dir is the folder for the entire Monte Carlo campaign 
  parser.add_argument('--run-dir', type=Path, default=None,
                      help='Monte Carlo campaign directory containing results/.')
  
  # result-dir is specificially for the subfolder containing the individual trial CSV files
  parser.add_argument('--results-dir', type=Path, default=None,
                      help='Directory containing per-trial CSV files.')

  parser.add_argument('--combined-csv', type=Path, default=None,
                      help='Output CSV file containing all trials.')
  
  parser.add_argument('--pull-plot', type=Path, default=None,
                      help='Base PNG filename for the two pull histograms.')
  
  parser.add_argument('--bins', type=int, default=30,
                      help='Number of histogram bins for the pull plot.')
  
  return parser.parse_args()

# scan all the one-row CSV file inside this directory:
def read_trial_rows(results_dir):
  rows = []
  fieldnames = []

  # Find every file in result_dir whose name begins with trial_ and ends with .csv
  for path in sorted(results_dir.glob('trial_*.csv')):
    with path.open(newline='') as f: # f is the opened file object (just a temporary variable name)
      '''
      Let's say we have:
      seed, true_lag, best_lag, pull 
      1000, 0.022, 0.027, 4
      Then csv.DictReader will directly save the column names as fieldnames
      reader.fieldnames = ['seed', 'true_lag', 'best_lag', 'pull']
      And reader will just be a dictionary: 
      ['seed': 1000, 'true_lag': 0.022, 'best_lag':0.027, 'pull': 4]
      '''

      reader = csv.DictReader(f) # it treats the first line of the CSV as the column names by default
      for row in reader:
        row['source_file'] = str(path) # add one more column
        rows.append(row)
        for fieldname in list(reader.fieldnames or []) + ['source_file']:
          if fieldname not in fieldnames:
            fieldnames.append(fieldname)

  return rows, fieldnames


def write_combined_csv(rows, fieldnames, combined_csv):
  combined_csv.parent.mkdir(parents=True, exist_ok=True)
  with combined_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def plot_pull_distribution(rows, pull_plot, bins):
  method_specs = (
      ('pull1', 'Second derivative', 'tab:green', 'second_derivative'),
      ('pull2', '0.5 method', 'tab:blue', '0p5_method'),
  )
  pulls_by_method = {}

  for field, label, color, filename_suffix in method_specs:
    values = []
    for row in rows:
      try:
        value = float(row[field])
      except (KeyError, TypeError, ValueError):
        continue
      if np.isfinite(value):
        values.append(value)
    if values:
      pulls_by_method[field] = {
          'label': label,
          'color': color,
          'filename_suffix': filename_suffix,
          'values': np.asarray(values, dtype=np.float64),
      }

  if not pulls_by_method:
    return None

  det1 = rows[0].get('det1', 'unknown')
  det2 = rows[0].get('det2', 'unknown')
  hist_bin_width = rows[0].get('hist_bin_width', 'unknown')

  pull_plot.parent.mkdir(parents=True, exist_ok=True)

  all_pulls = np.concatenate([
      method['values'] for method in pulls_by_method.values()
  ])
  pull_low = float(np.min(all_pulls))
  pull_high = float(np.max(all_pulls))
  if pull_low == pull_high:
    pull_low -= 0.5
    pull_high += 0.5
  common_bins = np.linspace(pull_low, pull_high, bins + 1)

  summaries = {}
  for field, method in pulls_by_method.items():
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    values = method['values']
    n = values.size
    best_fit_mean = float(np.mean(values))
    best_fit_std = float(np.std(values, ddof=0))
    best_fit_mean_error = best_fit_std / np.sqrt(n)
    best_fit_std_error = best_fit_std / np.sqrt(2.0 * n)

    ax.hist(values, bins=common_bins, density=True, alpha=0.55,
            color=method['color'], edgecolor=method['color'],
            label='{} trials (N={})'.format(method['label'], n))

    if best_fit_std > 0.0:
      x_padding = 0.2 * max(np.ptp(values), best_fit_std)
      x = np.linspace(np.min(values) - x_padding,
                      np.max(values) + x_padding, 500)
      normal_distribution = (
          np.exp(-0.5 * ((x - best_fit_mean) / best_fit_std) ** 2)
          / (best_fit_std * np.sqrt(2.0 * np.pi))
      )
      ax.plot(
          x, normal_distribution, color=method['color'], linewidth=2.5,
          label=('{} fit: mean={:.3f} ± {:.3f}, '
                 'std={:.3f} ± {:.3f}').format(
              method['label'],
              best_fit_mean, best_fit_mean_error,
              best_fit_std, best_fit_std_error))

    summaries[field] = {
        'n': int(n),
        'mean': best_fit_mean,
        'mean_error': best_fit_mean_error,
        'std': best_fit_std,
        'std_error': best_fit_std_error,
    }

    ax.axvline(0.0, color='tab:red', linestyle='--', label='zero')
    ax.set_xlabel('Pull = (best_lag - true_lag)/sigma')
    ax.set_ylabel('Probability density')
    ax.set_title('{} pull distribution, {} vs {}, bin-width = {}'.format(
        method['label'], det1, det2, hist_bin_width))
    ax.legend()
    fig.tight_layout()

    method_plot = pull_plot.with_name(
        '{}_{}{}'.format(
            pull_plot.stem, method['filename_suffix'], pull_plot.suffix))
    canvas.print_png(method_plot)
    summaries[field]['plot'] = method_plot

  return summaries


def main():
  args = parse_args()
  if args.run_dir is None and args.results_dir is None:
    raise RuntimeError('Please provide --run-dir or --results-dir')

  if args.results_dir is None:
    args.results_dir = args.run_dir / 'results'

  if args.run_dir is None:
    args.run_dir = args.results_dir

  if args.combined_csv is None:
    args.combined_csv = args.run_dir / 'summary.csv'

  if args.pull_plot is None:
    args.pull_plot = args.run_dir / 'pull_distribution_norm.png'

  rows, fieldnames = read_trial_rows(args.results_dir)
  if not rows:
    raise RuntimeError('No trial_*.csv files found in {}'.format(args.results_dir))

  write_combined_csv(rows, fieldnames, args.combined_csv)
  summary = plot_pull_distribution(rows, args.pull_plot, args.bins)

  print('combined trials: {}'.format(len(rows)))
  print('combined csv: {}'.format(args.combined_csv))
  if summary is not None:
    for field, label in (('pull1', 'second derivative'),
                         ('pull2', '0.5 method')):
      if field in summary:
        print('{} pull plot: {}'.format(label, summary[field]['plot']))
        print('{} pull mean: {:.6f} ± {:.6f}'.format(
            label, summary[field]['mean'], summary[field]['mean_error']))
        print('{} pull std: {:.6f} ± {:.6f}'.format(
            label, summary[field]['std'], summary[field]['std_error']))


if __name__ == '__main__':
  main()
