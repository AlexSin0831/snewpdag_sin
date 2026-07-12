"""
Read all the "SINGLE-ROW" files from each trial, and combine them into ONE SUMMARY TABLE 
Also plot the PULL DISTRIBUTION 
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_MC_DIR = PROJECT_ROOT / 'output' / 'mc_v4'


def parse_args():
  parser = argparse.ArgumentParser(
      description='Combine v4 Monte Carlo trial CSV files.'
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
                      help='Output PNG file for pull histogram.')
  
  parser.add_argument('--bins', type=int, default=20,
                      help='Number of histogram bins for the pull plot.')
  
  return parser.parse_args()

# scan all the one-row CSV file inside this directory:
def read_trial_rows(results_dir):
  rows = []
  fieldnames = None

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
        if fieldnames is None:
          fieldnames = list(reader.fieldnames or []) + ['source_file']

  return rows, fieldnames


def write_combined_csv(rows, fieldnames, combined_csv):
  combined_csv.parent.mkdir(parents=True, exist_ok=True)
  with combined_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def plot_pull_distribution(rows, pull_plot, bins):
  pulls = []
  for row in rows:
    try:
      pull = float(row['pull'])
    except (KeyError, TypeError, ValueError):
      continue
    if np.isfinite(pull):
      pulls.append(pull)

  pulls = np.asarray(pulls, dtype=np.float64)
  if pulls.size == 0:
    return None

  pull_plot.parent.mkdir(parents=True, exist_ok=True)

  fig = Figure(figsize=(8, 8))
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)

  # best-fit parameters of the normal distribution curve:
  best_fit_mean = float(np.mean(pulls))
  best_fit_std = float(np.std(pulls, ddof=0))

  # we normalised the histogram area to be = 1 by setting density = True
  ax.hist(pulls, bins=bins, density=True, edgecolor='black', alpha=0.75, label='Trials')

  if best_fit_std > 0.0:
    x_padding = 0.2 * max(np.ptp(pulls), best_fit_std)

    x = np.linspace(np.min(pulls) - x_padding, np.max(pulls) + x_padding, 500)

    normal_distribution = np.exp(-0.5 * ((x - best_fit_mean) / best_fit_std) ** 2) / (best_fit_std * np.sqrt(2.0 * np.pi))

    ax.plot(x, normal_distribution, color='tab:orange', linewidth=5, label='Normal fit: mean = {:.3f}, std = {:.3f}'.format(best_fit_mean, best_fit_std))

  ax.axvline(0.0, color='tab:red', linestyle='--', label='zero')
  ax.set_xlabel('Pull = (best_lag - true_lag) / sigma')
  ax.set_ylabel('Probability density')
  ax.set_title('Pull distribution, N = {}'.format(pulls.size))
  ax.legend()
  fig.tight_layout()
  canvas.print_png(pull_plot)

  return {
      'n': int(pulls.size),
      'mean': best_fit_mean,
      'std': float(np.std(pulls, ddof=1)) if pulls.size > 1 else np.nan,
  }


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
    args.pull_plot = args.run_dir / 'pull_distribution.png'

  rows, fieldnames = read_trial_rows(args.results_dir)
  if not rows:
    raise RuntimeError('No trial_*.csv files found in {}'.format(args.results_dir))

  write_combined_csv(rows, fieldnames, args.combined_csv)
  summary = plot_pull_distribution(rows, args.pull_plot, args.bins)

  print('combined trials: {}'.format(len(rows)))
  print('combined csv: {}'.format(args.combined_csv))
  if summary is not None:
    print('pull plot: {}'.format(args.pull_plot))
    print('pull mean: {:.6f}'.format(summary['mean']))
    print('pull std: {:.6f}'.format(summary['std']))


if __name__ == '__main__':
  main()
