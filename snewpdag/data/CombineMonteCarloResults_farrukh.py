"""
Combine Monte Carlo trial CSV files and compare Gaussian pull fits.

The reference fit is the unbinned Gaussian maximum-likelihood fit used by
CombineMonteCarloResults_norm.py.  The Farrukh fit minimizes a binned Pearson
chi-square, so its result can depend on the pull-histogram binning.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import minimize
from scipy.special import erf


def parse_args():
  parser = argparse.ArgumentParser(
      description='Combine Monte Carlo results and compare Gaussian pull fits.'
  )

  parser.add_argument(
      '--run-dir',
      type=Path,
      default=None,
      help='Monte Carlo campaign directory containing results/.'
  )
  parser.add_argument(
      '--results-dir',
      type=Path,
      default=None,
      help='Directory containing per-trial CSV files.'
  )
  parser.add_argument(
      '--combined-csv',
      type=Path,
      default=None,
      help='Output CSV file containing all trials.'
  )
  parser.add_argument(
      '--pull-plot',
      type=Path,
      default=None,
      help='Output PNG file for the pull histogram and fitted curves.'
  )
  parser.add_argument(
      '--bins',
      type=int,
      default=30,
      help='Number of histogram bins used for the pull plot and Farrukh fit.'
  )

  return parser.parse_args()


def read_trial_rows(results_dir):
  rows = []
  fieldnames = None

  for path in sorted(results_dir.glob('trial_*.csv')):
    with path.open(newline='') as f:
      reader = csv.DictReader(f)
      for row in reader:
        row['source_file'] = str(path)
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


def gaussian_pdf(x, mean, std):
  """Return the normal probability density at x."""
  return (
      np.exp(-0.5 * ((x - mean) / std) ** 2)
      / (std * np.sqrt(2.0 * np.pi))
  )


def gaussian_bin_probabilities(bin_edges, mean, std):
  """Return the Gaussian probability within each histogram bin."""
  z = (bin_edges - mean) / (std * np.sqrt(2.0))
  cdf = 0.5 * (1.0 + erf(z))
  return np.diff(cdf)


def unbinned_gaussian_fit(data):
  """
  Fit a Gaussian using its unbinned maximum-likelihood estimators.

  These are the same mean and population standard deviation used by
  CombineMonteCarloResults_norm.py.
  """
  return {
      'mean': float(np.mean(data)),
      'std': float(np.std(data, ddof=0))
  }


def chi_square_gaussian_fit(data, bin_edges):
  """Fit a Gaussian by minimizing the binned Pearson chi-square."""
  data = np.asarray(data, dtype=np.float64)
  bin_edges = np.asarray(bin_edges, dtype=np.float64)

  if data.ndim != 1 or data.size < 2:
    raise ValueError('At least two pull values are required for a Gaussian fit')
  if bin_edges.ndim != 1 or bin_edges.size < 4:
    raise ValueError('At least three histogram bins are required')
  if not np.all(np.isfinite(data)):
    raise ValueError('Pull values must all be finite')
  if not np.all(np.isfinite(bin_edges)) or np.any(np.diff(bin_edges) <= 0.0):
    raise ValueError('Histogram bin edges must be finite and increasing')

  observed, _ = np.histogram(data, bins=bin_edges)
  n = data.size
  initial_mean = float(np.mean(data))
  initial_std = float(np.std(data, ddof=0))

  if initial_std <= 0.0:
    raise ValueError('Cannot fit a Gaussian to identical pull values')

  data_span = max(float(np.ptp(data)), initial_std)
  minimum_std = max(data_span * 1.0e-6, np.finfo(np.float64).eps)
  maximum_std = data_span * 10.0
  mean_padding = data_span
  expected_floor = max(n * np.finfo(np.float64).eps, 1.0e-12)

  def chi_square(parameters):
    mean, log_std = parameters
    std = np.exp(log_std)
    probabilities = gaussian_bin_probabilities(bin_edges, mean, std)
    expected = n * np.clip(probabilities, 0.0, None)

    # Keeping a small positive expectation prevents an invalid all-zero-bin
    # solution when the optimizer briefly explores a very poor parameter set.
    expected = np.maximum(expected, expected_floor)
    return float(np.sum((observed - expected) ** 2 / expected))

  result = minimize(
      chi_square,
      x0=(initial_mean, np.log(initial_std)),
      method='L-BFGS-B',
      bounds=(
          (
              float(np.min(data)) - mean_padding,
              float(np.max(data)) + mean_padding
          ),
          (np.log(minimum_std), np.log(maximum_std))
      )
  )

  if not result.success or not np.isfinite(result.fun):
    raise RuntimeError(
        'Farrukh Gaussian fit failed: {}'.format(result.message)
    )

  mean, log_std = result.x
  return {
      'mean': float(mean),
      'std': float(np.exp(log_std)),
      'chi2': float(result.fun),
      'dof': int(observed.size - 2)
  }


def plot_pull_distribution(rows, pull_plot, bins):
  if bins < 3:
    raise ValueError('--bins must be at least 3 for the Farrukh fit')

  normalised_pulls = []
  for row in rows:
    try:
      pull = float(row['pull'])
    except (KeyError, TypeError, ValueError):
      continue

    if np.isfinite(pull):
      normalised_pulls.append(pull)

  normalised_pulls = np.asarray(normalised_pulls, dtype=np.float64)
  if normalised_pulls.size < 2:
    return None

  _, bin_edges = np.histogram(normalised_pulls, bins=bins)
  reference_fit = unbinned_gaussian_fit(normalised_pulls)
  farrukh_fit = chi_square_gaussian_fit(normalised_pulls, bin_edges)

  n = normalised_pulls.size
  reference_fit['mean_error'] = reference_fit['std'] / np.sqrt(n)
  reference_fit['std_error'] = reference_fit['std'] / np.sqrt(2.0 * n)
  farrukh_fit['mean_error'] = farrukh_fit['std'] / np.sqrt(n)
  farrukh_fit['std_error'] = farrukh_fit['std'] / np.sqrt(2.0 * n)

  pull_plot.parent.mkdir(parents=True, exist_ok=True)
  fig = Figure(figsize=(9, 8))
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)

  ax.hist(
      normalised_pulls,
      bins=bin_edges,
      density=True,
      edgecolor='black',
      alpha=0.75,
      label='Trials'
  )

  x_padding = 0.2 * max(
      float(np.ptp(normalised_pulls)),
      reference_fit['std'],
      farrukh_fit['std']
  )
  x = np.linspace(
      float(np.min(normalised_pulls)) - x_padding,
      float(np.max(normalised_pulls)) + x_padding,
      500
  )

  ax.plot(
      x,
      gaussian_pdf(x, reference_fit['mean'], reference_fit['std']),
      color='tab:orange',
      linestyle='--',
      linewidth=3,
      label=(
          'Norm/unbinned fit: mean = {:.3f} ± {:.3f}, '
          'std = {:.3f} ± {:.3f}'
      ).format(
          reference_fit['mean'],
          reference_fit['mean_error'],
          reference_fit['std'],
          reference_fit['std_error']
      )
  )
  ax.plot(
      x,
      gaussian_pdf(x, farrukh_fit['mean'], farrukh_fit['std']),
      color='tab:blue',
      linewidth=3,
      label=(
          'Farrukh/binned χ² fit: mean = {:.3f} ± {:.3f}, '
          'std = {:.3f} ± {:.3f}, χ²/dof = {:.1f}/{}'
      ).format(
          farrukh_fit['mean'],
          farrukh_fit['mean_error'],
          farrukh_fit['std'],
          farrukh_fit['std_error'],
          farrukh_fit['chi2'],
          farrukh_fit['dof']
      )
  )

  metadata = rows[0]
  det1 = metadata.get('det1', 'unknown')
  det2 = metadata.get('det2', 'unknown')
  hist_bin_width = metadata.get('hist_bin_width', 'unknown')

  ax.axvline(0.0, color='tab:red', linestyle='--', label='zero')
  ax.set_xlabel('Pull = (best_lag - true_lag)/sigma')
  ax.set_ylabel('Probability density')
  ax.set_title(
      'Pull distribution (N = {}), {} vs {}, bin-width = {}'.format(
          n, det1, det2, hist_bin_width
      )
  )
  ax.legend()
  fig.tight_layout()
  canvas.print_png(pull_plot)

  return {
      'n': int(n),
      'mean': farrukh_fit['mean'],
      'mean_error': farrukh_fit['mean_error'],
      'std': farrukh_fit['std'],
      'std_error': farrukh_fit['std_error'],
      'chi2': farrukh_fit['chi2'],
      'dof': farrukh_fit['dof'],
      'reference_mean': reference_fit['mean'],
      'reference_std': reference_fit['std']
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
    raise RuntimeError(
        'No trial_*.csv files found in {}'.format(args.results_dir)
    )

  write_combined_csv(rows, fieldnames, args.combined_csv)
  summary = plot_pull_distribution(rows, args.pull_plot, args.bins)

  print('combined trials: {}'.format(len(rows)))
  print('combined csv: {}'.format(args.combined_csv))
  if summary is not None:
    print('pull plot: {}'.format(args.pull_plot))
    print('norm/unbinned pull mean: {:.6f}'.format(summary['reference_mean']))
    print('norm/unbinned pull std: {:.6f}'.format(summary['reference_std']))
    print('Farrukh pull mean: {:.6f}'.format(summary['mean']))
    print('Farrukh pull std: {:.6f}'.format(summary['std']))
    print(
        'Farrukh chi2/dof: {:.6f}/{}'.format(
            summary['chi2'], summary['dof']
        )
    )
    print(
        'fit difference (Farrukh - norm): mean = {:+.6f}, std = {:+.6f}'.format(
            summary['mean'] - summary['reference_mean'],
            summary['std'] - summary['reference_std']
        )
    )
  else:
    print('pull plot not written: fewer than two finite pull values')


if __name__ == '__main__':
  main()
