"""
Read all the "SINGLE-ROW" files from each trial, combine them into one table,
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
        row['source_file'] = str(path) # add one more column for stating the source file
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

def coverage_determination(rows): 
    coverage_1sigma_1 = 0
    coverage_2sigma_1 = 0 
    coverage_1sigma_2 = 0 
    coverage_2sigma_2 = 0 

    for row in rows: 
        best_lag = float(row['best_lag'])
        true_lag = float(row['true_lag'])
        sigma1 = float(row['sigma1'])
        left_error = float(row['left_error'])
        right_error = float(row['right_error'])

        if abs(best_lag - true_lag) <= sigma1:
            coverage_1sigma_1 += 1
        if abs(best_lag - true_lag) <= 2.0 * sigma1: 
            coverage_2sigma_1 += 1
        if true_lag >= left_error and true_lag <= right_error:
            coverage_1sigma_2 += 1
        if true_lag >= best_lag - 2 * (best_lag - left_error) and true_lag <= best_lag + 2 * (right_error - best_lag): 
            coverage_2sigma_2 += 1
    
    n = np.size(rows)
    
    return coverage_1sigma_1/n, coverage_2sigma_1/n, coverage_1sigma_2/n, coverage_2sigma_2/n


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

    rows, fieldnames = read_trial_rows(args.results_dir)
    if not rows:
        raise RuntimeError('No trial_*.csv files found in {}'.format(args.results_dir))

    write_combined_csv(rows, fieldnames, args.combined_csv)
    r1, r2, r3, r4 = coverage_determination(rows)

    print('Percentage of trials within 1-sigma region (2nd derivative method) = {:.2f}%'.format(r1*100))
    print('Percentage of trials within 2-sigma region (2nd derivative method) = {:.2f}%'.format(r2*100))
    print('Percentage of trials within 1-sigma region (0.5 method) = {:.2f}%'.format(r3*100))
    print('Percentage of trials within 2-sigma region (0.5 method) = {:.2f}%'.format(r4*100))



if __name__ == '__main__':
  main()
