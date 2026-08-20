"""
Read all the "SINGLE-ROW" files from each trial, combine them into one table. 
Calculate the variance of the score_ref and the mean of the H_ref and H_best.
"""

import argparse
import csv 
from pathlib import Path 

import numpy as np 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
from matplotlib.figure import Figure 

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_MC_DIR = PROJECT_ROOT / 'output' / 'mc_v9_test'

def parse_args():
  parser = argparse.ArgumentParser(description='Compute the terms of godambe information')

  # run-dir is the folder for the entire Monte Carlo campaign 
  parser.add_argument('--run-dir', type=Path, default=None,
                      help='Monte Carlo campaign directory containing results/.')
  
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

def calculator(rows): 
  H_obs_list = []
  H_ref_list = []
  score_ref_list = []

  for row in rows: 
    H_obs_list.append(float(row['H_obs']))
    H_ref_list.append(float(row['H_ref']))
    score_ref_list.append(float(row['score_ref']))
   
  H_obs_expected = np.mean(H_obs_list)
  H_ref_expected = np.mean(H_ref_list)
  J_hat = np.var(score_ref_list, ddof=1)

  return H_obs_expected, H_ref_expected, J_hat 

def main(): 
    args = parse_args()
    if args.run_dir is None:
        raise RuntimeError('Please provide --run-dir')

    args.results_dir = args.run_dir / 'results'

    args.combined_csv = args.run_dir / 'summary.csv'

    rows, fieldnames = read_trial_rows(args.results_dir)
    if not rows:
        raise RuntimeError('No trial_*.csv files found in {}'.format(args.results_dir))

    write_combined_csv(rows, fieldnames, args.combined_csv)

    H_obs_expected, H_ref_expected, J_hat = calculator(rows)

    sigma_godambe = np.sqrt(J_hat)/H_ref_expected

    print('The expected value of curvature (best lag) = {:.4f}'.format(H_obs_expected))
    print('The expected value of curvature (true lag) = {:.4f}'.format(H_ref_expected))
    print('The variance of the score function (true lag) = {:.4f}'.format(J_hat))
    print('Standard Error (Godambe) = {:.4f} (ms)'.format(sigma_godambe*1000))

if __name__ == '__main__': 
   main()

  