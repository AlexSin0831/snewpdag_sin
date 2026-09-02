"""
Read all the "SINGLE-ROW" files from each trial, combine them into one table. 
Calculate the variance of score_ref and the mean of H_ref.
"""

import argparse
import csv 
from pathlib import Path 

import numpy as np 

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
  required_fields = {'H_ref', 'score_ref'}
  missing_fields = required_fields.difference(rows[0])
  if missing_fields:
    raise RuntimeError(
        'Trial CSV files are missing required fields: {}'.format(
            ', '.join(sorted(missing_fields))))

  H_ref_list = []
  score_ref_list = []

  for row in rows: 
    H_ref = float(row['H_ref'])
    score_ref = float(row['score_ref'])
    if np.isfinite(H_ref) and np.isfinite(score_ref):
      H_ref_list.append(H_ref)
      score_ref_list.append(score_ref)

  if len(score_ref_list) < 2:
    raise RuntimeError(
        'At least two trials with finite H_ref and score_ref are required')

  H_ref_expected = np.mean(H_ref_list)
  J_hat = np.var(score_ref_list, ddof=1)
  mean_score = np.mean(score_ref_list)
  mean_score_standard_error = np.sqrt(J_hat / len(score_ref_list))

  if not np.isfinite(H_ref_expected) or H_ref_expected <= 0.0:
    raise RuntimeError(
        'The mean reference curvature must be finite and positive; got {}'.format(
            H_ref_expected))

  return (H_ref_expected, J_hat, mean_score,
          mean_score_standard_error, len(score_ref_list))

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

    (H_ref_expected, J_hat, mean_score,
     mean_score_standard_error, number_of_trials) = calculator(rows)

    sigma_godambe = np.sqrt(J_hat)/H_ref_expected

    print('Number of finite trials = {}'.format(number_of_trials))
    print('The expected value of curvature (true lag) = {:.4f} (ms^-2)'.format(H_ref_expected / 1000**2))
    print('The variance of the score function (true lag) = {:.4f} (ms^-2)'.format(J_hat / 1000**2))
    print('The mean score function (true lag) = {:.4f} +/- {:.4f} (ms^-1)'.format(
        mean_score / 1000, mean_score_standard_error / 1000))
    print('Standard Error (Godambe) = {:.4f} (ms)'.format(sigma_godambe*1000))

if __name__ == '__main__':
   main()
