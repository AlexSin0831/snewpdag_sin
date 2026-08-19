"""
LikelihoodScanCollector: A collector of the likelihood scan result
                         If coarse-scan, it will narrow down the scanning region for the follow-up fine-scan
                         If fine-scan, it will return the summary for us.

configuration:
  input_field          : field name of the result from PoissonLagLikelihood
                         - lag vs log_likelihood (single lag)
                         - possible_time_lag_list vs log_likelihood_list (multiple lags)
  out_field            : field name of the summary
  scan_type            : coarse / fine
  clear_on             : delete the data saved inside our arrays
"""

import logging
import numbers

import numpy as np
import scipy.special as sc

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field


class LikelihoodScanCollector(Node):
  """Find the best sampled lag and optionally define a fine-scan ROI."""

  def __init__(self, input_field, out_field, **kwargs):
    self.input_field = input_field
    self.out_field = out_field
    self.scan_type = kwargs.pop('scan_type', 'coarse')
    self.clear_on = kwargs.pop('clear_on', ['revoke', 'reset'])
    self.possible_time_lag_list = []
    self.log_likelihood_list = []
    self.sum_lnJ_list = []
    self.sum_ln_fac_list = []
    super().__init__(**kwargs)

  def _scan_arrays(self, likelihood_table):
    if ('possible_time_lag_list' in likelihood_table
        and 'log_likelihood_list' in likelihood_table):
      possible_time_lag_list = np.asarray(
          likelihood_table['possible_time_lag_list'], dtype=np.float64)
      return (
          possible_time_lag_list,
          np.asarray(likelihood_table['log_likelihood_list'], dtype=np.float64),
          np.asarray(
              likelihood_table.get(
                  'sum_lnJ_list', np.full(possible_time_lag_list.shape, np.nan)),
              dtype=np.float64),
          np.asarray(
              likelihood_table.get(
                  'sum_ln_fac_list', np.full(possible_time_lag_list.shape, np.nan)),
              dtype=np.float64),
      )


    if 'lag' in likelihood_table and 'log_likelihood' in likelihood_table:
      self.possible_time_lag_list.append(float(likelihood_table['lag']))
      self.log_likelihood_list.append(float(likelihood_table['log_likelihood']))
      self.sum_lnJ_list.append(float(likelihood_table.get('sum_lnJ', np.nan)))
      self.sum_ln_fac_list.append(
          float(likelihood_table.get('sum_ln_fac', np.nan)))
      return (
          np.asarray(self.possible_time_lag_list, dtype=np.float64),
          np.asarray(self.log_likelihood_list, dtype=np.float64),
          np.asarray(self.sum_lnJ_list, dtype=np.float64),
          np.asarray(self.sum_ln_fac_list, dtype=np.float64),
      )
    return None, None, None, None

  def alert(self, data):
    likelihood_results, valid = fetch_field(data, self.input_field)
    if not valid:
      return False

    # summary:
    (possible_time_lag_list, log_likelihood_list,
     sum_lnJ_list, sum_ln_fac_list) = self._scan_arrays(likelihood_results)

    if possible_time_lag_list is None:
      return False
    if (len(possible_time_lag_list) == 0
        or not np.any(np.isfinite(log_likelihood_list))):
      return False

    index_max = int(np.nanargmax(log_likelihood_list))
    results = {
        'possible_time_lag_list': possible_time_lag_list,
        'log_likelihood_list': log_likelihood_list,
        'sum_lnJ_list': sum_lnJ_list,
        'sum_ln_fac_list': sum_ln_fac_list,
        'best_lag': float(possible_time_lag_list[index_max]),
        'best_log_likelihood': float(log_likelihood_list[index_max]),
        'best_index': index_max,
        'scan_type': self.scan_type,
    }

    # define roi for coarse scan:
    if self.scan_type == 'coarse':
      i_low = max(0, index_max - 5)
      i_high = min(len(possible_time_lag_list) - 1, index_max + 5)
      results['roi'] = {
          'low': float(possible_time_lag_list[i_low]),
          'high': float(possible_time_lag_list[i_high]),
          'center': float(possible_time_lag_list[index_max]),
    }

    store_field(data, self.out_field, results)
    return True

  def revoke(self, data):
    if 'revoke' in self.clear_on:
      self.possible_time_lag_list = []
      self.log_likelihood_list = []
      self.sum_lnJ_list = []
      self.sum_ln_fac_list = []
    return True

  def reset(self, data):
    if 'reset' in self.clear_on:
      self.possible_time_lag_list = []
      self.log_likelihood_list = []
      self.sum_lnJ_list = []
      self.sum_ln_fac_list = []
    return True
