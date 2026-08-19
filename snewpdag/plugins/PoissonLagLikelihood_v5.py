"""
PoissonLagLikelihood_v5           : Calculate the Poisson likelihood of a histogram pair corresponding to a certain time lag.
                                    Expect to have integer values in the histograms.
                                    The workflow logic of v5 is the same as v4.
                                    The only difference is that we use a c++ plugin to calculate the double loop for the lnJ table.

configuration: 
    in_hist_field                 : field name of the histogram pair
    out_field                     : field name of the likelihood summary (lag vs log_likelihood)

    (Optional)
    sen_1                         : sensitivity of det1 (per second)
    sen_2                         : sensitivity of det2 (per second)
    bg_1                          : background rate of det1 (per second)
    bg_2                          : background rate of det2 (per second)
"""

import logging
import numbers

import numpy as np
import scipy.special as sc

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field
from snewpdag.plugins.RecursionIntegral import buildLogJTable


def logJTableGenerator(nmax, mmax, a, b, p, q):
  return buildLogJTable(nmax, mmax, a, b, p, q)

class PoissonLagLikelihood_v5(Node):
  """Evaluate the likelihood for one histogram pair at one guessed lag."""

  def __init__(self, in_hist_field, out_field, **kwargs):
    self.in_hist_field = in_hist_field
    self.out_field = out_field
    self.sen_1 = kwargs.pop('sen_1', kwargs.pop('a', 1.0))
    self.sen_2 = kwargs.pop('sen_2', kwargs.pop('p', 1.0))
    self.bg_1 = kwargs.pop('bg_1',0.0)
    self.bg_2 = kwargs.pop('bg_2',0.0)
    self.logJTableCache = {}
    self.ln_factorial_cache = np.asarray([0.0], dtype=np.float64)
    self.ln_factorial_cache.flags.writeable = False
    super().__init__(**kwargs)

  def logJTable(self, nmax, mmax, a, b, p, q):
    key = (float(a), float(b), float(p), float(q))
    table = self.logJTableCache.get(key)
    
    # check whether the cache got a table bigger than or equal to the one we need
    if table is not None and table.shape[0] > nmax and table.shape[1] > mmax:
      return table

    if table is not None:
      nmax = max(nmax, table.shape[0] - 1)
      mmax = max(mmax, table.shape[1] - 1)

    table = logJTableGenerator(nmax, mmax, a, b, p, q)
    table.flags.writeable = False

    # update our cache
    self.logJTableCache[key] = table
    return table

  def ln_factorial_array_generator(self, max_count):
    # check whether the cache got an array bigger than or equal to the one we need 
    if self.ln_factorial_cache.size > max_count:
      return self.ln_factorial_cache

    array = sc.gammaln(np.arange(max_count + 1, dtype=np.float64) + 1.0)
    array.flags.writeable = False
    self.ln_factorial_cache = array
    return array

  
  def totalSumLogJCalculator(self, pair):
    bin_width = pair.get('bin_width', 1.0)
    a = self.sen_1 * bin_width
    p = self.sen_2 * bin_width 
    b = self.bg_1 * bin_width 
    q = self.bg_2 * bin_width 

    h1 = np.asarray(pair['hist1'], dtype=np.int64)
    h2 = np.asarray(pair['hist2'], dtype=np.int64)

    nmax = int(np.max(h1))
    mmax = int(np.max(h2))

    logJTable = self.logJTable(nmax, mmax, a, b, p, q)
    totalSumLogJ = np.sum(logJTable[h1,h2])

    return float(totalSumLogJ)

  def alert(self, data):
    hist_data, valid = fetch_field(data, self.in_hist_field)
    if not valid:
      return False

    if 'hist1' in hist_data and 'hist2' in hist_data:
      pair = hist_data
    elif 'pairs' in hist_data and len(hist_data['pairs']) == 1:
      pair = hist_data['pairs'][0]
    elif 'pairs' in hist_data and len(hist_data['pairs']) > 1:
      pairs = hist_data['pairs']
      lags = []
      like = []
      for pair in pairs:
        lag = float(pair['lag'])
        h1 = pair['hist1']
        h2 = pair['hist2']
        h1 = np.asarray(h1, dtype=np.int64)
        h2 = np.asarray(h2, dtype=np.int64)
        max_factorial_argument = int(max(np.max(h1), np.max(h2)))
        ln_factorial_array = self.ln_factorial_array_generator(max_factorial_argument)

        # We should not assume \sum(\ln(n!m!)) to be constant
        total_log_likelihood = (self.totalSumLogJCalculator(pair) 
                                - np.sum(ln_factorial_array[h1])
                                - np.sum(ln_factorial_array[h2]))

        if not np.isfinite(total_log_likelihood):
          lags.append(lag)
          like.append(np.nan)
        else:
          lags.append(lag)
          like.append(total_log_likelihood)
      result = {
        'possible_time_lag_list': np.asarray(lags, dtype=np.float64),
        'log_likelihood_list': np.asarray(like, dtype=np.float64)
      }
      store_field(data, self.out_field, result)
      return True
    else:
      logging.error('%s: Cannot receive one or more histogram pairs', self.name)
      return False

    h1 = pair['hist1']
    h2 = pair['hist2']
    h1 = np.asarray(h1, dtype=np.int64)
    h2 = np.asarray(h2, dtype=np.int64)
    max_factorial_argument = int(max(np.max(h1), np.max(h2)))
    ln_factorial_array = self.ln_factorial_array_generator(max_factorial_argument)

    # We should not assume \sum(\ln(n!m!)) to be constant
    total_log_likelihood = (self.totalSumLogJCalculator(pair) 
                            - np.sum(ln_factorial_array[h1])
                            - np.sum(ln_factorial_array[h2]))
    
    if not np.isfinite(total_log_likelihood):
      return False
    result = {
        'lag': float(pair['lag']),
        'log_likelihood': float(total_log_likelihood),
    }
    store_field(data, self.out_field, result)
    return True
