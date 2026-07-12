'''
Completely different approach (Recursion)
Maximise \sum_i^B (\ln(J_nm) + \ln(n!m!))
Implemented technique: cache
'''

import logging
import numbers

import numpy as np
import scipy.special as sc

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field

def log_power(x, n):
  if n == 0:
    return 0.0
  if x <= 0.0:
    return -np.inf
  return n * np.log(x)


def lnJ_table_generator(nmax, mmax, a, b, p, q): 
  s = a + p 
  if s <= 0.0:
    return np.full((nmax+1, mmax+1), -np.inf)
  ln_s = np.log(s)
  ln_a = np.log(a) if a > 0.0 else -np.inf
  ln_p = np.log(p) if p > 0.0 else -np.inf

  # Making J_{nm} table, initialising it where all J_{nm} = 0 
  lnJ_table = np.full((nmax+1, mmax+1), -np.inf) 

  # for IceCube, this will be the process that it spends the longest time on... (nxm loops)
  for n in range(nmax+1): 
    for m in range(mmax+1): 
      terms = []

      # first term: 
      terms.append(log_power(b,n) + log_power(q,m) - ln_s)
      # second term: 
      if n > 0 and a > 0.0: 
        terms.append(np.log(n) + ln_a - ln_s + lnJ_table[n-1,m])
      # third term:
      if m > 0 and p > 0.0: 
        terms.append(np.log(m) + ln_p - ln_s + lnJ_table[n,m-1])
      
      lnJ_table[n,m] = sc.logsumexp(terms)
  
  return lnJ_table

class PoissonLagLikelihood_v4(Node):
  """Evaluate the likelihood for one histogram pair at one guessed lag."""

  def __init__(self, in_hist_field, out_field, **kwargs):
    self.in_hist_field = in_hist_field
    self.out_field = out_field
    self.sensitivity_1 = kwargs.pop('sensitivity_1', kwargs.pop('a', 1.0))
    self.sensitivity_2 = kwargs.pop('sensitivity_2', kwargs.pop('p', 1.0))
    self.background_rate_1 = kwargs.pop('background_rate_1',0.0)
    self.background_rate_2 = kwargs.pop('background_rate_2',0.0)
    self.lnJ_table_cache = {}
    self.ln_factorial_cache = np.asarray([0.0], dtype=np.float64)
    self.ln_factorial_cache.flags.writeable = False
    super().__init__(**kwargs)

  def lnJ_table(self, nmax, mmax, a, b, p, q):
    key = (float(a), float(b), float(p), float(q))
    table = self.lnJ_table_cache.get(key)
    
    # check whether the cache got a table bigger than or equal to the one we need
    if table is not None and table.shape[0] > nmax and table.shape[1] > mmax:
      return table

    if table is not None:
      nmax = max(nmax, table.shape[0] - 1)
      mmax = max(mmax, table.shape[1] - 1)

    table = lnJ_table_generator(nmax, mmax, a, b, p, q)
    table.flags.writeable = False

    # update our cache
    self.lnJ_table_cache[key] = table
    return table

  def ln_factorial_array_generator(self, max_count):
    # check whether the cache got an array bigger than or equal to the one we need 
    if self.ln_factorial_cache.size > max_count:
      return self.ln_factorial_cache

    array = sc.gammaln(np.arange(max_count + 1, dtype=np.float64) + 1.0)
    array.flags.writeable = False
    self.ln_factorial_cache = array
    return array

  
  def total_sum_lnJ_calculator(self, pair):
    bin_width = pair.get('bin_width', 1.0)
    a = self.sensitivity_1 * bin_width
    p = self.sensitivity_2 * bin_width 
    b = self.background_rate_1 * bin_width 
    q = self.background_rate_2 * bin_width 

    h1 = np.asarray(pair['hist1'], dtype=np.int64)
    h2 = np.asarray(pair['hist2'], dtype=np.int64)

    nmax = int(np.max(h1))
    mmax = int(np.max(h2))

    lnJ_table = self.lnJ_table(nmax, mmax, a, b, p, q)
    total_sum_lnJ = np.sum(lnJ_table[h1,h2])

    return float(total_sum_lnJ)

  def alert(self, data):
    hist_data, valid = fetch_field(data, self.in_hist_field)
    if not valid:
      return False

    if 'hist1' in hist_data and 'hist2' in hist_data:
      pair = hist_data
    elif 'pairs' in hist_data and len(hist_data['pairs']) == 1:
      pair = hist_data['pairs'][0]
    else:
      logging.error('%s: Cannot receive one histogram pair', self.name)
      return False

    h1 = pair['hist1']
    h2 = pair['hist2']
    h1 = np.asarray(h1, dtype=np.int64)
    h2 = np.asarray(h2, dtype=np.int64)
    max_factorial_argument = int(max(np.max(h1), np.max(h2)))
    ln_factorial_array = self.ln_factorial_array_generator(max_factorial_argument)

    # We should not assume \sum(\ln(n!m!)) to be constant
    total_log_likelihood = (self.total_sum_lnJ_calculator(pair) 
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
