"""
PoissonLagLikelihood_v6           : Calculate the Poisson likelihood of a histogram pair corresponding to a certain time lag
                                    Expect to have FLOAT values in the histograms due to the smoothing. 
                                    Therefore, we used interpolation to estimate the values of lnJ(n,m), where n and m are floats

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

import numpy as np
import scipy.special as sc
from scipy.interpolate import RegularGridInterpolator

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field

def log_power(x, n):
  if n == 0:
    return 0.0
  if x <= 0.0:
    return -np.inf
  return n * np.log(x)

# The most important and expensive calculation part: 
def lnJ_table_generator(nmax, mmax, a, b, p, q): 
  s = a + p 
  if s <= 0.0:
    return np.full((nmax+1, mmax+1), -np.inf)
  ln_s = np.log(s) if s > 0.0 else -np.inf 
  ln_a = np.log(a) if a > 0.0 else -np.inf
  ln_p = np.log(p) if p > 0.0 else -np.inf

  # Initialisation of the table: 
  lnJ_table = np.full((nmax+1, mmax+1), -np.inf) 

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

class PoissonLagLikelihood_v6(Node):
  """Evaluate the likelihood for one histogram pair at one guessed lag."""

  def __init__(self, in_hist_field, out_field, **kwargs):
    self.in_hist_field = in_hist_field
    self.out_field = out_field
    self.sen_1 = kwargs.pop('sen_1', 1.0)
    self.sen_2 = kwargs.pop('sen_2', 1.0)
    self.bg_1 = kwargs.pop('bg_1', 0.0)
    self.bg_2 = kwargs.pop('bg_2', 0.0)
    self.lnJ_table_cache = {}
    self.ln_factorial_cache = np.asarray([0.0], dtype=np.float64)
    self.ln_factorial_cache.flags.writeable = False
    super().__init__(**kwargs)

  def retrieve_lnJ_table(self, nmax, mmax, a, b, p, q):
    # Table's key:
    key = (float(a), float(b), float(p), float(q))
    table = self.lnJ_table_cache.get(key)
    
    # Return our cached table directly, if all conditions are fulfilled: 
    if table is not None and table.shape[0] > nmax and table.shape[1] > mmax:
      return table

    # If there's a table cached originally, but it is not big enough, then we need to build a new, bigger one:
    if table is not None:
      nmax = max(nmax, table.shape[0] - 1)
      mmax = max(mmax, table.shape[1] - 1)
    table = lnJ_table_generator(nmax, mmax, a, b, p, q)
    table.flags.writeable = False

    # update our cache
    self.lnJ_table_cache[key] = table
    return table

  def table_interpolation_calculator(self, lnJ_table, n:float, m:float) -> float:
    """
    Let's say n and m are floats (due to the smoothing function)
    Then now we need to estimate the value of lnJ with float arguments by interpolating
    the values of lnJ of integer arguments (neighbouring our point of interest)
    """
    nl = int(np.floor(n))
    nu = int(np.ceil(n))
    ml = int(np.floor(m))
    mu = int(np.ceil(m))

    if nl == nu and ml == mu: 
        return lnJ_table[nl,ml]
    elif nl == nu: 
        # Do linear interpolation in m-axis only:
        diff = m - ml 
        result = (1 - diff) * lnJ_table[nl, ml] + diff * lnJ_table[nl, mu]
        return float(result)
    elif ml == mu: 
        # Do linear interpolation in n-axis only:
        diff = n - nl 
        result = (1 - diff) * lnJ_table[nl, ml] + diff * lnJ_table[nu, ml]
        return float(result)
    else: 
        # Do Bilinear interpolation: 
        n_axis = np.array([nl, nu])
        m_axis = np.array([ml, mu])

        integer_datapoints = np.array([
            [lnJ_table[nl,ml],lnJ_table[nl,mu]],
            [lnJ_table[nu,ml],lnJ_table[nu,mu]]
        ])

        interp = RegularGridInterpolator((n_axis,m_axis), integer_datapoints, method='linear') # this is a mathematical object
        count_pair = np.array([n,m])
        result = interp(count_pair)

        return float(result)

  def total_sum_lnJ_gammaln_calculator(self, pair):
    bin_width = pair.get('bin_width', 0.002)

    # Be cautious that, a,b,p,q are all defined with respect to the bin-width
    # While, those given parameters have units of per sec. 
    a = self.sen_1 * bin_width
    p = self.sen_2 * bin_width 
    b = self.bg_1 * bin_width 
    q = self.bg_2 * bin_width 

    # Smoothing function return floats
    h1 = np.asarray(pair['hist1'], dtype=np.float64)
    h2 = np.asarray(pair['hist2'], dtype=np.float64)

    # integer upper bound of our lnJ table:
    nmax = int(np.ceil(np.max(h1)))
    mmax = int(np.ceil(np.max(h2)))

    lnJ_table = self.retrieve_lnJ_table(nmax, mmax, a, b, p, q)

    # Summing the contribution of all bin-pairs:  
    total_sum_lnJ = 0.0
    sum_ln_n_fac = 0.0
    sum_ln_m_fac = 0.0

    for n, m in zip(h1,h2): 
      total_sum_lnJ += self.table_interpolation_calculator(lnJ_table, n, m)
      sum_ln_n_fac += sc.gammaln(n+1)
      sum_ln_m_fac += sc.gammaln(m+1)

    return float(total_sum_lnJ), float(sum_ln_n_fac), float(sum_ln_m_fac)

  def alert(self, data):
    hist_data, valid = fetch_field(data, self.in_hist_field)
    if not valid:
      return False

    if 'hist1' in hist_data and 'hist2' in hist_data:
      pair = hist_data
      h1 = pair['hist1']
      h2 = pair['hist2']
      h1 = np.asarray(h1, dtype=np.float64)
      h2 = np.asarray(h2, dtype=np.float64)

      # It is no longer meaningful to cache the ln(n!) values while most of those are non-integers.
      sum_lnJ, sum_ln_n_fac, sum_ln_m_fac = self.total_sum_lnJ_gammaln_calculator(pair) 
      total_log_likelihood = sum_lnJ - sum_ln_n_fac - sum_ln_m_fac
      
      if not np.isfinite(total_log_likelihood):
        return False
      result = {
          'lag': float(pair['lag']),
          'log_likelihood': float(total_log_likelihood),
      }
      store_field(data, self.out_field, result)
      return True
    elif 'pairs' in hist_data and len(hist_data['pairs']) == 1:
      pair = hist_data['pairs'][0]
      h1 = pair['hist1']
      h2 = pair['hist2']
      h1 = np.asarray(h1, dtype=np.float64)
      h2 = np.asarray(h2, dtype=np.float64)

      sum_lnJ, sum_ln_n_fac, sum_ln_m_fac = self.total_sum_lnJ_gammaln_calculator(pair) 
      total_log_likelihood = sum_lnJ - sum_ln_n_fac - sum_ln_m_fac
      
      if not np.isfinite(total_log_likelihood):
        return False
      result = {
          'lag': float(pair['lag']),
          'log_likelihood': float(total_log_likelihood),
      }
      store_field(data, self.out_field, result)
      return True
    elif 'pairs' in hist_data and len(hist_data['pairs']) > 1:
      pairs = hist_data['pairs']
      lags = []
      like = []
      for pair in pairs: 
        lag = float(pair['lag'])
        h1 = pair['hist1']
        h2 = pair['hist2']
        h1 = np.asarray(h1, dtype=np.float64)
        h2 = np.asarray(h2, dtype=np.float64)

        # It is no longer meaningful to cache the ln(n!) values while most of those are non-integers.
        sum_lnJ, sum_ln_n_fac, sum_ln_m_fac = self.total_sum_lnJ_gammaln_calculator(pair) 
        total_log_likelihood = sum_lnJ - sum_ln_n_fac - sum_ln_m_fac

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
