'''
Brutally compute every single terms
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

# MOST IMPORTANT PART!!!!!!
def log_likelihood_calculator(n, m, alpha, rho):
  # we are only calculating one single bin here 
  log_term_list_of_the_bin = []
  for r in range(n + 1):
    for j in range(m + 1):
      k = n - r + m - j
      single_log_term = (
          sc.gammaln(k + 1.0)
          - sc.gammaln(n - r + 1.0)
          - sc.gammaln(m - j + 1.0)
          + log_power(alpha, r)
          + log_power(rho, j)
          - sc.gammaln(r + 1.0)
          - sc.gammaln(j + 1.0)
      )
      if np.isfinite(single_log_term):
        log_term_list_of_the_bin.append(single_log_term)

  # Our main goal: Compute ln(T_1 + T_2 + ... + T_{(n_i+1)*(m_i+1)}) ___ (star)
  # However, this will be too dangerous for the computer to calculate this "directly"
  # We will calculate ln(T_1), ln(T_2), ...., ln(T_{(n_i+1)*(m_i+1)}) first
  # Then we use logsumexp, it is calculating ln(e^{ln(T_1)} + e^{ln(T_2)} + ... + e^{ln(T_{(n_i+1)*(m_i+1)})}) = (star) 

  return sc.logsumexp(log_term_list_of_the_bin)


class PoissonLagLikelihood_v1(Node):
  """Evaluate the likelihood for one histogram pair at one guessed lag."""

  def __init__(self, in_hist_field, out_field, **kwargs):
    self.in_hist_field = in_hist_field
    self.out_field = out_field
    self.sensitivity_1 = kwargs.pop('sensitivity_1', kwargs.pop('a', 1.0))
    self.sensitivity_2 = kwargs.pop('sensitivity_2', kwargs.pop('p', 1.0))
    self.background_1 = kwargs.pop('background_1', kwargs.pop('bg_1', 0.0))
    self.background_2 = kwargs.pop('background_2', kwargs.pop('bg_2', 0.0))
    self.background_is_rate = kwargs.pop('background_is_rate', False)
    super().__init__(**kwargs)

  
  def pair_total_log_likelihood(self, pair):
    bin_width = pair.get('bin_width', 1.0)
    b = self.background_1 * bin_width if self.background_is_rate else self.background_1
    q = self.background_2 * bin_width if self.background_is_rate else self.background_2

    alpha = b * (1 + self.sensitivity_2 / self.sensitivity_1)
    rho = q * (1 + self.sensitivity_1 / self.sensitivity_2)

    h1 = np.asarray(pair['hist1'], dtype=np.int64)
    h2 = np.asarray(pair['hist2'], dtype=np.int64)

    total_log_likelihood = 0 

    # F_i = \sum_{r=0}^{n_i} \sum_{j=0}^m_i \binom{n_i - r + m_i - j}{n_i - r} \frac{\alpha^r \rho^j}{r! j!}
    # ln(L_total) = ln(F_1) + ln(F_2) + ... + ln(F_B)
    for n, m in zip(h1, h2):
      total_log_likelihood += log_likelihood_calculator(int(n), int(m), alpha, rho)

    return float(total_log_likelihood)

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

    total_log_likelihood = self.pair_total_log_likelihood(pair)
    if not np.isfinite(total_log_likelihood):
      return False
    result = {
        'lag': float(pair['lag']),
        'log_likelihood': float(total_log_likelihood),
    }
    store_field(data, self.out_field, result)
    return True
