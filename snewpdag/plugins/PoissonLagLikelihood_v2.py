'''
Implement "NEGLECTING TERMS" method: 
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

def quadratic_solver(a, b, c): 
  det = b**2 - 4 * a * c 
  if  det < 0 or a == 0: 
    return None
  else: 
    x1 = (b + np.sqrt(det))/(2*a)
    x2 = (b - np.sqrt(det))/(2*a)
    return x1, x2
  
def equation_19(r, alpha, rho):
  return - rho * r / alpha + rho - rho / alpha - 1

# Compute ln(T_{rj}):
def single_log_term_calculator(n, r, m, j, alpha, rho): 
  k = n - r + m - j 
  single_log_term = (sc.gammaln(k + 1.0)
                     - sc.gammaln(n - r + 1.0)
                     - sc.gammaln(m - j + 1.0)
                     + log_power(alpha, r)
                     + log_power(rho, j)
                     - sc.gammaln(r + 1.0)
                     - sc.gammaln(j + 1.0))
  return single_log_term


# FIND THE "GLOBAL MAXIMUM" OF THE MATRIX
def find_expansion_centre(n, m, alpha, rho): 
    r1, r2 = quadratic_solver(alpha-rho,
                        alpha * (rho - alpha - n - m) - 2 * rho,
                        (alpha - 1) * (rho + n * alpha) - alpha * (m + 1))
    j1 = equation_19(r1, alpha, rho)
    j2 = equation_19(r2, alpha, rho)

    # BOUNDARY-CHECKING:
    r1_refined = np.ceil(min(max(r1,0), n))
    j1_refined = np.ceil(min(max(j1,0), m))
    r2_refined = np.ceil(min(max(r2,0), n))
    j2_refined = np.ceil(min(max(j2,0), m))

    log_T1 = single_log_term_calculator(n, r1_refined, m, j1_refined, alpha, rho)
    log_T2 = single_log_term_calculator(n, r2_refined, m, j2_refined, alpha, rho)

    # DETERMINE THE EXPANSION CENTRE
    if log_T1 > log_T2: 
        r_max, j_max = r1_refined, j1_refined
    else: 
        r_max, j_max = r2_refined, j2_refined
    
    return int(r_max), int(j_max)

def find_row_j_max(r_fixed, n, m, alpha, rho):
    j_max1, j_max2 = quadratic_solver(1, 
                                    1 - m + r_fixed - n - rho, 
                                    (rho - 1) * m + r_fixed -n)
    
    j_max1_refined = np.ceil(min(max(j_max1,0), m))
    j_max2_refined = np.ceil(min(max(j_max2,0), m))

    T_1 = single_log_term_calculator(n, r_fixed, m, j_max1_refined, alpha, rho)
    T_2 = single_log_term_calculator(n, r_fixed, m, j_max2_refined, alpha, rho)

    if T_1 > T_2: 
       return int(j_max1_refined)
    else: 
       return int(j_max2_refined)
    
def scan_row(r_fixed, j_max, log_sum, n, m, alpha, rho, log_epsilon):
    log_sum = np.logaddexp(log_sum, single_log_term_calculator(n, r_fixed, m, j_max, alpha, rho))
    # TO LEFT:
    j = j_max -1
    while j >= 0: 
        logT_rj = single_log_term_calculator(n, r_fixed, m, j, alpha, rho)
        if logT_rj - log_sum < log_epsilon: 
           break
        else: 
           log_sum = np.logaddexp(log_sum, logT_rj)
           j -= 1
    
    # TO RIGHT: 
    j = j_max + 1
    while j <= m:
        logT_rj = single_log_term_calculator(n, r_fixed, m, j, alpha, rho)
        if logT_rj - log_sum < log_epsilon:
           break
        else: 
           log_sum = np.logaddexp(log_sum, logT_rj)
           j += 1
    
    return log_sum
       
       


class PoissonLagLikelihood_v2(Node):
  """Evaluate the likelihood for one histogram pair at one guessed lag."""

  def __init__(self, in_hist_field, out_field, **kwargs):
    self.in_hist_field = in_hist_field
    self.out_field = out_field
    self.sensitivity_1 = kwargs.pop('sensitivity_1', kwargs.pop('a', 1.0))
    self.sensitivity_2 = kwargs.pop('sensitivity_2', kwargs.pop('p', 1.0))
    self.background_1 = kwargs.pop('background_1', kwargs.pop('bg_1', 0.0))
    self.background_2 = kwargs.pop('background_2', kwargs.pop('bg_2', 0.0))
    self.error = kwargs.pop('error', 0.005)
    super().__init__(**kwargs)

  
  def pair_total_log_likelihood(self, pair):
    bin_width = pair.get('bin_width', 1.0)
    b = self.background_1 * bin_width 
    q = self.background_2 * bin_width 

    alpha = b * (1 + self.sensitivity_2 / self.sensitivity_1)
    rho = q * (1 + self.sensitivity_1 / self.sensitivity_2)

    h1 = np.asarray(pair['hist1'], dtype=np.int64)
    h2 = np.asarray(pair['hist2'], dtype=np.int64)

    total_log_likelihood = 0 

    # F_i = \sum_{r=0}^{n_i} \sum_{j=0}^m_i \binom{n_i - r + m_i - j}{n_i - r} \frac{\alpha^r \rho^j}{r! j!}
    # ln(L_total) = ln(F_1) + ln(F_2) + ... + ln(F_B)

    for n, m in zip(h1, h2):
        r_max, j_max = find_expansion_centre(n, m, alpha, rho) # int
        
        log_epsilon = np.log(self.error / ((n+1) * (m+1)))

        logF_i = -np.inf

        logF_i = scan_row(r_max, j_max, logF_i, n, m, alpha, rho, log_epsilon)

        # UPWARD:
        r = int(r_max) - 1
        while r >= 0: 
            row_j_max = find_row_j_max(r, n, m, alpha, rho) #int
            logT_row_j_max = single_log_term_calculator(n, r, m, row_j_max, alpha, rho)
            if logT_row_j_max - logF_i < log_epsilon: 
                break 
           
            logF_i = scan_row(r, row_j_max, logF_i, n, m, alpha, rho, log_epsilon)
            r -= 1

        # DOWNWARD: 
        r = int(r_max) + 1
        while r <= n: 
            row_j_max = find_row_j_max(r, n, m, alpha, rho) #int
            logT_row_j_max = single_log_term_calculator(n, r, m, row_j_max, alpha, rho)
            if logT_row_j_max - logF_i < log_epsilon: 
                break 
           
            logF_i = scan_row(r, row_j_max, logF_i, n, m, alpha, rho, log_epsilon)
            r += 1

        total_log_likelihood += logF_i

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
