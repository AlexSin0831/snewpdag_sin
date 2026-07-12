'''
Implement "NEGLECT TERMS" + "CACHE GAMMA FUNCTIONS" + "FLEXIBLE SCANNING DIRECTION" method
'''

import logging
import numbers

import numpy as np
import scipy.special as sc

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field

# Useless in this method:
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
# log_factorial is an array, that saves ln(n!) values
def single_log_term_calculator_cached(n, r, m, j, log_alpha, log_rho, ln_factorial_cache):
  n = int(n)
  r = int(r)
  m = int(m)
  j = int(j)
  k = n - r + m - j

  single_log_term = (ln_factorial_cache[k]
                     - ln_factorial_cache[n - r]
                     - ln_factorial_cache[m - j]
                     + r * log_alpha
                     + j * log_rho
                     - ln_factorial_cache[r]
                     - ln_factorial_cache[j])
  return single_log_term

# FIND THE "GLOBAL MAXIMUM" OF THE MATRIX
def find_expansion_centre(n, m, alpha, rho, log_alpha, log_rho, log_factorial): 
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

    log_T1 = single_log_term_calculator_cached(
       n, r1_refined, m, j1_refined, log_alpha, log_rho, log_factorial
    )
    log_T2 = single_log_term_calculator_cached(
       n, r2_refined, m, j2_refined, log_alpha, log_rho, log_factorial
    )

    # DETERMINE THE EXPANSION CENTRE
    if log_T1 > log_T2: 
        r_max, j_max = r1_refined, j1_refined
    else: 
        r_max, j_max = r2_refined, j2_refined
    
    return int(r_max), int(j_max)


def find_row_j_max(r_fixed, n, m, alpha, rho, log_alpha, log_rho, log_factorial):
    j_max1, j_max2 = quadratic_solver(1, 
                                    1 - m + r_fixed - n - rho, 
                                    (rho - 1) * m + r_fixed -n)
    
    j_max1_refined = np.ceil(min(max(j_max1,0), m))
    j_max2_refined = np.ceil(min(max(j_max2,0), m))

    T_1 = single_log_term_calculator_cached(n, r_fixed, m, j_max1_refined, log_alpha, log_rho, log_factorial)
    T_2 = single_log_term_calculator_cached(n, r_fixed, m, j_max2_refined, log_alpha, log_rho, log_factorial)

    if T_1 > T_2: 
       return int(j_max1_refined)
    else: 
       return int(j_max2_refined)
    
def scan_row(r_fixed, j_max, log_sum, n, m, log_alpha, log_rho, log_factorial, log_epsilon):
    log_sum = np.logaddexp(log_sum, single_log_term_calculator_cached(n, r_fixed, m, j_max, log_alpha, log_rho, log_factorial))
    
    # TO LEFT:
    j = j_max -1
    while j >= 0: 
        logT_rj = single_log_term_calculator_cached(n, r_fixed, m, j, log_alpha, log_rho, log_factorial)
        if logT_rj - log_sum < log_epsilon: 
           break
        else: 
           log_sum = np.logaddexp(log_sum, logT_rj)
           j -= 1
    
    # TO RIGHT: 
    j = j_max + 1
    while j <= m:
        logT_rj = single_log_term_calculator_cached(n, r_fixed, m, j, log_alpha, log_rho, log_factorial)
        if logT_rj - log_sum < log_epsilon:
           break
        else: 
           log_sum = np.logaddexp(log_sum, logT_rj)
           j += 1
    
    return log_sum
       


class PoissonLagLikelihood_v3(Node):
  """Evaluate the likelihood for one histogram pair at one guessed lag."""

  def __init__(self, in_hist_field, out_field, **kwargs):
    self.in_hist_field = in_hist_field
    self.out_field = out_field
    self.sensitivity_1 = kwargs.pop('sensitivity_1', kwargs.pop('a', 1.0))
    self.sensitivity_2 = kwargs.pop('sensitivity_2', kwargs.pop('p', 1.0))
    self.background_rate_1 = kwargs.pop('background_rate_1', kwargs.pop('bg_1', 0.0))
    self.background_rate_2 = kwargs.pop('background_rate_2', kwargs.pop('bg_2', 0.0))
    self.ln_factorial_cache = np.asarray([0.0], dtype = np.float64)
    self.ln_factorial_cache.flags.writeable = False
    self.error = kwargs.pop('error', 0.005)
    super().__init__(**kwargs)


  def ln_factorial_array_generator(self, max_count): 
      if self.ln_factorial_cache.size > max_count:
        return self.ln_factorial_cache

      array = sc.gammaln(np.arange(max_count+1, dtype=np.float64) + 1.0)
      array.flags.writeable = False
      self.ln_factorial_cache = array

      return array
     
  # This MOST IMPORTANT function in this plugin: 
  def pair_total_log_likelihood(self, pair, ln_factorial_array):
    bin_width = pair.get('bin_width', 1.0)

    # Caution: b and q are defined as the background rate per bin!!!
    b = self.background_rate_1 * bin_width 
    q = self.background_rate_2 * bin_width

    # Pre-calculated alpha and rho (normal case / swapped case)
    alpha_1 = b * (1 + self.sensitivity_2 / self.sensitivity_1)
    rho_1 = q * (1 + self.sensitivity_1 / self.sensitivity_2)
    alpha_2 = rho_1
    rho_2 = alpha_1

    log_alpha_1 = np.log(alpha_1) if alpha_1 > 0.0 else -np.inf
    log_rho_1 = np.log(rho_1) if rho_1 > 0.0 else -np.inf
    log_alpha_2 = np.log(alpha_2) if alpha_2 > 0.0 else -np.inf
    log_rho_2 = np.log(rho_2) if rho_2 > 0.0 else -np.inf


    h1 = np.asarray(pair['hist1'], dtype=np.int64)
    h2 = np.asarray(pair['hist2'], dtype=np.int64)

    total_log_likelihood = 0 

    # F_i = \sum_{r=0}^{n_i} \sum_{j=0}^m_i \binom{n_i - r + m_i - j}{n_i - r} \frac{\alpha^r \rho^j}{r! j!}
    # ln(L_total) = ln(F_1) + ln(F_2) + ... + ln(F_B)

    for n, m in zip(h1, h2):
        # We choose scan in the direction that with fewer count to speed up our calculation
        if n > m: 
          n, m = m, n 
          alpha, rho = alpha_2, rho_2
          log_alpha, log_rho = log_alpha_2, log_rho_2
        else: 
          alpha, rho = alpha_1, rho_1
          log_alpha, log_rho = log_alpha_1, log_rho_1

        r_max, j_max = find_expansion_centre(n, m, alpha, rho, log_alpha, log_rho, ln_factorial_array) # int
        
        # Stop-scanning threshold:
        ln_epsilon = np.log(self.error / ((n+1) * (m+1)))

        # F_i == 0 initially ==> ln(F_i) --> -inf
        lnF_i = -np.inf

        # Scan the row of the expansion centre first
        lnF_i = scan_row(r_max, j_max, 
                          lnF_i, 
                          n, m,
                          log_alpha, log_rho, ln_factorial_array, ln_epsilon)


        # Scan the whole matrix: 
        # UPWARD:
        r = int(r_max) - 1
        while r >= 0: 
            row_j_max = find_row_j_max(r, n, m, alpha, rho, log_alpha, log_rho, ln_factorial_array) #int

            lnT_row_j_max = single_log_term_calculator_cached(n, r, m, row_j_max, log_alpha, log_rho, ln_factorial_array)

            if lnT_row_j_max - lnF_i < ln_epsilon: 
                break 
           
            lnF_i = scan_row(r, row_j_max, lnF_i, n, m,log_alpha, log_rho, ln_factorial_array, ln_epsilon)
            r -= 1

        # DOWNWARD: 
        r = int(r_max) + 1
        while r <= n: 
            row_j_max = find_row_j_max(
               r, n, m, alpha, rho, log_alpha, log_rho, ln_factorial_array
            ) #int
            lnT_row_j_max = single_log_term_calculator_cached(n, r, m, row_j_max, log_alpha, log_rho, ln_factorial_array)

            if lnT_row_j_max - lnF_i < ln_epsilon: 
                break 
           
            lnF_i = scan_row(r, row_j_max, lnF_i, n, m,log_alpha, log_rho, ln_factorial_array, ln_epsilon)
            r += 1

        total_log_likelihood += lnF_i

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

    h1 = pair['hist1']
    h2 = pair['hist2']
    h1 = np.asarray(h1, dtype=np.int64)
    h2 = np.asarray(h2, dtype=np.int64)
    max_factorial_argument = int(np.max(h1 + h2))
    ln_factorial_array = self.ln_factorial_array_generator(max_factorial_argument)

    total_log_likelihood = self.pair_total_log_likelihood(pair, ln_factorial_array)


    if not np.isfinite(total_log_likelihood):
      return False
    result = {
        'lag': float(pair['lag']),
        'log_likelihood': float(total_log_likelihood),
    }
    store_field(data, self.out_field, result)
    return True
