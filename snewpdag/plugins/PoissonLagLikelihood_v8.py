"""
PoissonLagLikelihood_v8           : Calculate the Poisson likelihood of a histogram pair corresponding to a certain time lag.
                                    Expect to have FLOAT values in the histograms due to the smoothing. 
                                    Therefore, we used interpolation to estimate the values of lnJ(n,m), where n and m are floats.
                                    On top of v7, we will add a correction term for low count bins, in order to tackle the normalisation problem

configuration: 
    in_hist_field                 : field name of the histogram pair
    out_field                     : field name of the likelihood summary (lag vs log_likelihood)

    (Optional)
    sen_1                         : sensitivity of det1 (per second)
    sen_2                         : sensitivity of det2 (per second)
    bg_1                          : background rate of det1 (per second)
    bg_2                          : background rate of det2 (per second)
    cutoff                        : the minimum number of events that we can avoid the correction term
    c1                            : coefficient of the first-order inverse polynomial of det1
    c2                            : coefficient of the first-order inverse polynomial of det2
"""

import logging

import numpy as np
import scipy.special as sc
from scipy.interpolate import RegularGridInterpolator

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field
from snewpdag.plugins.RecursionIntegral import buildLogJTable

def lnJ_table_generator(nmax, mmax, a, b, p, q):
  return buildLogJTable(nmax, mmax, a, b, p, q)


class PoissonLagLikelihood_v8(Node):
  """Evaluate the likelihood for one histogram pair at one guessed lag."""

  def __init__(self, in_hist_field, out_field, **kwargs):
    self.in_hist_field = in_hist_field
    self.out_field = out_field
    self.sensitivity_1 = kwargs.pop('sen_1', 1.0)
    self.sensitivity_2 = kwargs.pop('sen_2', 1.0)
    self.background_rate_1 = kwargs.pop('bg_1', 0.0)
    self.background_rate_2 = kwargs.pop('bg_2', 0.0)

    # if the count of one detector is lower than this cutoff, we will add the correction term:
    self.cutoff = kwargs.pop('cutoff',3.0) 

    # coefficients for the first order inverse polynomial:
    self.c1 = kwargs.pop('c1', 0.00045)
    self.c2 = kwargs.pop('c2', 0.00045)
    self.lnJ_table_cache = {}
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
    
     
  def evaluate_pair(self, pair):
    bin_width = pair.get("bin_width", 0.002)

    # values are relative to bin-width:
    a = self.sensitivity_1 * bin_width
    p = self.sensitivity_2 * bin_width
    b = self.background_rate_1 * bin_width
    q = self.background_rate_2 * bin_width

    h1 = np.asarray(pair["hist1"], dtype=np.float64)
    h2 = np.asarray(pair["hist2"], dtype=np.float64)

    nmax = int(np.ceil(np.max(h1)))
    mmax = int(np.ceil(np.max(h2)))

    lnJ_table = self.retrieve_lnJ_table(nmax, mmax, a, b, p, q)

    # detector pair's first-order inverse polynomial coefficient:
    k1 = self.c1 + self.c2


    total_log_likelihood = 0.0
    total_lnJ = 0.0
    total_ln_fac = 0.0
    total_correction = 0.0

    for n, m in zip(h1, h2):
        # Base likelihood for this individual bin.
        log_J = self.table_interpolation_calculator(lnJ_table, n, m)

        log_fac = sc.gammaln(n + 1.0) + sc.gammaln(m + 1.0)

        log_correction = 0.0

        # Correct whenever either detector is in the low-count regime.
        apply_correction = n < self.cutoff or m < self.cutoff

        if apply_correction and k1 > 0.0:
            x = n + m - 1.0

            if x > -1.0:
                log_Q = sc.gammaln(x + 1.0) - (x + 1.0) * np.log(a + p)
                
                log_ratio = (
                    np.log(k1)
                    + sc.xlogy(n, a)
                    + sc.xlogy(m, p)
                    + log_Q
                    - log_J
                )

                log_correction = np.logaddexp(
                    0.0, log_ratio
                )

        total_lnJ += log_J
        total_ln_fac += log_fac
        total_correction += log_correction

        total_log_likelihood += (
            log_J
            - log_fac
            + log_correction
        )

    return {
        "lag": float(pair["lag"]),
        "log_likelihood": float(total_log_likelihood),
        "sum_lnJ": float(total_lnJ),
        "sum_ln_fac": float(total_ln_fac),
        "sum_correction": float(total_correction),
    }
  
  def alert(self, data):
    hist_data, valid = fetch_field(data, self.in_hist_field)

    if not valid:
        return False

    if "hist1" in hist_data and "hist2" in hist_data:
        pairs = [hist_data]
    elif "pairs" in hist_data and len(hist_data["pairs"]) > 0:
        pairs = hist_data["pairs"]
    else:
        logging.error(
            "%s: Cannot receive one or more histogram pairs",
            self.name,
        )
        return False

    results = []

    for pair in pairs:
        result = self.evaluate_pair(pair)
        results.append(result)

    # One pair: preserve the scalar output API.
    if len(results) == 1:
        result = results[0]

        if not np.isfinite(result["log_likelihood"]):
            return False

        store_field(data, self.out_field, result)
        return True

    # Multiple pairs: convert results into scan arrays.
    lags = []
    likelihoods = []
    sum_lnJ = []
    sum_ln_fac = []
    sum_correction = []

    for result in results:
        lags.append(result["lag"])

        if np.isfinite(result["log_likelihood"]):
            likelihoods.append(result["log_likelihood"])
            sum_lnJ.append(result["sum_lnJ"])
            sum_ln_fac.append(result["sum_ln_fac"])
            sum_correction.append(
                result["sum_correction"]
            )
        else:
            likelihoods.append(np.nan)
            sum_lnJ.append(np.nan)
            sum_ln_fac.append(np.nan)
            sum_correction.append(np.nan)

    output = {
        "possible_time_lag_list": np.asarray(
            lags, dtype=np.float64
        ),
        "log_likelihood_list": np.asarray(
            likelihoods, dtype=np.float64
        ),
        "sum_lnJ_list": np.asarray(
            sum_lnJ, dtype=np.float64
        ),
        "sum_ln_fac_list": np.asarray(
            sum_ln_fac, dtype=np.float64
        ),
        "sum_correction_list": np.asarray(
            sum_correction, dtype=np.float64
        ),
    }

    store_field(data, self.out_field, output)
    return True
  
