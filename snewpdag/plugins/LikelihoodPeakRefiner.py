import logging
import numbers

import numpy as np
import scipy.special as sc

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field

class LikelihoodPeakRefiner(Node):
  """Refine the sampled maximum with a local quadratic interpolation."""

  def __init__(self, in_scan_field, out_field, **kwargs):
    self.in_scan_field = in_scan_field
    self.out_field = out_field
    self.fit_points = kwargs.pop('fit_points', 3)
    super().__init__(**kwargs)

  def alert(self, data):
    scan, valid = fetch_field(data, self.in_scan_field)
    if not valid:
      return False
    possible_time_lag_list = np.asarray(
        scan.get('possible_time_lag_list'),
        dtype=np.float64)
    log_likelihood_list = np.asarray(scan.get('log_likelihood_list'),
                                     dtype=np.float64)
    if (len(possible_time_lag_list) == 0
        or not np.any(np.isfinite(log_likelihood_list))):
      return False

    imax = int(np.nanargmax(log_likelihood_list))
    refined_lag = float(possible_time_lag_list[imax])
    refined_log_likelihood = float(log_likelihood_list[imax])
    curvature = None
    variance = None

    half = max(1, int(self.fit_points) // 2)
    i0 = max(0, imax - half)
    i1 = min(len(possible_time_lag_list), imax + half + 1)
    if i1 - i0 >= 3:
      try:
        a, b, c = np.polyfit(possible_time_lag_list[i0:i1],
                             log_likelihood_list[i0:i1], 2)
        if a < 0.0:
          x0 = -b / (2.0 * a)
          if possible_time_lag_list[i0] <= x0 <= possible_time_lag_list[i1 - 1]:
            refined_lag = float(x0)
            refined_log_likelihood = float(a * x0 * x0 + b * x0 + c)
          curvature = float(2.0 * a)
          variance = float(-1.0 / curvature)
      except Exception:
        logging.debug('%s: quadratic likelihood interpolation failed',
                      self.name)

    result = scan.copy()
    result.update({
        'refined_lag': refined_lag,
        'refined_log_likelihood': refined_log_likelihood,
        'curvature': curvature,
        'variance': variance,
    })
    store_field(data, self.out_field, result)
    return True
