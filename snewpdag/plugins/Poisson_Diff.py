import logging
import numbers

import numpy as np
import scipy.special as sc

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field


class PoissonLightcurveDiff(Node):
  def __init__(self, in_series1_field, in_series2_field, bg_1, bg_2, out_field, **kwargs):
    self.in_series1_field = in_series1_field
    self.in_series2_field = in_series2_field
    self.bg_1 = bg_1
    self.bg_2 = bg_2
    self.out_field = out_field

    self.out_key = kwargs.pop('out_key', ('D1', 'D2'))
    self.nbins = kwargs.pop('nbins', 200)
    self.window = kwargs.pop('window', 1.0)
    self.lead_time = kwargs.pop('lead_time', -0.1)
    self.scan_low = kwargs.pop('scan_low', -0.042)
    self.scan_high = kwargs.pop('scan_high', 0.042)
    self.coarse_scan_steps = kwargs.pop('coarse_scan_steps', 20)
    self.fine_scan_steps = kwargs.pop('fine_scan_steps', 400)
    self.covariance_mode = kwargs.pop('covariance_mode', 'diagonal')
    self.alpha = kwargs.pop('alpha',None)

    self.true_lag = kwargs.pop('true_lag', 0.0)
    self.true_t1 = kwargs.pop('true_t1', 0.0)
    self.true_t2 = kwargs.pop('true_t2', 0.0)
    self.min_variance = kwargs.pop('min_variance', 1.0e-8)
    self.sigma_fudge = kwargs.pop('sigma_fudge', 1.0)
    super().__init__(**kwargs)

  def log_likelihood(self, ts1, ts2, dt):
    if len(ts1.times) == 0 or len(ts2.times) == 0:
      return -np.inf

    start = np.min(ts1.times) + self.lead_time
    h1, _ = ts1.histogram(self.nbins, start, start + self.window)
    h2, _ = ts2.histogram(self.nbins, start - dt, start - dt + self.window)

    # Signal-only version of the marginalized Poisson lightcurve likelihood.
    # The -log(n!) term is constant in dt for h1, but keeping it makes the
    # expression symmetric and easier to compare with the paper.
    return np.sum(
        sc.gammaln(h1 + h2 + 1.0)
        - sc.gammaln(h1 + 1.0)
        - sc.gammaln(h2 + 1.0)
    )

  def scan_profile(self, ts1, ts2):
    dt = np.arange(self.scan_low, self.scan_high + 0.5*self.scan_step,
                   self.scan_step)
    y = np.array([self.log_likelihood(ts1, ts2, x) for x in dt])
    if len(y) == 0 or not np.any(np.isfinite(y)):
      return None, None, None, None

    imax = int(np.nanargmax(y))
    best = dt[imax]
    variance = None

    if 0 < imax < len(dt) - 1:
      xs = dt[imax-1:imax+2]
      ys = y[imax-1:imax+2]
      try:
        a, b, _ = np.polyfit(xs, ys, 2)
        if a < 0.0:
          x0 = -b / (2.0*a)
          if xs[0] <= x0 <= xs[-1]:
            best = x0
          curvature = 2.0*a
          variance = -1.0 / curvature
      except Exception:
        logging.debug('%s: quadratic lag fit failed', self.name)

    if variance is None or not np.isfinite(variance) or variance <= 0.0:
      variance = self.profile_width(dt, y, imax)
    if variance is None or not np.isfinite(variance) or variance <= 0.0:
      variance = self.min_variance

    variance = max(float(variance), self.min_variance)
    return best, variance, dt, y

  def profile_width(self, dt, y, imax):
    y0 = y[imax]
    target = y0 - 0.5
    left = None
    right = None

    for i in range(imax, 0, -1):
      if y[i-1] <= target <= y[i]:
        left = np.interp(target, [y[i-1], y[i]], [dt[i-1], dt[i]])
        break
    for i in range(imax, len(dt)-1):
      if y[i+1] <= target <= y[i]:
        right = np.interp(target, [y[i+1], y[i]], [dt[i+1], dt[i]])
        break

    if left is not None and right is not None and right > left:
      return ((right - left) / 2.0) ** 2
    return None

  def covariance_terms(self, variance):
    sigma = np.sqrt(variance)
    if self.covariance_mode == 'split':
      s = sigma / np.sqrt(2.0)
      return s, -s
    if self.covariance_mode == 'first':
      return sigma, 0.0
    if self.covariance_mode == 'second':
      return 0.0, -sigma
    return 0.0, 0.0

  def fetch_optional_time(self, data, field):
    if field == 0.0:
      return 0.0
    if isinstance(field, numbers.Number):
      return field
    value, valid = fetch_field(data, field)
    return value if valid else 0.0

  def alert(self, data):
    ts1, valid = fetch_field(data, self.in_series1_field)
    if not valid:
      return False
    ts2, valid = fetch_field(data, self.in_series2_field)
    if not valid:
      return False

    best, variance, prof_x, prof_y = self.scan_profile(ts1, ts2)
    if best is None:
      return False

    variance *= self.sigma_fudge * self.sigma_fudge
    rms = np.sqrt(variance)
    dsig1, dsig2 = self.covariance_terms(variance)

    true_t1 = self.fetch_optional_time(data, self.true_t1)
    true_t2 = self.fetch_optional_time(data, self.true_t2)
    true_dt12 = true_t1 - true_t2
    dt_true = best - self.true_lag - true_dt12
    pull = dt_true / rms if rms > 0.0 else 0.0

    t1 = np.min(ts1.times)
    t2 = t1 - best

    dts, exists = fetch_field(data, self.out_field)
    dts = dts.copy() if exists else {}
    dts[self.out_key] = {
        'delta': best,
        'exp_delta': 0.0,
        'dt': best,
        'dt_true': dt_true,
        'dtf_true': dt_true,
        'pull': pull,
        'pull_fudge': pull,
        'rms': rms,
        'rms_fudge': rms,
        'bias': 0.0,
        'var': variance,
        'dsig1': dsig1,
        'dsig2': dsig2,
        't1': t1,
        't2': t2,
        'profile_x': prof_x,
        'profile_y': prof_y,
    }
    store_field(data, self.out_field, dts)
    return True
