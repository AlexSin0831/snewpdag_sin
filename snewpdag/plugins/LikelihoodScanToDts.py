"""
LikelihoodScanToDts: Convert a likelihood scan summary into one entry of the
                     dts payload used by DiffPointing.

configuration:
  input_field                    : field name of the results from LikelihoodScanCollector
  out_field                      : field name of the dts payload 
  out_key                        : detector pair, ordered as (det1, det2)
  reference_time                 : approximate burst time in Unix seconds,
                                   or a payload field containing that value.
                                   This common time is stored as both t1 and
                                   t2 for DiffPointing's Earth orientation.

  (Optional)
  lag_sign                       : multiply best_lag by this value to obtain
                                   dt = t1 - t2. Default is -1.0 because the
                                   pair nodes define lag as t2 - t1.
  sigma_fudge                    : scale the fitted standard deviation
  min_variance                   : lower bound for the output variance
  variance_mode                  : plot / detector_db
  covariance_mode                : diagonal / detector_db 
  detector_location              : detector database filename. Required when
                                   covariance_mode is detector_db.
  bias                           : timing bias in dt, as a number or payload field. Default is 0.0.
  allow_boundary                 : allow a best lag on the edge of the scan.
                                   Default is False.
"""

import logging
import numbers

import numpy as np

from snewpdag.dag import Node, DetectorDB
from snewpdag.dag.lib import fetch_field, store_field


class LikelihoodScanToDts(Node):
  """Build one detector-pair entry for DiffPointing."""

  def __init__(self, input_field, out_field, out_key, reference_time, **kwargs):
    self.input_field = input_field
    self.out_field = out_field
    if not isinstance(out_key, (list, tuple)):
      raise TypeError('out_key must be a list or tuple of detector names')
    self.out_key = tuple(out_key)
    self.reference_time = reference_time
    self.lag_sign = float(kwargs.pop('lag_sign', -1.0))
    self.sigma_fudge = float(kwargs.pop('sigma_fudge', 1.0))
    self.min_variance = float(kwargs.pop('min_variance', 1.0e-6))
    self.variance_mode = kwargs.pop('variance_mode', 'detector_db')
    self.covariance_mode = kwargs.pop('covariance_mode', 'detector_db')
    self.detector_location = kwargs.pop('detector_location', None)
    self.bias = kwargs.pop('bias', 0.0)
    self.allow_boundary = kwargs.pop('allow_boundary', False)

    if self.covariance_mode not in ('diagonal', 'detector_db', 'split', 'first', 'second'):
      raise ValueError(
          'covariance_mode must be diagonal, detector_db, split, first, or second')

    self.db = None
    if self.covariance_mode == 'detector_db':
      if self.detector_location is None:
        raise ValueError(
            'detector_location is required when covariance_mode is detector_db')
      self.db = DetectorDB(self.detector_location)

    super().__init__(**kwargs)

  # reference time / bias:
  def retrieve_value(self, data, value_or_field):
    if isinstance(value_or_field, numbers.Number):
      value = float(value_or_field)
      return (value, True) if np.isfinite(value) else (None, False)
    value, valid = fetch_field(data, value_or_field)
    if not valid:
      return None, False
    value = float(value)
    return (value, True) if np.isfinite(value) else (None, False)

  # results from LikelihoodScanCollector
  def retrieve_scan_results(self, data):
    results, valid = fetch_field(data, self.input_field)
    if not valid or not isinstance(results, dict):
      return None, None
    if ('possible_time_lag_list' not in results or 'log_likelihood_list' not in results):
      return None, None

    # ensure good format:
    lags = np.asarray(results['possible_time_lag_list'], dtype=np.float64).reshape(-1)
    likelihoods = np.asarray(results['log_likelihood_list'], dtype=np.float64).reshape(-1)

    # 1st check: 
    if len(lags) != len(likelihoods) or len(lags) == 0:
      raise ValueError('Invalid data before cleaning')
      return None, None

    # Data cleaning: 
    valid = np.isfinite(lags) & np.isfinite(likelihoods)
    lags = lags[valid]
    likelihoods = likelihoods[valid]

    # 2nd check:
    if len(lags) == 0:
      raise ValueError('Invalid data after cleaning')
      return None, None

    order = np.argsort(lags)
    return lags[order], likelihoods[order]

  # "0.5 method" to determine the sigma:
  def log_half_method(self, lags, likelihoods, index_max):
    target = likelihoods[index_max] - 0.5
    left = None
    right = None

    for i in range(index_max, 0, -1):
      if likelihoods[i - 1] <= target <= likelihoods[i]:
        left = np.interp(target,
                        [likelihoods[i - 1], likelihoods[i]],
                        [lags[i - 1], lags[i]])
        break

    for i in range(index_max, len(lags) - 1):
      if likelihoods[i + 1] <= target <= likelihoods[i]:
        right = np.interp(target,
                        [likelihoods[i + 1], likelihoods[i]],
                        [lags[i + 1], lags[i]])
        break

    if left is not None and right is not None and right > left:
      return ((right - left) / 2.0) ** 2
    return None

  # calculate the variance of the time lag
  def variance_calculator(self, lags, likelihoods, index_max):
    if self.variance_mode == 'plot':
      if 0 < index_max < len(lags) - 1:
        low = max(0,index_max-10)
        high = min(len(lags), index_max +11) 
        x = lags[low:high]
        y = likelihoods[low:high]
        if len(np.unique(x)) == 21:
          try:
            fit_curve = np.polynomial.Polynomial.fit(x,y,4)
            best_lag = lags[index_max]
            second_derivative = fit_curve.deriv(2)(best_lag)
            if second_derivative < 0.0:
              standard_error = (-second_derivative) ** (-0.5)
              if np.isfinite(standard_error) and standard_error > 0.0:
                variance = standard_error ** 2 
                return float(variance), 'curvature'
          except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            logging.debug('%s: local likelihood fit failed', self.name)

      variance = self.log_half_method(lags, likelihoods, index_max)
      if variance is not None and np.isfinite(variance) and variance > 0.0:
        return float(variance), 'delta_log_likelihood'

      if len(lags) > 1:
        spacing = np.diff(np.unique(lags))
        spacing = spacing[spacing > 0.0]
        if len(spacing) > 0:
          return max(float(np.min(spacing) ** 2), self.min_variance), 'grid'
    elif self.variance_mode == 'detector_db': 
      det1 = self.db.get(self.out_key[0])
      det2 = self.db.get(self.out_key[1])

      if det1 is None or det2 is None:
        raise ValueError('Cannot find one or more detectors in detector_location')
      
      return det1.sigma**2 + det2.sigma**2, 'detector_db'
    return self.min_variance, 'minimum'

  # necessary for building the weighted matrix in DiffPointing 
  def covariance_terms(self, variance):
    sigma = np.sqrt(variance)
    
    # Fake / Testing case:
    if self.covariance_mode == 'diagonal':
      # Treat detector-pair lag estimates as independent.
      # DiffPointing will therefore set all off-diagonal covariance-matrix elements to zero.
      return 0.0, 0.0

    # We are still using our polynomial fitted variance as the total variance.
    # The detector database is only used to determine the relative contribution
    # from detector 1 and detector 2.
    if self.covariance_mode == 'detector_db':
      det1 = self.db.get(self.out_key[0])
      det2 = self.db.get(self.out_key[1])

      if det1 is None or det2 is None:
        raise ValueError('Cannot find one or more detectors in detector_location')

      database_sigma = np.sqrt(det1.sigma ** 2 + det2.sigma ** 2)
      if database_sigma <= 0.0:
        raise ValueError('Invalid detector sigma from detector_location')

      # fractional contribution: 
      dsig1 = sigma * det1.sigma / database_sigma
      dsig2 = -sigma * det2.sigma / database_sigma
      return dsig1, dsig2

  def alert(self, data):
    lags, likelihoods = self.retrieve_scan_results(data)
    if lags is None:
      return False

    reference_time, valid = self.retrieve_value(data, self.reference_time)
    if not valid:
      return False
    
    bias, valid = self.retrieve_value(data, self.bias)
    if not valid:
      return False

    index_max = int(np.nanargmax(likelihoods))
    if (index_max == 0 or index_max == len(lags) - 1) and not self.allow_boundary:
      logging.warning('%s: likelihood maximum is on the scan boundary', self.name)
      return False

    best_lag = float(lags[index_max])
    variance, variance_method = self.variance_calculator(lags, likelihoods, index_max)
    variance = max(variance * self.sigma_fudge * self.sigma_fudge, self.min_variance)

    # DiffPointing defines dt = t1 - t2, so we need to add a minus sign in front of our best_lag
    dt = self.lag_sign * best_lag

    # Originally, when we were still using the first-event method, we compute dt by finding the 
    # first-event time of different detectors. 
    # Therefore, we will have t1 and t2 naturally
    # However, we will no longer calculate t1 and t2 in our new likelihood method, while DiffPointing
    # requires t1 and t2 for determing detectors' geographical locations
    # Notice O(dt) << O(t_rotation), so we just do an approximation: t1 = t2 = epoch_base (during our analysis)
    # However, in reality, maybe we can still use the first-event method to find t1 and t2. 
    # Only for the sake of determing the locations, but not the dt.

    t1 = reference_time
    t2 = reference_time
    dsig1, dsig2 = self.covariance_terms(variance)

    dts = {self.out_key: {'dt': dt,
                          't1': t1,
                          't2': t2,
                          'bias': bias,
                          'var': variance,
                          'dsig1': dsig1,
                          'dsig2': dsig2,
                          'rms': np.sqrt(variance),
                          'best_lag': best_lag,
                          'variance_method': variance_method}}
    
    store_field(data, self.out_field, dts)
    return True

  def revoke(self, data):
    store_field(data, self.out_field, {self.out_key: {}})
    return True
