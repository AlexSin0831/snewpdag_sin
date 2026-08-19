"""Tests for converting likelihood scan summaries into DiffPointing input."""

import unittest

import numpy as np

from snewpdag.plugins.DiffPointing import DiffPointing
from snewpdag.plugins.LikelihoodScanToDts import LikelihoodScanToDts


def likelihood_scan(center=0.02, sigma=0.03):
  lags = np.asarray([-0.04, -0.02, 0.0, 0.02, 0.04], dtype=np.float64)
  likelihoods = -0.5 * ((lags - center) / sigma) ** 2
  return {
      'possible_time_lag_list': lags,
      'log_likelihood_list': likelihoods,
      'best_lag': center,
      'scan_type': 'fine',
  }


class TestLikelihoodScanToDts(unittest.TestCase):

  def test_polynomial_variance_is_finite(self):
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SK'), 1635744156.328,
        covariance_mode='diagonal',
        name='scan-to-dts',
    )
    lags = np.linspace(-0.05, 0.05, 11)
    sigma = 0.02
    likelihoods = -0.5 * (lags / sigma) ** 2

    variance, variance_method = node.variance_calculator(
        lags, likelihoods, 5,
    )

    self.assertAlmostEqual(variance, sigma ** 2)
    self.assertEqual(variance_method, 'curvature')

  def test_default_sign_and_curvature_variance(self):
    reference_time = 1635744156.328
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SK'), reference_time,
        name='scan-to-dts',
    )
    node.update({'action': 'alert', 'scan': likelihood_scan()})

    result = node.last_data['dts'][('IC', 'SK')]
    self.assertAlmostEqual(result['best_lag'], 0.02)
    self.assertAlmostEqual(result['dt'], -0.02)
    self.assertAlmostEqual(result['t1'], reference_time)
    self.assertAlmostEqual(result['t2'], reference_time)
    self.assertAlmostEqual(result['var'], 0.03 ** 2)
    self.assertAlmostEqual(result['rms'], 0.03)
    self.assertEqual(result['variance_method'], 'curvature')
    self.assertEqual(result['dsig1'], 0.0)
    self.assertEqual(result['dsig2'], 0.0)

  def test_reference_time_field_and_split_covariance(self):
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'JUNO'), ('meta', 'epoch'),
        covariance_mode='split',
        sigma_fudge=2.0,
        name='scan-to-dts',
    )
    node.update({
        'action': 'alert',
        'meta': {'epoch': 1635744156.328},
        'scan': likelihood_scan(),
    })

    result = node.last_data['dts'][('IC', 'JUNO')]
    self.assertAlmostEqual(result['var'], 4.0 * 0.03 ** 2)
    self.assertAlmostEqual(result['dsig1'], np.sqrt(result['var'] / 2.0))
    self.assertAlmostEqual(result['dsig2'], -np.sqrt(result['var'] / 2.0))

  def test_detector_db_covariance_preserves_polynomial_variance(self):
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SK'), 1635744156.328,
        covariance_mode='detector_db',
        detector_location='snewpdag/data/detector_location.csv',
        name='scan-to-dts',
    )

    fitted_variance = 0.03 ** 2
    dsig1, dsig2 = node.covariance_terms(fitted_variance)
    database_sigma = np.sqrt(0.001 ** 2 + 0.0009 ** 2)

    self.assertAlmostEqual(
        dsig1,
        np.sqrt(fitted_variance) * 0.001 / database_sigma,
    )
    self.assertAlmostEqual(
        dsig2,
        -np.sqrt(fitted_variance) * 0.0009 / database_sigma,
    )
    self.assertAlmostEqual(dsig1 ** 2 + dsig2 ** 2, fitted_variance)

  def test_detector_db_covariance_requires_detector_location(self):
    with self.assertRaises(ValueError):
      LikelihoodScanToDts(
          'scan', 'dts', ('IC', 'SK'), 1635744156.328,
          covariance_mode='detector_db',
          name='scan-to-dts',
      )

  def test_output_is_isolated_to_this_pair(self):
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SNOP'), 1635744156.328,
        name='scan-to-dts',
    )
    node.update({
        'action': 'alert',
        'scan': likelihood_scan(),
        'dts': {('IC', 'SK'): {'dt': 0.0}},
    })

    self.assertEqual(set(node.last_data['dts']), {('IC', 'SNOP')})

  def test_invalid_scan_does_not_forward(self):
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SK'), 1635744156.328,
        name='scan-to-dts',
    )
    node.update({
        'action': 'alert',
        'scan': {
            'possible_time_lag_list': [0.0, 0.1],
            'log_likelihood_list': [0.0],
        },
    })
    self.assertEqual(node.last_data, {})

  def test_revoke_identifies_the_pair_to_remove(self):
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SK'), 1635744156.328,
        name='scan-to-dts',
    )
    node.update({'action': 'revoke'})
    self.assertEqual(node.last_data['dts'], {('IC', 'SK'): {}})

  def test_lag_sign_can_be_overridden(self):
    node = LikelihoodScanToDts(
        'scan', 'dts', ('SK', 'IC'), 1635744156.328,
        lag_sign=1.0,
        name='scan-to-dts',
    )
    node.update({'action': 'alert', 'scan': likelihood_scan()})
    self.assertAlmostEqual(node.last_data['dts'][('SK', 'IC')]['dt'], 0.02)

  def test_out_key_must_be_two_distinct_detector_names(self):
    with self.assertRaises(TypeError):
      LikelihoodScanToDts(
          'scan', 'dts', 'IC', 1635744156.328,
          name='scan-to-dts',
      )
    with self.assertRaises(ValueError):
      LikelihoodScanToDts(
          'scan', 'dts', ('IC', 'IC'), 1635744156.328,
          name='scan-to-dts',
      )

  def test_nonfinite_reference_time_does_not_forward(self):
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SK'), np.nan,
        name='scan-to-dts',
    )
    node.update({'action': 'alert', 'scan': likelihood_scan()})
    self.assertEqual(node.last_data, {})

  def test_boundary_maximum_is_rejected_by_default(self):
    scan = {
        'possible_time_lag_list': [-0.1, 0.0, 0.1],
        'log_likelihood_list': [0.0, -1.0, -2.0],
    }
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SK'), 1635744156.328,
        name='scan-to-dts',
    )
    node.update({'action': 'alert', 'scan': scan})
    self.assertEqual(node.last_data, {})

  def test_boundary_maximum_can_use_grid_fallback(self):
    scan = {
        'possible_time_lag_list': [-0.1, 0.0, 0.1],
        'log_likelihood_list': [0.0, -1.0, -2.0],
    }
    node = LikelihoodScanToDts(
        'scan', 'dts', ('IC', 'SK'), 1635744156.328,
        allow_boundary=True,
        name='scan-to-dts',
    )
    node.update({'action': 'alert', 'scan': scan})
    result = node.last_data['dts'][('IC', 'SK')]
    self.assertEqual(result['variance_method'], 'grid')
    self.assertAlmostEqual(result['var'], 0.1 ** 2)

  def test_three_pair_nodes_converge_in_diff_pointing(self):
    point = DiffPointing(
        'snewpdag/data/detector_location.csv',
        nside=1,
        min_dts=3,
        name='point',
    )
    pairs = (
        ('IC', 'SK'),
        ('IC', 'SNOP'),
        ('IC', 'JUNO'),
    )
    nodes = [
        LikelihoodScanToDts(
            'scan', 'dts', pair, 1635744156.328,
            name='scan-to-dts-{}'.format(i),
        )
        for i, pair in enumerate(pairs)
    ]
    for node in nodes:
      node.attach(point)
      node.update({'action': 'alert', 'scan': likelihood_scan()})

    self.assertEqual(set(point.cache), set(pairs))
    self.assertIn('map', point.last_data)
    self.assertEqual(len(point.last_data['map']), 12)


if __name__ == '__main__':
  unittest.main()
