"""Tests for single-lag and multiple-lag Poisson likelihood versions."""

import unittest

import numpy as np

from snewpdag.plugins import (
    LikelihoodScanCollector,
    PoissonLagLikelihood_v4,
    PoissonLagLikelihood_v5,
    PoissonLagLikelihood_v7,
)


LIKELIHOOD_VERSIONS = (
    PoissonLagLikelihood_v4,
    PoissonLagLikelihood_v5,
    PoissonLagLikelihood_v7,
)


def histogram_pair(lag, hist1, hist2):
  return {
      'lag': lag,
      'hist1': np.asarray(hist1),
      'hist2': np.asarray(hist2),
      'bin_width': 0.1,
  }


class TestPoissonLagLikelihoodVersions(unittest.TestCase):

  def test_direct_and_one_item_pair_inputs_match(self):
    pair = histogram_pair(0.1, [1, 2, 1], [1, 2, 1])

    for likelihood_class in LIKELIHOOD_VERSIONS:
      with self.subTest(version=likelihood_class.__name__):
        direct_node = likelihood_class(
            'hist_pair', 'likelihood', name='direct-likelihood')
        direct_node.update({'action': 'alert', 'hist_pair': pair})

        wrapped_node = likelihood_class(
            'hist_pair', 'likelihood', name='wrapped-likelihood')
        wrapped_node.update({
            'action': 'alert',
            'hist_pair': {'pairs': [pair]},
        })

        direct_result = direct_node.last_data['likelihood']
        wrapped_result = wrapped_node.last_data['likelihood']
        self.assertEqual(direct_result['lag'], 0.1)
        self.assertEqual(wrapped_result['lag'], 0.1)
        self.assertAlmostEqual(
            direct_result['log_likelihood'],
            wrapped_result['log_likelihood'])

  def test_multiple_pairs_return_scan_arrays(self):
    pairs = [
        histogram_pair(-0.1, [1, 2, 1], [0, 1, 2]),
        histogram_pair(0.0, [1, 2, 1], [1, 2, 1]),
        histogram_pair(0.1, [1, 2, 1], [2, 1, 0]),
    ]

    for likelihood_class in LIKELIHOOD_VERSIONS:
      with self.subTest(version=likelihood_class.__name__):
        node = likelihood_class(
            'hist_pair', 'likelihood', name='likelihood')
        node.update({'action': 'alert', 'hist_pair': {'pairs': pairs}})

        result = node.last_data['likelihood']
        np.testing.assert_allclose(
            result['possible_time_lag_list'], [-0.1, 0.0, 0.1])
        self.assertEqual(result['log_likelihood_list'].shape, (3,))
        self.assertTrue(np.all(np.isfinite(result['log_likelihood_list'])))

        single_results = []
        for pair in pairs:
          single_node = likelihood_class(
              'hist_pair', 'likelihood', name='single-likelihood')
          single_node.update({
              'action': 'alert',
              'hist_pair': {'pairs': [pair]},
          })
          single_results.append(
              single_node.last_data['likelihood']['log_likelihood'])
        np.testing.assert_allclose(
            result['log_likelihood_list'], single_results)

  def test_multiple_pair_output_is_accepted_by_scan_collector(self):
    pairs = [
        histogram_pair(-0.1, [1, 2, 1], [0, 1, 2]),
        histogram_pair(0.0, [1, 2, 1], [1, 2, 1]),
        histogram_pair(0.1, [1, 2, 1], [2, 1, 0]),
    ]

    for likelihood_class in LIKELIHOOD_VERSIONS:
      with self.subTest(version=likelihood_class.__name__):
        node = likelihood_class(
            'hist_pair', 'likelihood', name='likelihood')
        node.update({'action': 'alert', 'hist_pair': {'pairs': pairs}})

        collector = LikelihoodScanCollector(
            'likelihood', 'scan', scan_type='coarse', name='collector')
        collector.update(node.last_data)

        scan = collector.last_data['scan']
        self.assertIn(scan['best_lag'], [-0.1, 0.0, 0.1])
        self.assertEqual(len(scan['possible_time_lag_list']), 3)
        self.assertEqual(len(scan['log_likelihood_list']), 3)

  def test_invalid_likelihood_preserves_lag_coordinate(self):
    pairs = [
        histogram_pair(-0.1, [1], [1]),
        histogram_pair(0.1, [1], [1]),
    ]

    for likelihood_class in LIKELIHOOD_VERSIONS:
      with self.subTest(version=likelihood_class.__name__):
        node = likelihood_class(
            'hist_pair', 'likelihood', sen_1=0.0, sen_2=0.0,
            name='likelihood')
        node.update({'action': 'alert', 'hist_pair': {'pairs': pairs}})

        result = node.last_data['likelihood']
        np.testing.assert_allclose(
            result['possible_time_lag_list'], [-0.1, 0.1])
        self.assertTrue(np.all(np.isnan(result['log_likelihood_list'])))


if __name__ == '__main__':
  unittest.main()
