"""Tests for single-lag and multiple-lag Poisson likelihood payloads."""

import importlib
import unittest

import numpy as np


likelihood_module = importlib.import_module(
    'snewpdag.plugins.PoissonLagLikelihood_v6')
PoissonLagLikelihood_v6 = likelihood_module.PoissonLagLikelihood_v6
collector_module = importlib.import_module(
    'snewpdag.plugins.LikelihoodScanCollector')
LikelihoodScanCollector = collector_module.LikelihoodScanCollector


def histogram_pair(lag, hist1, hist2):
  return {
      'lag': lag,
      'hist1': np.asarray(hist1, dtype=np.float64),
      'hist2': np.asarray(hist2, dtype=np.float64),
      'bin_width': 0.1,
  }


class TestPoissonLagLikelihoodV6(unittest.TestCase):

  def test_single_pair_keeps_single_lag_output(self):
    pair = histogram_pair(0.1, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])
    node = PoissonLagLikelihood_v6(
        'hist_pair', 'likelihood', name='likelihood')

    node.update({
        'action': 'alert',
        'hist_pair': {
            'pairs': [pair],
            **pair,
        },
    })

    result = node.last_data['likelihood']
    self.assertEqual(result['lag'], 0.1)
    self.assertTrue(np.isfinite(result['log_likelihood']))
    self.assertNotIn('possible_time_lag_list', result)

  def test_multiple_pairs_return_scan_arrays(self):
    pairs = [
        histogram_pair(-0.1, [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]),
        histogram_pair(0.0, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0]),
        histogram_pair(0.1, [1.0, 2.0, 1.0], [2.0, 1.0, 0.0]),
    ]
    node = PoissonLagLikelihood_v6(
        'hist_pair', 'likelihood', name='likelihood')

    node.update({
        'action': 'alert',
        'hist_pair': {
            'pairs': pairs,
            'possible_time_lag_list': np.array([-0.1, 0.0, 0.1]),
        },
    })

    result = node.last_data['likelihood']
    np.testing.assert_allclose(
        result['possible_time_lag_list'], [-0.1, 0.0, 0.1])
    self.assertEqual(result['log_likelihood_list'].shape, (3,))
    self.assertTrue(np.all(np.isfinite(result['log_likelihood_list'])))

    single_results = []
    for pair in pairs:
      single_node = PoissonLagLikelihood_v6(
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
        histogram_pair(-0.1, [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]),
        histogram_pair(0.0, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0]),
        histogram_pair(0.1, [1.0, 2.0, 1.0], [2.0, 1.0, 0.0]),
    ]
    node = PoissonLagLikelihood_v6(
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
        histogram_pair(-0.1, [1.0], [1.0]),
        histogram_pair(0.1, [1.0], [1.0]),
    ]
    node = PoissonLagLikelihood_v6(
        'hist_pair', 'likelihood', sen_1=0.0, sen_2=0.0,
        name='likelihood')

    node.update({'action': 'alert', 'hist_pair': {'pairs': pairs}})

    result = node.last_data['likelihood']
    np.testing.assert_allclose(
        result['possible_time_lag_list'], [-0.1, 0.1])
    self.assertTrue(np.all(np.isnan(result['log_likelihood_list'])))


if __name__ == '__main__':
  unittest.main()
