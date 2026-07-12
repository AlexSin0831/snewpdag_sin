"""
Unit tests for the composable Poisson lightcurve lag nodes.
"""
import unittest
import importlib.util
from pathlib import Path
import numpy as np

from snewpdag.dag import Node
from snewpdag.values import TimeSeries

def _load_plugin(filename):
  module_path = Path(__file__).parents[1] / 'plugins' / filename
  spec_name = filename[:-3].replace('/', '_')
  spec = importlib.util.spec_from_file_location(spec_name, module_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

def _run_node(node, data):
  node.update(data)
  return node.last_data

NewTimeSeries = _load_plugin('ops/NewTimeSeries.py').NewTimeSeries
GaussianPeak = _load_plugin('gen/GaussianPeak.py').GaussianPeak
PairTimeSeriesToHist = _load_plugin('PairTimeSeriesToHist.py').PairTimeSeriesToHist
TimeLagGenerator = _load_plugin('TimeLagGenerator.py').TimeLagGenerator
PoissonLagLikelihood = _load_plugin('PoissonLagLikelihood.py').PoissonLagLikelihood
LikelihoodScanCollector = _load_plugin(
    'LikelihoodScanCollector.py').LikelihoodScanCollector
LikelihoodPeakRefiner = _load_plugin(
    'LikelihoodPeakRefiner.py').LikelihoodPeakRefiner


class TestPoissonLagNodes(unittest.TestCase):

  def test_lag_generator(self):
    n = TimeLagGenerator('possible_time_lag_list', scan_low=-0.02,
                         scan_high=0.02, coarse_step=0.01,
                         name='possible-time-lag-list')
    n.update({'action': 'alert'})
    np.testing.assert_allclose(n.last_data['possible_time_lag_list'],
                               [-0.02, -0.01, 0.0, 0.01, 0.02])

  def test_pair_time_series_to_hist_for_possible_time_lag_list(self):
    ts1 = TimeSeries()
    ts1.add(np.array([0.01, 0.11, 0.21]))
    ts2 = TimeSeries()
    ts2.add(np.array([0.11, 0.21, 0.31]))

    n = PairTimeSeriesToHist('ts1', 'ts2', 'pairs', 0.1, start=0.0,
                             window=0.3,
                             possible_time_lag_list_field=(
                                 'possible_time_lag_list'),
                             name='pair')
    n.update({'action': 'alert', 'ts1': ts1, 'ts2': ts2,
              'possible_time_lag_list': np.array([0.0, 0.1])})

    pairs = n.last_data['pairs']['pairs']
    self.assertEqual(len(pairs), 2)
    self.assertEqual(pairs[0]['hist1'].tolist(), [1, 1, 1])
    self.assertEqual(pairs[0]['hist2'].tolist(), [0, 1, 1])
    self.assertEqual(pairs[1]['hist2'].tolist(), [1, 1, 1])

  def test_likelihood_outputs_one_lag_result(self):
    ts1 = TimeSeries()
    ts1.add(np.array([0.01, 0.11, 0.21]))
    ts2 = TimeSeries()
    ts2.add(np.array([0.11, 0.21, 0.31]))

    pair = PairTimeSeriesToHist('ts1', 'ts2', 'pairs', 0.1, start=0.0,
                                window=0.4, lag=0.1, name='pair')
    pair.update({'action': 'alert', 'ts1': ts1, 'ts2': ts2,
                 'possible_time_lag_list': np.array([0.0, 0.1])})

    like = PoissonLagLikelihood('pairs', 'likelihood', name='like')
    like.update(pair.last_data)

    self.assertAlmostEqual(like.last_data['likelihood']['lag'], 0.1)
    self.assertIn('log_likelihood', like.last_data['likelihood'])
    self.assertNotIn('best_lag', like.last_data['likelihood'])

  def test_collector_coarse_roi(self):
    n = LikelihoodScanCollector('scan', 'coarse', scan_type='coarse',
                                name='collector')
    n.update({'action': 'alert',
              'scan': {
                  'possible_time_lag_list': np.array([-0.01, 0.0, 0.01]),
                  'log_likelihood_list': np.array([1.0, 3.0, 2.0]),
              }})
    self.assertEqual(n.last_data['coarse']['best_lag'], 0.0)
    self.assertEqual(n.last_data['coarse']['roi']['low'], -0.01)
    self.assertEqual(n.last_data['coarse']['roi']['high'], 0.01)

  def test_collector_accumulates_single_lag_results(self):
    n = LikelihoodScanCollector('likelihood', 'coarse', scan_type='coarse',
                                name='collector')
    n.update({'action': 'alert',
              'likelihood': {'lag': 0.0, 'log_likelihood': 1.0}})
    n.update({'action': 'alert',
              'likelihood': {'lag': 0.1, 'log_likelihood': 2.0}})
    self.assertEqual(n.last_data['coarse']['best_lag'], 0.1)
    np.testing.assert_allclose(
        n.last_data['coarse']['possible_time_lag_list'], [0.0, 0.1])
    np.testing.assert_allclose(n.last_data['coarse']['log_likelihood_list'],
                               [1.0, 2.0])

  def test_peak_refiner_quadratic(self):
    lags = np.array([0.20, 0.22, 0.24, 0.26])
    y = -1000.0 * (lags - 0.23) ** 2 + 7.0
    n = LikelihoodPeakRefiner('fine', 'refined', name='refine')
    n.update({'action': 'alert',
              'fine': {'possible_time_lag_list': lags,
                       'log_likelihood_list': y}})
    self.assertAlmostEqual(n.last_data['refined']['refined_lag'], 0.23)
    self.assertLess(n.last_data['refined']['curvature'], 0.0)

  def test_simple_generated_time_series_scan_recovers_lag(self):
    true_lag = 0.12
    Node.rng = np.random.default_rng(12345)

    data = {'action': 'alert'}
    data = _run_node(NewTimeSeries('ts1', name='new-ts1'), data)
    data = _run_node(GaussianPeak('ts1', event=300, stdev=0.03,
                                  expv=0.40, name='peak-ts1'), data)
    data = _run_node(NewTimeSeries('ts2', name='new-ts2'), data)
    data = _run_node(GaussianPeak('ts2', event=300, stdev=0.03,
                                  expv=0.40 + true_lag,
                                  name='peak-ts2'), data)

    data = _run_node(TimeLagGenerator('possible_time_lag_list',
                                      scan_low=0.06, scan_high=0.18,
                                      step=0.005,
                                      name='possible-time-lag-list'), data)

    pair = PairTimeSeriesToHist('ts1', 'ts2', 'hist_pair', 0.01,
                                start=0.20, window=0.50,
                                lag_field='trial_lag', name='pair')
    like = PoissonLagLikelihood('hist_pair', 'likelihood', name='like')
    collect = LikelihoodScanCollector('likelihood', 'scan',
                                      scan_type='coarse',
                                      name='collector')

    for lag in data['possible_time_lag_list']:
      trial = data.copy()
      trial['trial_lag'] = float(lag)
      pair.update(trial)
      like.update(pair.last_data)
      collect.update(like.last_data)

    scan = collect.last_data['scan']
    self.assertAlmostEqual(scan['best_lag'], true_lag, delta=0.015)
    self.assertGreater(len(scan['possible_time_lag_list']), 1)


if __name__ == '__main__':
  unittest.main()
