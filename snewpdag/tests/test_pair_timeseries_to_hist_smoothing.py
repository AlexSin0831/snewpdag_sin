"""Tests for soft-histogram construction and detector-1 caching."""

import importlib
import unittest
from unittest.mock import patch

import numpy as np

from snewpdag.values import TimeSeries

smoothing_module = importlib.import_module(
    'snewpdag.plugins.PairTimeSeriesToHist_Smoothing')
PairTimeSeriesToHist_Smoothing = smoothing_module.PairTimeSeriesToHist_Smoothing
toy_smoothing_module = importlib.import_module(
    'snewpdag.plugins.PairTimeSeriesToHist_Smoothing_Toy')
PairTimeSeriesToHist_Smoothing_Toy = (
    toy_smoothing_module.PairTimeSeriesToHist_Smoothing_Toy)


class TestPairTimeSeriesToHistSmoothing(unittest.TestCase):

  def test_multiple_timestamps_keep_bin_width_scalar(self):
    result = smoothing_module.ts_to_soft_hist(
        np.array([0.05, 0.15, 0.35]),
        lambda offsets: np.ones_like(offsets),
        bin_width=0.1,
        impact_range=0.1,
        window_start=0.0,
        window_stop=0.5,
    )

    self.assertEqual(result.shape, (5,))
    self.assertTrue(np.all(np.isfinite(result)))

  def test_integrated_kernel_mean_matches_asymmetric_mixture(self):
    result = smoothing_module.mean_calculator(
        10.0,
        noise=0.05,
        rise_const=0.2,
        fall_const=0.3,
        frac_rise=0.5,
        frac_fall=0.5,
    )

    expected_mean = 0.5 * 0.3 - 0.5 * 0.2
    self.assertAlmostEqual(result, expected_mean, places=6)

  def test_detector_1_histogram_is_reused_across_lags(self):
    ts1 = TimeSeries()
    ts1.add(np.array([0.05, 0.15, 0.35]))
    ts2 = TimeSeries()
    ts2.add(np.array([0.10, 0.20, 0.40]))

    node = PairTimeSeriesToHist_Smoothing(
        'ts1', 'ts2', 'hist_pair', hist_bin_width=0.1,
        window_start=0.0, window_size=0.5, lag_field='lag',
        rise_constant=0.2, fall_constant=0.3,
        sigma1=0.05, sigma2=0.05,
        frac_fall=0.5, frac_rise=0.5, impact_range=0.2,
        cache_hist1=True, name='pair-smoothed')

    original_smoother = smoothing_module.ts_to_soft_hist
    with patch.object(
        smoothing_module,
        'ts_to_soft_hist',
        wraps=original_smoother,
    ) as smoother:
      node.update({'action': 'alert', 'ts1': ts1, 'ts2': ts2, 'lag': 0.0})
      first_hist1 = node.last_data['hist_pair']['hist1'].copy()
      first_hist2 = node.last_data['hist_pair']['hist2'].copy()

      node.update({'action': 'alert', 'ts1': ts1, 'ts2': ts2, 'lag': 0.1})
      second_hist1 = node.last_data['hist_pair']['hist1']
      second_hist2 = node.last_data['hist_pair']['hist2']

    self.assertEqual(smoother.call_count, 3)
    np.testing.assert_array_equal(first_hist1, second_hist1)
    self.assertTrue(np.all(np.isfinite(first_hist2)))
    self.assertGreater(np.sum(first_hist2), 0.0)
    self.assertFalse(np.array_equal(first_hist2, second_hist2))

  def test_automatic_window_uses_histogram_time_bounds(self):
    hist1 = smoothing_module.Hist1D(5, 0.0, 0.5)
    hist1.bins[:] = [0.0, 1.0, 0.0, 2.0, 0.0]
    ts2 = TimeSeries()
    ts2.add(np.array([0.2, 0.4]))

    node = PairTimeSeriesToHist_Smoothing(
        'ts1', 'ts2', 'hist_pair', hist_bin_width=0.1,
        in_hist1_field='hist1', hist1_ic=True,
        rise_constant=0.2, fall_constant=0.3,
        sigma1=0.05, sigma2=0.05,
        frac_fall=0.5, frac_rise=0.5, impact_range=0.2,
        name='pair-smoothed')

    node.update({'action': 'alert', 'hist1': hist1, 'ts2': ts2})

    pair = node.last_data['hist_pair']
    self.assertEqual(pair['start'], 0.0)
    self.assertEqual(pair['stop'], 0.5)

  def test_detector_1_cache_is_invalidated_for_new_times(self):
    ts1 = TimeSeries()
    ts1.add(np.array([0.05]))
    ts2 = TimeSeries()
    ts2.add(np.array([0.10]))

    node = PairTimeSeriesToHist_Smoothing(
        'ts1', 'ts2', 'hist_pair', hist_bin_width=0.1,
        window_start=0.0, window_size=0.5, lag_field='lag',
        rise_constant=0.2, fall_constant=0.3,
        sigma1=0.05, sigma2=0.05,
        frac_fall=0.5, frac_rise=0.5, impact_range=0.2,
        cache_hist1=True, name='pair-smoothed')

    node.update({'action': 'alert', 'ts1': ts1, 'ts2': ts2, 'lag': 0.0})
    first_hist1 = node.last_data['hist_pair']['hist1'].copy()

    ts1.add(np.array([0.25]))
    node.update({'action': 'alert', 'ts1': ts1, 'ts2': ts2, 'lag': 0.1})
    second_hist1 = node.last_data['hist_pair']['hist1']

    self.assertGreater(np.sum(second_hist1), np.sum(first_hist1))

  def test_multiple_lags_generate_multiple_smoothed_pairs(self):
    ts1 = TimeSeries()
    ts1.add(np.array([0.05, 0.15, 0.35]))
    ts2 = TimeSeries()
    ts2.add(np.array([0.10, 0.20, 0.40]))

    node = PairTimeSeriesToHist_Smoothing(
        'ts1', 'ts2', 'hist_pair', hist_bin_width=0.1,
        window_start=0.0, window_size=0.5, lags_field='lags',
        rise_constant=0.2, fall_constant=0.3,
        sigma1=0.05, sigma2=0.05,
        frac_fall=0.5, frac_rise=0.5, impact_range=0.2,
        cache_hist1=True, name='pair-smoothed')

    lags = np.array([0.0, 0.1, 0.2])
    node.update({'action': 'alert', 'ts1': ts1, 'ts2': ts2, 'lags': lags})

    result = node.last_data['hist_pair']
    pairs = result['pairs']
    self.assertEqual(len(pairs), len(lags))
    np.testing.assert_allclose([pair['lag'] for pair in pairs], lags)
    np.testing.assert_allclose(result['possible_time_lag_list'], lags)
    np.testing.assert_array_equal(pairs[0]['hist1'], pairs[1]['hist1'])
    self.assertIsNot(pairs[0]['hist1'], pairs[1]['hist1'])
    self.assertFalse(np.array_equal(pairs[0]['hist2'], pairs[1]['hist2']))
    self.assertNotIn('hist1', result)

  def test_multiple_lags_generate_multiple_toy_pairs_for_icecube(self):
    hist1 = toy_smoothing_module.Hist1D(5, 0.0, 0.5)
    hist1.bins[:] = [0.0, 1.0, 0.0, 2.0, 0.0]
    ts2 = TimeSeries()
    ts2.add(np.array([0.10, 0.20, 0.40]))

    node = PairTimeSeriesToHist_Smoothing_Toy(
        in_ts2_field='ts2', out_hist_field='hist_pair',
        hist_bin_width=0.1, in_hist1_field='hist1', hist1_ic=True,
        window_start=0.0, window_size=0.5, lags_field='lags',
        sigma_gnd=0.05, impact_range=0.2,
        cache_hist1=True, det1='IceCube', name='pair-smoothed-toy')

    lags = np.array([0.0, 0.1, 0.2])
    node.update({'action': 'alert', 'hist1': hist1, 'ts2': ts2, 'lags': lags})

    result = node.last_data['hist_pair']
    pairs = result['pairs']
    self.assertEqual(len(pairs), len(lags))
    np.testing.assert_allclose([pair['lag'] for pair in pairs], lags)
    np.testing.assert_allclose(result['possible_time_lag_list'], lags)
    np.testing.assert_array_equal(pairs[0]['hist1'], pairs[1]['hist1'])
    self.assertIsNot(pairs[0]['hist1'], pairs[1]['hist1'])
    self.assertFalse(np.array_equal(pairs[0]['hist2'], pairs[1]['hist2']))
    self.assertNotIn('hist1', result)


if __name__ == '__main__':
  unittest.main()
