"""Tests for uniform background generation in Hist1D values."""

import unittest
from unittest.mock import patch

import numpy as np

from snewpdag.dag import Node
from snewpdag.plugins.gen.Uniform import Uniform
from snewpdag.values import Hist1D


class MeanReturningRng:
  """Return Poisson means unchanged so tests can inspect bin geometry."""

  def poisson(self, lam, size=None):
    values = np.asarray(lam, dtype=np.float64)
    if size is not None:
      return np.full(size, values, dtype=np.float64)
    return values.copy()


class TestUniformHistogram(unittest.TestCase):

  def run_uniform(self, histogram, rate, tmin, tmax):
    node = Uniform('hist', rate, tmin, tmax, name='uniform')
    with patch.object(Node, 'rng', MeanReturningRng()):
      node.update({'action': 'alert', 'hist': histogram})
    return node.last_data['hist']

  def test_only_fills_overlap_with_requested_interval(self):
    histogram = Hist1D(4, 0.0, 4.0)

    result = self.run_uniform(histogram, rate=10.0, tmin=0.5, tmax=2.25)

    np.testing.assert_allclose(result.bins, [5.0, 10.0, 2.5, 0.0])

  def test_interval_outside_histogram_adds_no_counts(self):
    histogram = Hist1D(4, 0.0, 4.0)

    result = self.run_uniform(histogram, rate=10.0, tmin=-2.0, tmax=-1.0)

    np.testing.assert_array_equal(result.bins, np.zeros(4))

  def test_full_bin_means_match_rate_times_bin_width(self):
    histogram = Hist1D(4, 0.0, 2.0)

    result = self.run_uniform(histogram, rate=6.0, tmin=0.0, tmax=2.0)

    np.testing.assert_allclose(result.bins, [3.0, 3.0, 3.0, 3.0])


if __name__ == '__main__':
  unittest.main()
