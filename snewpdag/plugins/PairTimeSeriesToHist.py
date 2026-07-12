"""
Build aligned histogram pairs for one lag or a list of lags.
Also add flat Poisson background noise here.
"""

import numpy as np

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field


def add_flat_background(background_rate, bin_width, shape):
  mean_background = float(background_rate) * float(bin_width)
  if mean_background <= 0.0:
    return np.zeros(shape, dtype=np.int64)
  return Node.rng.poisson(mean_background, size=shape).astype(np.int64)


def histogram_pair(ts1, ts2, nbins, start, stop, lag, background_rate_1, background_rate_2, bin_width):
  edges = np.linspace(start, stop, nbins + 1)

  # Positive lag means detector 2 is later, so align it by shifting earlier.
  shifted_ts2_times = np.asarray(ts2.times, dtype=np.float64) - float(lag)

  hist_1, edges = np.histogram(ts1.times, bins=edges)
  hist_2, shifted_edges = np.histogram(shifted_ts2_times, bins=edges)

  hist_1 = np.asarray(hist_1, dtype=np.int64)
  hist_2 = np.asarray(hist_2, dtype=np.int64)

  # background_rate is per second, so each bin gets Poisson(rate * bin_width)
  hist_1 = hist_1 + add_flat_background(background_rate_1, bin_width, hist_1.shape)
  hist_2 = hist_2 + add_flat_background(background_rate_2, bin_width, hist_2.shape)

  return {
      'lag': float(lag),
      'hist1': hist_1,
      'hist2': hist_2,
      'edges': edges,
      'shifted_edges': shifted_edges,
      'ts2_shift': -float(lag),
      'start': float(start),
      'stop': float(stop),
      'bin_width': float((stop - start) / nbins),
  }

class PairTimeSeriesToHist(Node):
  def __init__(self, in_ts1_field, in_ts2_field, out_hist_field,
               bin_width=None, background_rate_1=0.0, background_rate_2=0.0, **kwargs):
    if bin_width is None:
      bin_width = kwargs.pop('hist_bin_width', None)
    if bin_width is None:
      bin_width = kwargs.pop('bin_width', None)
    if bin_width is None:
      raise TypeError('PairTimeSeriesToHist requires bin_width')

    self.in_ts1_field = in_ts1_field
    self.in_ts2_field = in_ts2_field
    self.out_hist_field = out_hist_field
    self.background_rate_1 = kwargs.pop('background_rate_1', background_rate_1)
    self.background_rate_2 = kwargs.pop('background_rate_2', background_rate_2)
    self.bin_width = float(bin_width)

    self.start = kwargs.pop('start', None)
    self.stop = kwargs.pop('stop', None)
    self.window = kwargs.pop('window', None)
    self.lead_time = kwargs.pop('lead_time', 0.0)
    self.lag = kwargs.pop('lag', 0.0)
    self.lag_field = kwargs.pop('lag_field', None)
    self.lags_field = kwargs.pop('possible_time_lag_list_field', None)
    super().__init__(**kwargs)

  def define_window(self, ts1, ts2):
    if len(ts1.times) == 0 or len(ts2.times) == 0:
      return None, None, None
    start = np.min(ts1.times) + self.lead_time if self.start is None else self.start
    if self.stop is not None:
      stop = self.stop
    elif self.window is not None:
      stop = start + self.window
    else:
      stop = max(np.max(ts1.times), np.max(ts2.times)) + self.bin_width
    nbins = int(np.ceil((stop - start) / self.bin_width))
    stop = start + nbins * self.bin_width
    return float(start), float(stop), nbins

  def lags(self, data):
    if self.lags_field is not None:
      lags, valid = fetch_field(data, self.lags_field)
      if valid:
        return np.asarray(lags, dtype=np.float64)
    if self.lag_field is not None:
      lag, valid = fetch_field(data, self.lag_field)
      if valid:
        return np.asarray([lag], dtype=np.float64)
    return np.asarray([self.lag], dtype=np.float64)

  def alert(self, data):
    ts1, valid = fetch_field(data, self.in_ts1_field)
    if not valid:
      return False
    ts2, valid = fetch_field(data, self.in_ts2_field)
    if not valid:
      return False

    start, stop, nbins = self.define_window(ts1, ts2)
    if nbins is None or nbins <= 0:
      return False

    lags = self.lags(data)
    pairs = [
        histogram_pair(
            ts1, ts2, nbins, start, stop, lag, self.background_rate_1,
            self.background_rate_2, self.bin_width)
        for lag in lags
    ]
    payload = {
        'pairs': pairs,
        'possible_time_lag_list': lags.copy(),
        'start': start,
        'stop': stop,
        'bin_width': self.bin_width,
        'nbins': nbins,
    }
    if len(pairs) == 1:
      payload.update(pairs[0])
    store_field(data, self.out_hist_field, payload)
    return True
