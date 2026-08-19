"""
PairTimeSeriesToHist: Build aligned histogram pairs for one lag or a list of lags. 

configuration: 
  in_ts1_field / in_hist1_field : field name of the TimeSeries / Histogram of det1
  in_ts2_field                  : field name of the TimeSeries of det2 (must be non-IceCube)
  out_hist_filed                : field name of the histogram output
  bin_width                     : bin-width of the histogram generated

  Optional: 
  hist1_ic                      : if True, we analyse the histogram by IceCube.
                                  if False, we analyse the TimeSeries by IceCube. 
  window_start                  : the starting time of our region of interest to a detector's TimeSeries / Histogram
  window_size                   : how long should we pay attention to the TimeSeries / Histogram?
  lag / lag_field / lags_field  : if lag, it is just a number 
                                  if lag_field, it is the field name of one lag 
                                  if lags_field, it is the field name of a list of lags 

output payload: 
  pairs                         : if lag / lag_field, only one hist pair
                                  if lags_field, there will be multiple hist pairs
"""

import numpy as np

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field
from snewpdag.values import Hist1D


def histogram_pair(data1, ts2, nbins, start, stop, lag, bin_width, hist1_ic):
  edges = np.linspace(start, stop, nbins + 1)

  # Positive guessed time lag ==> we guess the detector 2 detects later
  # ==> so we align it by shifting the time series to the opposite direction.
  shifted_ts2_times = np.asarray(ts2.times, dtype=np.float64) - float(lag)

  if hist1_ic == True:
    source_edges = np.linspace(data1.xlow, data1.xhigh, data1.nbins + 1)
    source_bin_centres = (source_edges[:-1] + source_edges[1:]) / 2.0
    hist_1, unshifted_edges = np.histogram(source_bin_centres,
                                          bins=edges,
                                          weights=data1.bins)
  else:
    hist_1, unshifted_edges = np.histogram(data1.times, bins=edges)

  hist_2, shifted_edges = np.histogram(shifted_ts2_times, bins=edges)

  hist_1 = np.asarray(hist_1)
  hist_2 = np.asarray(hist_2, dtype=np.int64)

  return {
      'lag': float(lag),
      'hist1': hist_1,
      'hist2': hist_2,
      'edges': unshifted_edges,
      'shifted_edges': shifted_edges,
      'ts2_shift': -float(lag),
      'start': float(start),
      'stop': float(stop),
      'bin_width': float(bin_width),
      'real_window_nbins': int(nbins)
  }

class PairTimeSeriesToHist(Node):
  def __init__(self, in_ts1_field=None, in_ts2_field=None, out_hist_field=None,
               bin_width=None, **kwargs):
    if bin_width is None:
      bin_width = kwargs.pop('hist_bin_width', None)
    if bin_width is None:
      bin_width = kwargs.pop('bin_width', None)
    if bin_width is None:
      raise TypeError('PairTimeSeriesToHist requires bin_width')

    self.in_ts1_field = in_ts1_field
    self.in_ts2_field = in_ts2_field
    self.in_hist1_field = kwargs.pop('in_hist1_field', None)
    self.hist1_ic = kwargs.pop('hist1_ic', False)

    self.out_hist_field = out_hist_field
    self.bin_width = float(bin_width)
    
    self.window_start = kwargs.pop('window_start', None)
    if self.window_start is None:
      self.window_start = kwargs.pop('start', None)
    else:
      kwargs.pop('start', None)

    self.window_stop = kwargs.pop('window_stop', None)
    if self.window_stop is None:
      self.window_stop = kwargs.pop('stop', None)
    else:
      kwargs.pop('stop', None)

    self.window_size = kwargs.pop('window_size', None)
    if self.window_size is None:
      self.window_size = kwargs.pop('window', None)
    else:
      kwargs.pop('window', None)
    self.lead_time = kwargs.pop('lead_time', 0.0)

    # A lag can be supplied directly or fetched from the payload.
    self.lag = kwargs.pop('lag', 0.0)
    self.lag_field = kwargs.pop('lag_field', None)
    self.lags_field = kwargs.pop('lags_field', None)
    if self.lags_field is None:
      self.lags_field = kwargs.pop('possible_time_lag_list_field', None)
    else:
      kwargs.pop('possible_time_lag_list_field', None)
    super().__init__(**kwargs)

  def define_window(self, data1, ts2):
    # Error-tracking
    if self.hist1_ic == True:
      data1_is_empty = data1.nbins == 0
    else:
      data1_is_empty = len(data1.times) == 0

    if data1_is_empty or len(ts2.times) == 0:
      if self.window_start is None or (self.window_stop is None and self.window_size is None):
        raise ValueError('Cannot define window with empty detector data and no explicit window parameters.')
        return None, None, None

    # window_start:
    if self.window_start is not None:
      window_start = self.window_start
    elif self.hist1_ic == True:
      window_start = data1.xlow + self.lead_time
    else:
      window_start = np.min(data1.times) + self.lead_time

    # window_stop:
    if self.window_stop is not None:
      window_stop = self.window_stop
    elif self.window_size is not None:
      window_stop = window_start + self.window_size
    elif self.hist1_ic == True:
      window_stop = max(data1.xhigh, np.max(ts2.times) + self.bin_width)
    else:
      window_stop = max(np.max(data1.times), np.max(ts2.times)) + self.bin_width

    estimated_nbins = (window_stop - window_start) / self.bin_width
    if np.isclose(estimated_nbins, round(estimated_nbins)):
      nbins = int(round(estimated_nbins))
    else:
      nbins = int(np.ceil(estimated_nbins))

    # Adjust the stop so the histogram is an exact integer number of bins.
    window_stop = window_start + nbins * self.bin_width
    return float(window_start), float(window_stop), nbins

  def retrieve_lags(self, data):
    # for multiple lags:
    if self.lags_field is not None:
      lags, valid = fetch_field(data, self.lags_field)
      if valid:
        return np.asarray(lags, dtype=np.float64)

    # for single lag:
    if self.lag_field is not None:
      lag, valid = fetch_field(data, self.lag_field)
      if valid:
        return np.asarray([lag], dtype=np.float64)

    return np.asarray([self.lag], dtype=np.float64)

  # Main Part:
  def alert(self, data):
    if self.hist1_ic == True:
      data1, valid = fetch_field(data, self.in_hist1_field)
      if not valid:
        raise ValueError("Hard Hist field from IceCube '{}' is not found".format(self.in_hist1_field))
        return False
      if not isinstance(data1, Hist1D):
        raise TypeError("Field '{}' must contain a Hist1D".format(self.in_hist1_field))
    else:
      data1, valid = fetch_field(data, self.in_ts1_field)
      if not valid:
        return False

    ts2, valid = fetch_field(data, self.in_ts2_field)
    if not valid:
      return False

    window_start, window_stop, nbins = self.define_window(data1, ts2)
    if nbins is None or nbins <= 0:
      return False

    lags = self.retrieve_lags(data)
    
    
    pairs = [histogram_pair(data1, ts2, nbins, window_start, window_stop, lag, self.bin_width, self.hist1_ic) for lag in lags]

    payload = {
        'pairs': pairs,
        'possible_time_lag_list': lags.copy(),
        'start': window_start,
        'stop': window_stop,
        'bin_width': self.bin_width,
        'nbins': nbins,
    }
    if len(pairs) == 1:
      payload.update(pairs[0])
    store_field(data, self.out_hist_field, payload)
    return True
