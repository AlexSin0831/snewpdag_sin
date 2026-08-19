"""
GenTimeDist:  generates a time distribution on each alert, based on a histogram

configuration:
  field:     field name. Must be TimeSeries or TimeHist. Modified in place.
  sig_mean:  mean number of events to generate.
             If it's a number, use the number itself.  Default is the
               area of the histogram read from the input spectrum.
             If it's a field designator, read from the payload.
  sig_distance:  distance in kpc.  Default is 10.
             Signal is scaled by (10/d)^2.
             If it's a number, use the number itself.
             If it's a field designator, read from the payload.
  sig_smear: True if apply Poisson fluctuation to mean (optional, def True)
  sig_t0:    observed core bounce time. This will correspond to the t=0
             of the input distribution.  Optional, default 1658580450.0,
               which happens to be sometime on 23 July 2022 in the UK.
             If it's a float, use the number as a timestamp.
             If it's a field, read the time stamp from the payload.
  sig_once:  True if only generate one series of offsets for ll GenTimeDists's,
             as one might do for a perverse test (default False!).
             Note that the offsets will be shifted according to sig_t0,
             and the number of events will match the first module to run.
  epoch_base (optional): starting time for epoch, float value or field specifier
    (string or tuple)

  A "field specifier" is either a string or tuple of strings
  which navigate into the payload.

Originally based on Vladimir's TimeDistFileInput, via TimeDist

Need a generator for SN direction and core bounce times for each detector.
"""


import logging
import numpy as np
import numbers

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field
from snewpdag.values import Hist1D, TimeSeries
from . import TimeDistSource
from astropy.time import Time


class GenTimeDist(TimeDistSource): 

  # for sig_once: 
  one_series = () 
  one_mean = 0 

  def __init__(self, field, **kwargs):
    self.field = field
    # optional arguments:
    ts = kwargs.pop('sig_t0', 0.0)
    if isinstance(ts, (list, tuple, str)): # field specifier
        self.sig_t0 = ts
    elif isinstance(ts, numbers.Number): 
        self.sig_t0 = ts
    self.sig_mean = kwargs.pop('sig_mean', 0.0)
    self.sig_distance = kwargs.pop('sig_distance', 10.0)
    self.sig_smear = kwargs.pop('sig_smear', True)
    self.sig_once = kwargs.pop('sig_once', False) 
    self.epoch_base = kwargs.pop('epoch_base', 0.0)

    if not isinstance(self.epoch_base, (numbers.Number, str, list, tuple)):
      logging.error('GenTimeDist.__init__: unrecognized epoch_base {}. Set to 0.'.format(self.epoch_base))
      self.epoch_base = 0.0

    # TimeDistSource.py 要read filename 同 filetype 所以其實 we must provide it...
    # Otherwise self.mu will be undefined...
    super().__init__(**kwargs)
    self.area = np.sum(self.mu) # self.mu is the theoretical light curve by the model we provided
    self.mu_norm = self.mu / self.area # become probability density 
    self.tedges = np.append(self.t, self.thi) # all the edges cuz self.t doesn't include the highest edge

    # This controls the bias between large detectors and small detectors 
    # But, if we forget to provide it, then it will use the model detection number to continue the calculation. 
    if self.sig_mean == 0 or self.sig_mean == "":
      self.sig_mean = self.area 
      logging.info('{}:  mean set to area {}'.format(self.name, self.area))

    # pre-generate single series
    if self.sig_once and np.shape(GenTimeDist.one_series) == (0,): # this means the shared set has not yet been generated
      j = Node.rng.choice(len(self.mu_norm), # [0,1,2,...,len(self.mu_norm)-1]?
                          self.sig_mean, # choose how many?
                          p=self.mu_norm, # probability distribution
                          replace=True, # can have repeated time stamps
                          shuffle=False) 
      ta = self.tedges[j]
      dt = self.tedges[j+1] - ta
      # we need the dt to show the gap between each time stamp, and we use it to be the factor for us to add noise. 
      GenTimeDist.one_series = ta + Node.rng.random(self.sig_mean) * dt
      GenTimeDist.one_mean = self.sig_mean
    # 如果開咗 sig_once: 
    # 咁 detA 可能度到： [100.002, 100.006, 100.011]
    # det B 可能度到： [100.012, 100.016, 100.021]
    # Note that the relative separation is the same, only the starting point has shifted 
    # Remove the effect by random event generation ==> Only see how the time shift affect
  def alert(self, data):
    # v can be interpreted as some empty time series created by ops.NewTimeSeries.py 
    v, flag = fetch_field(data, self.field)

    if flag:

      # epoch base
      if isinstance(self.epoch_base, numbers.Number):
        te = self.epoch_base
      elif isinstance(self.epoch_base, (str, list, tuple)):
        te = fetch_field(data, self.epoch_base)

      # adjust offsets for t0 and TimeSeries reference timestamps.
      # For instance, if core bounce is at 100s, but ref time is 90s, (ref time means epoch_base)
      # then an event at t=0 should have an offset of 10s.
      if isinstance(self.sig_t0, (str, tuple, list)): # interpret as field
        t0, flag = fetch_field(data, self.sig_t0) # s in unix epoch # 如果係文字就要 fetch 返個數字出嚟
        if not flag:
          logging.error('{}: {} not found in payload'.format(self.name, self.sig_t0))
          return False
      else:
        t0 = self.sig_t0 # 如果係數字就直接係咁

      offset = t0 - te
      logging.debug('{}: t0 = {}, ref time {}, offset {}'.format(self.name, t0, te, offset))

      if self.sig_once:
        n = int(self.sig_mean / GenTimeDist.one_mean)
        b = np.empty((n, len(GenTimeDist.one_series)))
        for i in range(n):
          np.copyto(b[i], GenTimeDist.one_series)
        a = np.ravel(b) # flatten to 1D
      else:
        # set mean number of events to generate
        if isinstance(self.sig_mean, numbers.Number):
          mean = self.sig_mean
        else:
          mean, flag = fetch_field(data, self.sig_mean)
          if not flag:
            mean = self.area # area of source histogram
            logging.error('{}:  sig_mean field {} not found'.format(self.name, self.sig_mean))
        # scale mean number of events by distance
        if isinstance(self.sig_distance, numbers.Number):
          f = 10.0 / self.sig_distance
        else:
          d, flag = fetch_field(data, self.sig_distance)
          if flag:
            f = 10.0 / d
          else:
            f = 1.0
            logging.error('{}:  sig_distance field {} not found'.format(self.name, self.sig_distance))
        mean = mean * f * f

        # Poisson fluctuation in mean, if requested
        nev = Node.rng.poisson(mean) if self.sig_smear else mean

        # generate time series of offsets, with t=0 at core bounce
        j = Node.rng.choice(len(self.mu_norm), 
                            nev,
                            p=self.mu_norm, 
                            replace=True, 
                            shuffle=False)
        ta = self.tedges[j]
        dt = self.tedges[j+1] - ta
        a = ta + Node.rng.random(nev) * dt

      # add offsets in seconds - works for Hist1D or TimeSeries
      a += offset
      v.add(a)
      return data
    else:
      return False
