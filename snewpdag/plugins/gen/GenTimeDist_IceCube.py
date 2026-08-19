"""
GenTimeDist_IceCube:  generates a histogram for IceCube on each alert, based on a histogram

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
  epoch_base (optional): starting time for epoch, float value or field specifier
    (string or tuple)

  A "field specifier" is either a string or tuple of strings
  which navigate into the payload.

Originally based on Vladimir's TimeDistFileInput, via TimeDist

Need a generator for SN direction and core bounce times for each detector.
"""
"""
I guess sig_once is not that useful for IceCube?
"""
import logging
import numpy as np
import numbers

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field
from snewpdag.values import Hist1D, TimeSeries
from . import TimeDistSource
from astropy.time import Time

class GenTimeDist_IceCube (TimeDistSource):
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
        self.epoch_base = kwargs.pop('epoch_base', 0.0)
        self.bin_width = kwargs.pop('bin_width', 0.002)   #IceCube default bin-width

        if not isinstance(self.epoch_base, (numbers.Number, str, list, tuple)):
            logging.error('GenTimeDist.__init__: unrecognized epoch_base {}. Set to 0.'.format(self.epoch_base))
            self.epoch_base = 0.0

        super().__init__(**kwargs) # sig_filename, sig_filetype

        # we need to put super().__init__(**kwargs) above, otherwise self.mu & self.t & self.thi would be undefined. 
        self.area = np.sum(self.mu)
        self.mu_norm = self.mu / self.area # kinda like a probability distribution
        self.tedges = np.append(self.t, self.thi) # model edges aren't necessarily same as the hist's edges

        if self.sig_mean == 0 or self.sig_mean == "": 
            self.sig_mean = self.area 
            logging.info('{}:  mean set to area {}'.format(self.name, self.area))

    def alert(self, data):
        hist, flag = fetch_field(data, self.field) # an empty histogram
        
        # assume that is a Hist1D object for simplicity
        if not isinstance(hist, Hist1D): 
            return False 
        
        if flag: 
            # we ensured that self.epoch_base must be number/str/list/tuple in __init__
            if isinstance(self.epoch_base, numbers.Number): 
                t_epoch = self.epoch_base 
            elif isinstance(self.epoch_base, (str, list, tuple)): 
                t_epoch, flag = fetch_field(data, self.epoch_base)
            
            if isinstance(self.sig_t0, (str, tuple, list)): 
                t_start_given, flag = fetch_field(data, self.sig_t0)
                if not flag: 
                    logging.error('{}:{} not found in payload'.format(self.name, self.sig_t0))
                    return False
            else: 
                t_start_given = self.sig_t0
            
            offset = t_start_given - t_epoch
            logging.debug('{}: t0 = {}, ref time {}, offset {}'.format(self.name, t_start_given, t_epoch, offset))

            # self.sig_mean is either the expected counts calculated by the model, or a field specifier
            if isinstance(self.sig_mean, numbers.Number):
                expected_total_count = self.sig_mean
            else: 
                expected_total_count, flag = fetch_field(data, self.sig_mean)
                if not flag: 
                    expected_total_count = self.area # use model's result ; still goes on first
                    logging.error('{}: sig_mean field {} not found'.format(self.name, self.sig_mean))

            if isinstance(self.sig_distance, numbers.Number): 
                f = 10.0 / self.sig_distance
            else: 
                d, flag = fetch_field(data, self.sig_distance)
                if flag: 
                    f = 10.0 / d
                else: 
                    f = 1.0 # default to set the distance as 10kpc ; still goes on first
                    logging.error('{}:  sig_distance field {} not found'.format(self.name, self.sig_distance))
            expected_total_count = expected_total_count * f * f 

            # generate a histogram with respect to the epoch_base time scale
            # so do self.tedges start from 0? 
            model_edges = self.tedges + offset 
            # problem: we need to have a common "zero-point" in the timeline 
            # how can we do that? 
            # is the histogram time axis generated according to the epoch_base? 
            hist_edges = np.linspace(hist.xlow, 
                                     hist.xhigh, 
                                     hist.nbins+1)

            # it is more convenient to use cdf for interpolation:
            model_cdf = np.concatenate(([0.0], np.cumsum(self.mu_norm)))
            hist_cdf = np.interp(hist_edges, # wanted x-values  
                                 model_edges, # known x-values
                                 model_cdf, # known y-values
                                 left = model_cdf[0],
                                 right = model_cdf[-1]
                                 )
            
            # array that saves the counts of the histogram
            expected_bin_counts = expected_total_count * np.diff(hist_cdf) 

                                       
            if self.sig_smear:
                experimental_bin_counts = Node.rng.poisson(expected_bin_counts)
            else: 
                experimental_bin_counts = expected_bin_counts.copy()

            hist.bins += experimental_bin_counts
            return data 
        else: 
            return False

