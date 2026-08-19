import logging
import numpy as np

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field
from snewpdag.values import Hist1D

class FirstEventIceCubeInterpolation (Node): 
    def __init__(self, in_start_time_field, in_hist_field, in_truth_field, out_field, out_delta_field, factor, **kwargs):
        self.in_start_time_field = in_start_time_field
        self.in_hist_field = in_hist_field
        self.in_truth_field = in_truth_field # only stores the "idealistic" arrival time of neutrinos
        self.out_field = out_field 
        self.out_delta_field = out_delta_field
        self.factor = factor
        super().__init__(**kwargs)
    
    def alert(self, data): 
        given_start_time, valid = fetch_field(data, self.in_start_time_field) # IceCube provides 
        if not valid: 
            return False 
        
        hist, valid = fetch_field(data, self.in_hist_field) 
        if not valid: 
            return False
        # we probably need another node to convert the data from IceCube into a Hist1D object
        # also to store the "given start time"
        # but for simplicity, I assume hist is already a Hist1D object
        if not isinstance(hist, Hist1D):
            return False
        
        # Interpolation of the detection rate: 
        factor = int(self.factor) # number of equal pieces that we chop the 2 ms into
        if factor <= 0: 
            return False
        
        # number of bins / intervals:
        new_nbins = hist.nbins * factor 

        # bin_width:
        old_bin_width = hist.xwidth / hist.nbins
        new_bin_width = old_bin_width / factor 

        old_bin_counts = hist.bins.astype(float)
        old_edges = np.linspace(hist.xlow, hist.xhigh, hist.nbins + 1) # that's why we need +1 here
        old_centres = 0.5 * (old_edges[:-1] + old_edges[1:])
        old_rates = old_bin_counts / old_bin_width # per 2 ms

        new_edges = np.linspace(hist.xlow, hist.xhigh, new_nbins+ 1)
        new_centres = 0.5 * (new_edges[:-1] + new_edges[1:])
        # interpolated detection rates (per 2 ms):
        new_rates = np.interp(new_centres, # wanted x-values
                              old_centres, # known x-values 
                              old_rates) # known x-values

        # new_bin_width < old_bin_width ==> new_bin_counts < old_bin_counts 
        # resolution will become higher but detection per bin will become smaller
        new_bin_counts = new_rates * new_bin_width 

        # original FirstEventDebias.py workflow:
        new_bin_start = int((given_start_time - hist.xlow) / new_bin_width)
        if new_bin_start <= 0:
            return False
        
        # calculate the background noise using the given start time 
        # note that new_bin_start is larger than old_bin_start ==> bg_rate will drop 
        # but this makes sense, cuz the bin-width becomes smaller, so the background contribution
        # to each bin will get smaller
        bg_rate = np.sum(new_bin_counts[:new_bin_start])/new_bin_start # per new time scale (smaller!)
        new_bin_counts_without_bg = new_bin_counts - bg_rate 
        
        # algorithm from FirstEventDebias
        i_first = np.argmax(new_bin_counts_without_bg)
        i_last = i_first 
        while new_bin_counts_without_bg[i_first] > 0 and i_first > 0: 
            i_first -= 1
        i_first = i_first + 1 # the first bin that gives positive value
        experimental_start_time = new_edges[i_first]
        difference = np.abs(given_start_time - experimental_start_time)
        logging.debug('Difference between start time h = {}'.format(difference))

        while i_last < len(new_bin_counts_without_bg):
            if new_bin_counts_without_bg[i_last] < 0: 
                break
            i_last += 1 # the last bin that gives positive value 
        
        region_of_interest = new_bin_counts_without_bg[i_first:i_last]
        # we minus experimental_start_time to ensure the bias won't blow up / calculate some non-sense big numbers
        local_time_stamps = new_edges[i_first:i_last] - experimental_start_time

        mu = np.cumsum(region_of_interest)
        exp = np.exp(-mu)

        numerator = region_of_interest[0] * local_time_stamps[0] + np.sum(region_of_interest[1:] * local_time_stamps[1:] * exp[:-1])
        denominator = region_of_interest[0] + np.sum(region_of_interest[1:] * exp[:-1])

        bias = numerator / denominator # Equation (9) in the Paper
        
        corrected_start_time = experimental_start_time - bias

        store_field(data, self.out_field, corrected_start_time)
        
        true_time, valid = fetch_field(data, self.in_truth_field)
        if valid: 
            delta = corrected_start_time - true_time
            store_field(data, self.out_delta_field, delta)
        
        return True 