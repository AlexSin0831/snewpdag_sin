"""
It would be an interesting exercise to code up the first-events algorithm where the reference detector is IceCube.  
IceCube will have something close to a million events (on top of a huge background of something like 1.5M events/sec), 
but instead of sending us individual event times, we should assume it sends us a “start time” of when it thinks the burst started, 
along with a histogram with 2ms bins.  The formula for this situation is in the paper.
"""

import numpy as np

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field
from snewpdag.values import Hist1D

class FirstEventIceCube (Node): 
    def __init__(self, in_start_time_field, in_hist_field, in_truth_field, out_field, out_delta_field, **kwargs):
        self.in_start_time_field = in_start_time_field
        self.in_hist_field = in_hist_field
        self.in_truth_field = in_truth_field
        self.out_field = out_field 
        self.out_delta_field = out_delta_field
        super().__init__(**kwargs)
    
    def alert(self, data): 
        start_time, valid = fetch_field(data, self.in_start_time_field) # IceCube provides 
        if not valid: 
            return False 
        
        hist, valid = fetch_field(data, self.in_hist_field) # assume hist is an Hist1D object
        if not valid: 
            return False
        
        # find the bin corresponded to the given starting time:
        bin_width = hist.xwidth / hist.nbins
        bin_start = int((start_time - hist.xlow) / bin_width)
        if bin_start <= 0:
            return False
        
        bg_rate = np.sum(hist.bins[:bin_start])/bin_start # background events / 2ms
        hist_without_bg = hist.bins - bg_rate # np.array

        i_first = np.argmax(hist_without_bg)
        i_last = i_first 
        while hist_without_bg[i_first] > 0 and i_first > 0: 
            i_first -= 1
        i_first = i_first + 1 # the first bin that gives positive value
        while i_last < len(hist_without_bg):
            if hist_without_bg[i_last] < 0: 
                break
            i_last += 1 # the last bin that gives positive value 
        
        region_of_interest = hist_without_bg[i_first:i_last]
        indices = np.arange(i_first,i_last)
        time_stamps = hist.bin_edge(indices)

        mu = np.cumsum(region_of_interest)
        exp = np.exp(-mu)

        numerator = region_of_interest[0] * time_stamps[0] + np.sum(region_of_interest[1:] * time_stamps[1:] * exp[:-1])
        denominator = region_of_interest[0] + np.sum(region_of_interest[1:] * exp[:-1])

        expected_value = numerator / denominator # Equation (9) in the Paper
        
        correct_start_time = start_time - expected_value

        store_field(data, self.out_field, correct_start_time)
        
        true_time, valid = fetch_field(data, self.in_truth_field)
        if valid: 
            delta = correct_start_time - true_time
            store_field(data, self.out_delta_field, delta)
        
        return True 