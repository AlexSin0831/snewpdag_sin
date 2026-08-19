"""
PairTimeSeriesToHist_Smoothing_Toy: build aligned SOFT histogram pairs for one lag or a list of lags.
                                    using Gaussian normal distribution of the smoothing kernel. 

configuration: 
    in_ts1_field / in_hist1_field : field name of the TimeSeries / Histogram of det1
    in_ts2_field                  : field name of the TimeSeries of det2 (must be non-IceCube)
    out_hist_filed                : field name of the histogram output
    bin_width                     : bin-width of the histogram generated

    (Optional:)
    hist1_ic                      : if True, we analyse the histogram by IceCube.
                                    if False, we analyse the TimeSeries by IceCube. 
    window_start                  : the starting time of our region of interest to a detector's TimeSeries / Histogram
    window_size                   : how long should we pay attention to the TimeSeries / Histogram?
    lag / lag_field / lags_field  : if lag, it is just a number 
                                    if lag_field, it is the field name of one lag 
                                    if lags_field, it is the field name of a list of lags   
    sigma_gnd                     : standard deviation of the Gaussian normal distribution
    impact_range                  : how wide can a time stamp smear out in our soft histogram?
    cache_hist1                   : True / False (cache the soft histogram of det1 or not, after all the histogram of det1 will not change
                                    throughout the likelihood-scanning calculation)
    det1                          : name of the det1
"""

import numpy as np
from scipy.special import erf, erfc
import scipy.integrate as integrate 

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field
from snewpdag.values import Hist1D

# ~10 ms is a good range that can give us good-looking plot: 
def gaussian_noise(x:float, sigma:float) -> float:
    denom = np.sqrt(2 * np.pi * (sigma)**2)
    expo = np.exp(-x**2/(2*(sigma)**2))
    return expo/denom

def shift_timeseries(ts, lag):
    return np.asarray(ts, dtype=np.float64) - float(lag)
        
# For non-IceCube detectors:
def ts_to_soft_hist(ts, kernel_calculator, bin_width: float, impact_range: float, window_start: float, window_stop: float): 
    """
    Transform a time series to a soft histogram according to the kernel distribution. 
    Do a numerical mid-point integration on every time stamp of the time series. 
    There's an impact range for each time stamp, and we only add the contribution inside the "valid impact range"
    """

    ts = np.array(ts, dtype=np.float64)
    nbins = int(np.ceil((window_stop - window_start) / bin_width))
    soft_hist = np.zeros(nbins, dtype=np.float64)
    
    for time_stamp in ts: 
        first_impact_value = time_stamp - impact_range 
        last_impact_value = time_stamp + impact_range 

        # define bin_indice = 0 for window_start 
        # lower limit of the integration: 
        first_impact_value = max(window_start, first_impact_value)
        first_impact_bin = int(np.floor((first_impact_value - window_start) / bin_width))

        # upper limit of the integration:
        last_impact_value = min(window_stop, last_impact_value)
        last_impact_bin = int(np.ceil((last_impact_value - window_start) / bin_width))

        if first_impact_bin >= last_impact_bin:
            continue

        # integration region:         
        impact_range_bin_indices = np.arange(first_impact_bin, last_impact_bin)
        bin_centre_values = window_start + (impact_range_bin_indices + 0.5) * bin_width

        # kernel_calculator is a function, defined by "offset" with respect to a timestamp 
        t_contrib = kernel_calculator(bin_centre_values - time_stamp) * bin_width
        soft_hist[impact_range_bin_indices] += t_contrib

    return soft_hist

# For IceCube: 
def hard_hist_to_soft_hist(hard_hist, kernel_calculator, bin_width: float, impact_range: float, window_start: float, window_stop: float):
    """
    Transform a hard histogram to a soft histogram according to the kernel distribution. 
    Do a numerical mid-point integration on every time stamp of the time series. 
    There's an impact range for each bin, and we only add the contribution inside the "valid impact range"
    """

    counts = hard_hist.bins.copy()
    time_edges = np.linspace(hard_hist.xlow, hard_hist.xhigh, hard_hist.nbins + 1)
    time_centres = 0.5 * (time_edges[:-1] + time_edges[1:])
    
    nbins = int(np.ceil((window_stop - window_start) / bin_width))
    soft_hist = np.zeros(nbins, dtype=np.float64)

    # use enumerate then can also get the indices: 
    for i, count in enumerate(counts):
        t = time_centres[i]
        first_impact_value = t - impact_range 
        last_impact_value = t + impact_range 

        # define bin_indice = 0 for window_start 
        # lower limit of the integration: 
        first_impact_value = max(window_start, first_impact_value)
        first_impact_bin = int(np.floor((first_impact_value - window_start) / bin_width))

        # upper limit of the integration:
        last_impact_value = min(window_stop, last_impact_value)
        last_impact_bin = int(np.ceil((last_impact_value - window_start) / bin_width))

        if first_impact_bin >= last_impact_bin:
            continue

        # integration region:         
        impact_range_bin_indices = np.arange(first_impact_bin, last_impact_bin)
        bin_centre_values = window_start + (impact_range_bin_indices + 0.5) * bin_width

        # kernel_calculator is a function, defined by "offset" with respect to a timestamp 
        t_contrib = (kernel_calculator(bin_centre_values - t) * bin_width) * count
        soft_hist[impact_range_bin_indices] += t_contrib

    return soft_hist

class PairTimeSeriesToHist_Smoothing_Toy(Node): 
    def __init__(self, in_ts1_field=None, in_ts2_field=None, out_hist_field=None, bin_width=None, **kwargs):
        if bin_width is None:
            bin_width = kwargs.pop('hist_bin_width', None)
        if bin_width is None:
            bin_width = kwargs.pop('bin_width', None)
        if bin_width is None:
            raise TypeError('bin_width is a compulsory argument in this node, thank you.')

        self.in_ts1_field = in_ts1_field
        self.in_ts2_field = in_ts2_field
        self.in_hist1_field = kwargs.pop('in_hist1_field', None) # In case for IceCube
        self.out_hist_field = out_hist_field
        self.bin_width = float(bin_width)

        self.window_start = kwargs.pop('window_start', None)
        self.window_size = kwargs.pop('window_size', None)
        self.sigma_gnd = kwargs.pop('sigma_gnd', 0.005)
        self.impact_range = kwargs.pop('impact_range', 2.0) # one time stamp can smear to [t_j - r, t_j + r] in the histogram. 
        self.hist1_ic = kwargs.pop('hist1_ic', False) # asking whether we use hist for IceCube or not

        self.cache_hist1 = kwargs.pop('cache_hist1', True) # asking cache or not 
        self._hist1_cache_key = None
        self._hist1_cache = None
        self.det1 = kwargs.pop('det1', None)

        self.lag = kwargs.pop('lag', 0.0)
        self.lag_field = kwargs.pop('lag_field', None)
        self.lags_field = kwargs.pop('lags_field', None)
        super().__init__(**kwargs)

    def reset(self, data):
        self._hist1_cache_key = None
        self._hist1_cache = None
        return True

    def hist1_cache_key(self, det1, data1, window_start, window_size):
        if self.hist1_ic == True: 
            id_time = id(data1.bins)
            data_type = 'hist'
        else: 
            id_time = id(data1.times)
            data_type = 'timeseries'
        
        return (
            det1, 
            id(data1), # id() function can return a unique id for specified object, so somehow the computer can "remember" and "recognise" ts1 now.  
            id_time,
            data_type,
            float(window_start),
            float(window_size),
            self.bin_width,
            self.impact_range,
            self.sigma_gnd,
        )
    
    def define_window(self, data_1, data_2):
        if self.window_start is not None and self.window_size is not None:
            start = self.window_start
            stop = self.window_start + self.window_size
        elif self.hist1_ic == True: # hist vs ts
            start = min(data_1.bins[0], data_2.time[0])
            stop = max(data_1.bins[-1], data_2.times[-1])
        else: # ts vs ts
            start = min(data_1.times[0], data_2.times[0])
            stop = max(data_1.times[-1], data_2.times[-1])

        nbins = int(np.ceil((stop - start) / self.bin_width))
        # redefine stop into one that matches our nbins:
        stop = start + nbins * self.bin_width

        return float(start), float(stop), nbins
    
    def retrieve_lag(self, data):
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

        # direct-provided lag:
        return np.asarray([self.lag], dtype=np.float64)
    
    def alert(self, data):
        if self.hist1_ic == True: 
            hard_hist1, valid = fetch_field(data, self.in_hist1_field)
            if not valid: 
                raise ValueError(f"Hard Hist field from IceCube '{self.in_hist1_field}'is not found")
                return False
        else:
            ts1, valid = fetch_field(data, self.in_ts1_field)
            if not valid:
                raise ValueError(f"Time series 1 field '{self.in_ts1_field}' not found in data.")
                return False
            
        ts2, valid = fetch_field(data, self.in_ts2_field)
        if not valid:
            raise ValueError(f"Time series 2 field '{self.in_ts2_field}' not found in data.")
            return False 

        lags = self.retrieve_lag(data)
        
        # window-setting: 
        if self.hist1_ic == True: 
            window_start, window_stop, nbins = self.define_window(hard_hist1, ts2)
        else: 
            window_start, window_stop, nbins = self.define_window(ts1, ts2)
        if nbins is None or nbins <= 0:
            raise ValueError("Invalid nbins, please adjut bin-width or window parameters.")
            return False

        # These 2 functions will be fed into ts_to_soft_hist / hard_hist_to_soft_hist
        def kernel_contribution_calculator_1(offset):
            return gaussian_noise(offset, self.sigma_gnd)
            
        def kernel_contribution_calculator_2(offset):
            return gaussian_noise(offset, self.sigma_gnd)
        
        # Take the cache_key
        if self.hist1_ic == True: 
            hist1_cache_key = self.hist1_cache_key(self.det1, hard_hist1, window_start, self.window_size)
        else:
            hist1_cache_key = self.hist1_cache_key(self.det1, ts1, window_start, self.window_size)

        # Retrieve hist1 if we have cached it before: 
        # Otherwise, make one by ts_to_soft_hist or hard_hist_to_soft_hist
        if self.cache_hist1 and self._hist1_cache_key == hist1_cache_key and self._hist1_cache is not None:
            hist1 = self._hist1_cache.copy()
        elif self.hist1_ic == False:
            hist1 = ts_to_soft_hist(ts1.times, kernel_contribution_calculator_1, self.bin_width, self.impact_range, window_start, window_stop)
            if self.cache_hist1: 
                self._hist1_cache_key = hist1_cache_key
                self._hist1_cache = hist1.copy()
        else: 
            hist1 = hard_hist_to_soft_hist(hard_hist1, kernel_contribution_calculator_1, self.bin_width, self.impact_range, window_start, window_stop)
            if self.cache_hist1:
                self._hist1_cache_key = hist1_cache_key
                self._hist1_cache = hist1.copy()

        # Store the histograms in the output field:
        pairs = []
        for lag in lags:
            shifted_ts2_times = shift_timeseries(ts2.times, lag) # shift the second time series by the lag.
            hist2 = ts_to_soft_hist(shifted_ts2_times, kernel_contribution_calculator_2, self.bin_width, self.impact_range, window_start, window_stop)
            pairs.append({
                'lag': float(lag),
                'hist1': hist1.copy(), 
                'hist2': hist2,
                'edges': np.linspace(window_start, window_stop, nbins + 1),
                'shifted_edges': np.linspace(window_start, window_stop, nbins + 1),
                'start': float(window_start),
                'stop': float(window_stop),
                'bin_width': float(self.bin_width),
                'real_window_nbins': int(nbins)
            })

        payload = {'pairs': pairs,
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
