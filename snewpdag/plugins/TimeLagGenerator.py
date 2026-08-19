"""
TimeLagGenerator: Generate lag grids for coarse or fine likelihood scans.

configuration: 
    out_field       : field name of the time lags

    (Optional:)
    scan_low        : lower limit of the scanning range
    scan_high       : higher limit of the scanning range 
    scan_centre     : centre of the scanning region
    scan_half_width : span / 2 of the scanning region
    step_size       : step size of scanning
    coarse_step     : step size of coarse-scanning 
    fine_step       : step size of fine-scanning
    scan_type       : coarse / fine 
    in_roi_field    : field name of the fine scan region 
"""

import logging
import numbers

import numpy as np
import scipy.special as sc

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field


def making_grids(low, high, step_size):
  if step_size <= 0.0:
    raise ValueError('step must be positive')
  if high < low:
    raise ValueError('scan high must be >= scan low')
  
  number_of_steps = int(np.floor((high - low) / step_size + 1.0e-12)) 
  values_array = low + step_size * np.arange(number_of_steps + 1)

  if len(values_array) == 0 or values_array[-1] < high - 1.0e-12:
    values_array = np.append(values_array, high)
  else:
    values_array[-1] = min(values_array[-1], high)
  
  return values_array

class TimeLagGenerator(Node):
  def __init__(self, out_field, **kwargs):
    self.out_field = out_field
    self.scan_low = kwargs.pop('scan_low', None)
    self.scan_high = kwargs.pop('scan_high', None)
    self.scan_centre = kwargs.pop('scan_centre', None)
    self.scan_half_width = kwargs.pop('scan_half_width', None)

    # for simple testing:
    self.step_size = kwargs.pop('step_size', kwargs.pop('step', None))

    # for more general workflow: 
    self.coarse_step = kwargs.pop('coarse_step', None)
    self.fine_step = kwargs.pop('fine_step', None)
    self.scan_type = kwargs.pop('scan_type', 'coarse')
    self.in_roi_field = kwargs.pop('in_roi_field', None)
    super().__init__(**kwargs)

  def define_bounds(self, data):
    # redefine roi for fine_scan:
    if self.in_roi_field is not None:
      roi, valid = fetch_field(data, self.in_roi_field)
      if valid:
        if isinstance(roi, dict):
          return roi['low'], roi['high']
        return roi[0], roi[1]
    
    # scan_centre case: 
    if self.scan_centre is not None and self.scan_half_width is not None:
      return self.scan_centre - self.scan_half_width, self.scan_centre + self.scan_half_width
    
    # simple cases:
    if self.scan_low is not None and self.scan_high is not None:
      return self.scan_low, self.scan_high

  def alert(self, data):
    real_scan_low, real_scan_high = self.define_bounds(data)

    if real_scan_low is None or real_scan_high is None:
      return False
    

    real_step_size = self.step_size
    if real_step_size is None:
      real_step_size = self.fine_step if self.scan_type == 'fine' else self.coarse_step
    
    # didn't provide all kinds of step_size
    if real_step_size is None:
      return False
    
    time_lags = making_grids(float(real_scan_low), float(real_scan_high), float(real_step_size))
    store_field(data, self.out_field, time_lags)
    return True
