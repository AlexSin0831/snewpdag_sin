"""
DAG library routines
"""
import logging
import numbers
import numpy as np

# deprecated: ns_per_second
# deprecated: time_tuple_from_float(x)
# deprecated: time_tuple_from_offset(ns)
# deprecated: time_tuple_from_string(s)
# deprecated: time_tuple_from_field(s)
# deprecated: offset_from_time_tuple(tt)
# deprecated: normalize_time(a)
# deprecated: normalize_time_difference(a)
# deprecated: subtract_time(a, b)

def fetch_field(data, fields): # Always return 2 things (Value? + Valid?) 
  """
  Fetch a field from the payload (data).
  If the field is a string, attempt to split by /'s.
    (Note: don't leave a trailing slash! It looks like an empty field name)
  If the field is list-like, interpret each element as the field name
    in each inner dictionary.
  Return value (or None), and True/False depending on field(s) existing.
  """
  # isinstance(object, type): If object is really that data type, then return True. Otherwise, return False
  # 如果係string先用 / 去分
  # 如果係 list or tuple， 我地就用
  fs = fields.split('/') if isinstance(fields, str) else fields # fs := field specifier 
  # return a list storing smaller strings, e.g. keys to a dictionary? 
  # list can store mixed types of stuff, but array can only store the same type
  if isinstance(fs, (list, tuple)):
    d = data
    for f in fs:
      if isinstance(d, dict) and f in d: # 專登 check 下 data 裏面係咪真係有呢個 key
        d = d[f] # 不斷縮細我哋嘅搜索範圍
      elif isinstance(d, (list, tuple, np.ndarray)) and f < len(d): # 睇吓係咪真係 within the range of the list/tuple/array
        d = d[f]
      else:
        return None, False 
    return d, True
  else:
    if fs in data:
      return data[fs], True
    else:
      return None, False

def store_field(data, field, value): 
  fs = field.split('/') if isinstance(field, str) else field
  if isinstance(fs, (list, tuple)):
    d = data
    for f in fs[:-1]: # let's say we have fs = [a,b,c,d,e,f], then fs[:-1] = [a,b,c,d,e] 
      # 我地整走最尾嗰個 key 因為 that's where we want to store our stuff!
      if isinstance(d, dict) and f in d:
        d = d[f] 
      else:
        d[f] = {} #有機會係喺中途退出 ==> 新開左一個branch inside the parent dictionary
        d = d[f] #then we keep building things inside the newly opened branch 
    d[fs[-1]] = value # 去到最後先擺返個 value 落去
  else:
    data[fs] = value
  return True

def fill_filename(pattern, module_name, count, data):
  """
  Get filename, and fill out the details.
  pattern = '[field specifier]' - fetch pattern from payload using fetch_field.
    'filename' - use this literal string as the pattern.
  The pattern is filled as follows:
    {0} - module_name (plugin 嘅名)
    {1} - count 
    {2} - data['burst_id']
  """
  ps = pattern.strip() # Removes extra spaces from the beginning/end of the pattern. 
  # Special case: 
  if ps[0] == '[' and ps[-1] == ']':
    # Treat pattern as the key to get the real format of our filename
    s, valid = fetch_field(data, ps[1:-1]) # Remove [] and search the value from the payload
    if not valid:
      return None
    ps = s.strip()
  fn = ps.format(module_name, count, data.get('burst_id', 0)) 
  # .get('key', value) 點用？
  # If data has key 'burst_id', return data['burst_id'].
  # Otherwise, return 0.
  return fn

