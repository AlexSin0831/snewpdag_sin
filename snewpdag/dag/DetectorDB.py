"""
DetectorDB - detector database
"""
import csv
import logging

from . import Detector

class DetectorDB:

  # 逢親擺喺 method 出面嘅東西都係 shared across the whole programme!
  dets = {}
  files = []

  def __init__(self, filename):
    if len(DetectorDB.dets) > 0 and filename in DetectorDB.files:
      # check if already read in this file
      return # 有嘅話就 immediately stop!!!
    
    logging.info('Read detector database file {}'.format(filename))
    DetectorDB.files.append(filename)
    with open(filename, 'r') as f: #read mode
      cr = csv.reader(f)
      for det in cr: #looping the rows # det 其實係一整行
        name = det[0] 
        if len(name) > 0:
          lon = float(det[1])
          lat = float(det[2])
          height = float(det[3])
          sigma = float(det[4])
          bias = float(det[5])
          d = Detector(name, lon, lat, height, sigma, bias) # 要擺返 Detector.py self 嗰個次序
          DetectorDB.dets[name] = d # 到最後先擺返落去 dets 度D

  def has(self, name):
    return name in DetectorDB.dets

  def get(self, name): #used in DiffTimes.py
    if self.has(name):
      return DetectorDB.dets[name]
    else:
      return None

