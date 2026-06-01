"""
Detector - detector instance
"""
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

class Detector:
  def __init__(self, name, lon, lat, height, sigma, bias):
    self.name = name
    self.lon = lon # degrees
    self.lat = lat # degrees
    self.height = height # [m]
    self.sigma = sigma # time resolution [s]
    self.bias = bias # time bias [s], observed - true (通常要遲幾多先會真係 度到第一個 singal)
    delta = np.radians(90.0 - lat) # 由上邊撥落去
    alpha = np.radians(lon)
    self.r = np.array([ np.sin(delta) * np.cos(alpha),
                        np.sin(delta) * np.sin(alpha),
                        np.cos(delta) ])
    self.loc = EarthLocation(lon=lon, lat=lat) # 注意看：EarthLocation 係由 astropy 度彈出嚟 ==> self.loc 已經係一個 astropy 嘅 object!
  
  # Time-dependent!!!!! 因為地球不斷轉緊 ==> 個 detector 嘅 pointing direction 都會不斷轉緊！
  def get_gcrs(self, obstime):
    #k = EarthLocation.of_site('keck')
    t = Time(obstime) # make sure it's in astropy Time form
    g = self.loc.get_gcrs(obstime=t) # g.ra and g.dec # some functions from astropy # 同一個名。。。
    return g

  # The 3D vector yielded by this method is relative to the starry room's (celestial sphere) coordinate system
  def get_xyz(self, obstime): 
    g = self.get_gcrs(obstime)
    radius = np.sqrt(self.loc.x**2 + self.loc.y**2 + self.loc.z**2)
    codelta = np.radians(g.dec)
    alpha = np.radians(g.ra)
    sphi = np.sin(alpha)
    cphi = np.cos(alpha)
    ctheta = np.sin(codelta)
    stheta = np.cos(codelta)
    return radius * np.array([ stheta * cphi, stheta * sphi, ctheta ])

