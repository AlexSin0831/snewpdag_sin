"""
CelestialPixels - keep maps of ICRS to GCRS directions (ICRS 係 fixed coordinate system like celestial sphere)

to use, just instantiate and call get_map().
If the same (nside,time) is requested, where time is a Unix timestamp to
integer precision (any fractional part is lopped off), then this will
just return one that was created before.
"""
import logging
import numpy as np
import healpy as hp
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import GCRS, SkyCoord, CartesianRepresentation

class CelestialPixels:

  maps = {} # the cache in this class 
  # Note that this cache is not defined inside the __int__ method! 
  # Maps 係一個遊走於整個 code 嘅 dictionary！

  def __init__(self):
    pass # lazy class

  def delete_all_maps(self):
    CelestialPixels.maps = {}

  def list_maps(self):
    return CelestialPixels.maps.keys()

  def get_map(self, nside, time):
    """
    Get an array of unit vectors pointing to ICRS skymap pixel centers.
    Arrays are keyed with nside and time.
    If no such array already exists, create one.
    nside = healpix resolution.
    time = Unix timestamp.  Only kept at second granularity.
    """
    time_tag = int(time)
    tag = (nside, time_tag)
    if tag in CelestialPixels.maps:
      return CelestialPixels.maps[tag]

    # need to create a map
    t = Time(time_tag, format='unix') # unix 嘅 time format 係唔 work 的 所以我地要將佢變成 astropy 嘅 format！
    npix = hp.nside2npix(nside) # npix = 12 x nside^2
    # pixel centers in ICRS coordinates.
    # c will an array of lon,lat with shape (2,npix).
    # c = (
    #     [lon_0, lon_1, lon_2, lon_3, ... , lon_last],  # This entire array is c[0]
    #     [lat_0, lat_1, lat_2, lat_3, ... , lat_last]   # This entire array is c[1]
    #     )
    c = hp.pixelfunc.pix2ang(nside, range(npix), nest=True, lonlat=True) # 呢個係 centre of the pixel
    sc = SkyCoord(ra=c[0], dec=c[1], unit=u.deg, frame='icrs', \
                  representation_type='unitspherical', obstime=t)
    gc = sc.transform_to(GCRS)
    # gc is now an array of SkyCoord, but in (ra,dec) in GCRS
    n = gc.represent_as(CartesianRepresentation)
    # n is an array of (x,y,z) unit vectors
    rs = np.stack( (n.x, n.y, n.z) ) # shape (3,npix)
    rs.flags.writeable = False
    CelestialPixels.maps[tag] = rs
    return rs

  def delete_map(self, nside, time):
    time_tag = int(time)
    tag = (nside, time_tag)
    if tag in CelestialPixels.maps:
      del CelestialPixels.maps[tag]

