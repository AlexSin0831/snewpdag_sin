"""
GenPoint - generate times and time differences for a given SN direction

Arguments:
  detector_location: filename of detector database for DetectorDB
  pair_list: list of detector pairs for which to generate dts
    (optional, default empty, in which case only generate core bounce times)
  ra: right ascension (degrees) (of the supernova)
  dec: declination (degrees) (of the supernova)
  time: time string, e.g., '2021-11-01 05:22:36.328',
        indicating time neutrino wavefront arrives at center of Earth (default in the paper)
  smear: (optional, default True) whether to smear output times
  epoch_base (optional): starting time for epoch, float value or field specifier
    (string or tuple)

ra,dec in ICRS coordinate system.

Input:
  [epoch_base]: float value of starting time for epoch, in unix epoch

Output:
  truth/sn_ra: right ascension (radians), ICRS
  truth/sn_dec: declination (radians), ICRS
  truth/time_center: arrival time at center of Earth, astropy.Time object
  truth/dets/<det>/true_t: arrival time of core bounce for det, no bias
  dts/(<det1>,<det2>)/dt: time difference between detectors
  dts/(<det1>,<det2>)/t1: arrival time for det1
  dts/(<det1>,<det2>)/t2: arrival time for det2
  dts/(<det1>,<det2>)/bias: bias1 - bias2 (sec)
  dts/(<det1>,<det2>)/var: combined variance (sec^2)
  dts/(<det1>,<det2>)/dsig1: sigma1 (sec)
  dts/(<det1>,<det2>)/dsig2: -sigma2 (sec)
"""
import logging
import numbers
import numpy as np
import healpy as hp
from astropy.time import Time
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import GCRS, SkyCoord, CartesianRepresentation

from snewpdag.dag import Node, Detector, DetectorDB
from snewpdag.dag.lib import fetch_field

class GenPoint(Node):
  def __init__(self, detector_location, ra, dec, time, **kwargs):
    # the detectors we chose for the MC trial 
    self.db = DetectorDB(detector_location) # detector_location is a filename?
    self.pairs = kwargs.pop('pair_list', ())
    self.ra = np.radians(ra)
    self.dec = np.radians(dec)
    self.tc = Time(time) # convert it into a Astropy time object ==> contain much richer information 
    self.tc_unix = self.tc.to_value('unix', 'long') # float, unix epoch
    # used for shifting the whole time series, i.e. re-define the zero ==> the unix time number can be smaller 
    self.epoch_base = kwargs.pop('epoch_base', 0.0) # default as 0

    if not isinstance(self.epoch_base, (numbers.Number, str, list, tuple)):
      logging.error('GenPoint.__init__: unrecognized epoch_base {}. Set to 0.'.format(self.epoch_base))
      self.epoch_base = 0.0

    self.smear = kwargs.pop('smear', True)

    # sc is a rich Astropy object, it contains more information than just storing the celestial sphere coordinates 
    sc = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame='icrs', \
                  representation_type='unitspherical', obstime=self.tc)
    # therefore, we can actually convert the "same stuff" into another language, i.e. geocentric coordinates
    gc = sc.transform_to(GCRS)
    d = gc.represent_as(CartesianRepresentation)
    # snr is pointing from the Earth to the supernova!
    self.snr = np.array( [ d.x, d.y, d.z ] ) # should be unit length! 
    logging.info('ra(lon) = {}, dec(lat) = {}'.format(self.ra, self.dec))
    logging.info('SN location = {}'.format(self.snr))
    self.dets = DetectorDB.dets.keys() # because self.db is run above, so the DetectorDB only saves the detectors we are interested in
    super().__init__(**kwargs) 

  def alert(self, data):
    # record truth information
    if 'truth' not in data:
      data['truth'] = {}
    if 'dets' not in data['truth']:
      data['truth']['dets'] = {}
    data['truth']['sn_ra'] = self.ra # radians
    data['truth']['sn_dec'] = self.dec # radians
    data['truth']['time_center'] = self.tc # astropy.Time object # in what unit? 

    # calculate arrival of SN time at Earth center in local epoch
    # i.e. subtracting epoch_base
    if isinstance(self.epoch_base, numbers.Number):
      t_epoch = self.epoch_base
    elif isinstance(self.epoch_base, (str, list, tuple)):
      t_epoch, valid = fetch_field(data, self.epoch_base)
      if not valid:
        logging.error('{}: epoch_base field {} not found in payload'.format(self.name, self.epoch_base))
        return False
    else:
      logging.error('{}: unrecognized epoch_base field {}'.format(self.name, self.epoch_base))
      return False
    tc_local = self.tc_unix - t_epoch #shift the time 

    # generate times for each detector, including bias.
    # given time is when wavefront arrives at Earth origin.
    ts = {}
    bias = {}
    sigma = {}
    for dname in self.dets:
      #c = 3.0e8 # m/s
      det = self.db.get(dname) # det is the whole detector object ==> we can call some useful methods / info from it conveniently
      # (0,0,0) is the centre of the Earth
      pos = det.get_xyz(self.tc) # detector in GCRS at given time (a method in Detector.py) 
      logging.info('pos[{}] = {}'.format(dname, pos))
      logging.info('  sn pos = {}'.format(self.snr))
      # note that get_xyz depends on observed time 
      dt = - np.dot(det.get_xyz(self.tc), self.snr) / const.c # intersect
      logging.info('  dt before bias = {}'.format(dt))
      # store unbiased time in data['truth']
      tcb = tc_local + dt.to(u.s).value # convert to second-scale and then remove the unit
      data['truth']['dets'][dname] = { 'true_t': tcb }

      # apply bias and smear
      dt += det.bias * u.s # u.s ==> second as the unit
      if self.smear:
        dt += det.sigma * Node.rng.normal() * u.s # smear (s) # random number generator w.r.t. the normal distribution
      logging.info('  biased/smeared dt = {}'.format(dt))
      ts[dname] = tc_local + dt.to(u.s).value # 去到呢到先 remove unit!
      bias[dname] = det.bias
      sigma[dname] = det.sigma
      logging.info('  time[{}] = {}'.format(dname, ts[dname]))

    # generate pair times
    # this part will only be run when we intentionally provide the pair list to this plugin 
    if len(self.pairs) > 0:
      dts = {}
      for p in self.pairs:
        dt = ts[p[0]] - ts[p[1]]
        logging.info('dt[{}] = {}'.format(p, dt))
        dts[p] = {
                   'dt': dt, # s
                   't1': ts[p[0]], # s
                   't2': ts[p[1]], # s
                   'bias': bias[p[0]] - bias[p[1]], # sec
                   'var': sigma[p[0]]**2 + sigma[p[1]]**2, # sec**2
                   'dsig1': sigma[p[0]], # sec
                   'dsig2': - sigma[p[1]], # sec
                 }
      data['dts'] = dts

    return data

