#!/usr/bin/env python
# code for fitting lagged histograms
import platform
host=platform.node()
from scipy.stats import norm
import matplotlib.mlab as mlab
import json
import random as rndm
import matplotlib.pyplot as pltlag
import matplotlib.pyplot as pltscanrange
import matplotlib.pyplot as plterr
import matplotlib.pyplot as pltshift 
from timeit import default_timer as timer
import math
import sys
#from scipy import special as sp
import scipy as sp
import scipy.special as spc
from scipy.stats import poisson
from scipy.optimize import minimize, curve_fit
import numpy as np
import os

os.chdir("/home/azfar/python/snewpdag")
start=timer()

# Definition of fit functions, background levels, bin widths and cuts on time
# these very variables values are also included in png file name 
# the larger data sample is has b in associated descriptions eg datab, filenameb
# etc the smaller has c. Thus this defines the scale factor So please larger
# filename first 
# Useage: 
# "python3 queue_version_file_number_arguments.py biggerfilename smallerfilename"
# Fit functions

def Gaussian(xval, *p):
    a, mu, sigma = p
    return a * math.exp(-(xval - mu)**2.0 / (2 * sigma**2))

gaussian = np.vectorize(Gaussian)
BADVALUE = 99999999.0

# initialize though I should be and have missed initializing other stuff
lagfit_at_max_llk=BADVALUE

# where to truncate the time series we started with 5 ...
truncate =5.0

num_bins = 150 #\\\\\xs this gives 25 millisecond bins 
low_edge = -2.0   # low edge for both histograms
high_edge = truncate-1.0 # 20 seconds - will change after discussion with Jeff
bin_size = (high_edge-low_edge)/num_bins

# Background rates events per millisecond 
bb=0.0001 # events per millisecond for IC 0.0003 for SUPERK 0.0001 for SNOP
bc=0.0003 # events per millisecond 
# bc = 0.0003 for SUPERK 0.0001 for SNOP

# set scan window size 
scan_window = 0.4 # scan window
scan_steps = 400 #  .2/200 1 millisecond steps

scan_step_size = scan_window/scan_steps

# read in arguments
filenameb = sys.argv[1]
filenamec = sys.argv[2]

print(" Running " + sys.argv[0] + " on " + host) 
print(" File Names are : " + filenameb + " " + filenamec)
print("Python version " + sys.version)


# the detector names 
cdetname = ""
bdetname = ""



if "-IC-" in filenameb:
    bdetname = "IC"
elif "-SNOP-" in filenameb:
    bdetname = "SNOP"
elif "-SK-" in filenameb:
    bdetname = "SK"
else:
    bdetname = "ERROR"


if "-IC-" in filenamec:
    cdetname = "IC"
elif "-SNOP-" in filenamec:
    cdetname = "SNOP"
elif "-SK-" in filenamec:
    cdetname = "SK"
else:
    cdetname = "ERROR"

# use the file path and name to generate the bits needed
# to create the png filenames 
tmpb = filenameb.split("/")[len(filenameb.split("/"))-1]
tmpc = filenamec.split("/")[len(filenamec.split("/"))-1]

tmpb = tmpb.replace("-0.json","")
tmpc = tmpc.replace("-0.json","")
tmpb = tmpb.replace("json-","")
tmpc = tmpc.replace("json-","")
tmpb = tmpb.replace("nlog-dump-","")
tmpc = tmpc.replace("nlog-dump-","")



prename=tmpb+"-"+tmpc
print(" Prename : " + prename)

# file name for illustrations
prename += "-cut" + str(truncate)
prename += "-binw" + str(bin_size) + "-b" + str(bb) + "-c" + str(bc)+ '-step' + str(scan_step_size)
predir = bdetname+"-"+cdetname
# on pplxintX machines this is the output area - 
output_prepath = "/home/azfar/forolly/output"
#output_prepath = "/home/azfarl/snewpdag/analysis/outest2/"
#output_prepath = "/home/azfarl/snewpdag/analysis/outest3/"
output_dir = output_prepath + predir + "-binw" + str(bin_size) + "-b" + str(bb) + "-c" + str(bc)+'-step' + str(scan_step_size)


if os.path.isdir(output_dir):
    print(output_dir, "exists")
    #sys.exit(1)
else:
    os.mkdir(output_dir)


# use events / millisecond to convert to actual numbers
nbb = int((truncate-low_edge)*1000*bb)
nbc = int((truncate-low_edge)*1000*bc)



# arrays to store lag values and likelihoods at each step of scan
lag_values = np.zeros(scan_steps)
llk_values = np.zeros(scan_steps)



def scan_errors(lags, logls):
    imaxlog = logls.argmax()
    normedlogs = logls - logls[imaxlog]

    x=[lags[m] for m in range(imaxlog-5, imaxlog+6)]
    y=[normedlogs[m] for m in range(imaxlog-5, imaxlog+6)]    
    
    p4 = np.polynomial.Polynomial.fit(x,y,4)
    roots = (p4.deriv()).roots()
    
    print(" roots ", roots)
    print(" 2nd derivative ", (p4.deriv()).deriv()(roots[1]))
    inverr = (p4.deriv()).deriv()(roots)

    return math.sqrt(abs(1.0/inverr))

# prepare figure 
figlag, [axlags, axlagf] = pltlag.subplots(2,1)
figscanrange, [axscanlo, axscanhi] = pltscanrange.subplots(2,1)
figshift, [axshiftu, axshifta] =pltshift.subplots(2,1)

    
# loop over all files - generate their names 

    
fb = open(filenameb)
bdata = json.load(fb)
bstart = min(bdata['timeseries']['times'])

fc = open(filenamec)
cdata = json.load(fc)
    # note we are subtracting the superk event start from these as well -
    # we need the absolute times     # note we are subtracting the superk event start from these as well we need absolute times
b_event_times_pre = [time-bstart for time in bdata['timeseries']['times']]
c_event_times_pre = [time-bstart for time in cdata['timeseries']['times']]


n_c = len(c_event_times_pre) # size of array c
n_b = len(b_event_times_pre) # size of array b



# should this not be determined from the file itself ? 
cstart_true = cdata['truth']['dets'][cdetname]['true_t']
bstart_true = bdata['truth']['dets'][bdetname]['true_t']

true_value = cstart_true-bstart_true

print(" True lag value " + str(true_value))

scale = float(n_c)/(float(n_b) + float(num_bins)) # ratio of smaller to larger detector ... 
print(' Scale = ', scale)
print(' Number of C before truncation ', n_c)
print(' Number of B before truncation ', n_b)
    
# now select for reference histograms
b_event_times = [time for time in b_event_times_pre if time <= truncate]
print(' Number of ' + bdetname + ' after truncation ', len(b_event_times))
# generated background between the low edge and high edge of histogram
# so here we have taken the number of superk events below 5 (original numb) 
nb_cut = len(b_event_times)
# we now generate a random background between the low and high edges of the
# histogram, the background is generated over a range larger than
# the extent of the signal -we scan upto LESS than "truncate" 
back_b = [rndm.uniform(low_edge, truncate) for iii in range(0, nbb)]
# append the background 
b_event_times += back_b

# make the reference histogram ie the one with respect to which the other
# will be lagged 
tsk, tsk_e = np.histogram(b_event_times, num_bins, [low_edge, high_edge])

# select snop times for times less than 5 (original number)
c_event_times = [time for time in c_event_times_pre if time <= truncate]
print(' Number of ' + cdetname + ' after truncation ', len(c_event_times))

# note again the background is generated over a range larger than
# the extent of the signal -we scan upto LESS than "truncate" 
back_c = [rndm.uniform(low_edge, truncate) for iii in range(0, nbc)]
c_event_times += back_c


# define constants for use in loop
a = scale

# backgrounds
b=bin_size*nbb/(high_edge-low_edge)
c=bin_size*nbc/(high_edge-low_edge)

# set up everything assuming c/a-b is positive
lowerIncompGammaLim = b*(1+a)
factorToPowerJ = (c/a-b)*(1+a)


# now check if it is not positive
negativeFactor = (b-c/a)>=0

if negativeFactor:
    lowerIncompGammaLim = c*(1+a)/a
    factorToPowerJ = (b-c/a)*(1+a)
    
    
# define scan window 
ts = -scan_window/2.0
for i in range(0, scan_steps):
    t = ts + i*(scan_window/(scan_steps))
        
    # and now shift them 
    shifted_times = [time + t for time in c_event_times]


        
    # these are the shifted times being popped into a histogram
    lsh, lshe = np.histogram(shifted_times, num_bins, [low_edge, high_edge])


    

    # plot shifted histograms for the extremes of scan range
    if i==0:
        edges = 0.5*(lshe[1:]+lshe[:-1])
        axscanlo.set_yscale("log")
        axscanlo.plot(edges,lsh,color="red")
        axscanlo.plot(edges,tsk,color="blue")
    if i==scan_steps-1:
        edges = 0.5*(lshe[1:]+lshe[:-1])
        axscanhi.set_yscale("log")
        axscanhi.plot(edges,lsh,color="red")
        axscanhi.plot(edges,tsk,color="blue")
        

                         
# prepare for loop 
    loglik = 0.0


    for k in range(0, num_bins):
        func=0.0
        m=lsh[k]
        n=tsk[k]

        internalSum = m
        binomLower = n
        
        if negativeFactor:
            internalSum = n
            binomLower = m
 
        logfuncarray=[]
        for j in range(0, internalSum+1):
            prefunc = sp.special.binom(n+m-j,binomLower)*sp.special.gammaincc(m+n-j+1,lowerIncompGammaLim)
            logprefunc = sp.special.loggamma(n+m-j+1)-sp.special.loggamma(binomLower+1)-sp.special.loggamma(n+m-j-binomLower+1)
            logprefunc += math.log(sp.special.gammaincc(m+n-j+1,lowerIncompGammaLim))
            # protection against 0 background 
            logpow=0
            if prefunc !=0:
                if factorToPowerJ !=0:
                    logpow = j*math.log(factorToPowerJ)
                loggamma = sp.special.loggamma(j+1)
                logfunc = logpow-loggamma + logprefunc
                logfuncarray.append(np.float128(logfunc))
#                func += pow(factorToPowerJ,j)*prefunc/sp.special.gamma(j+1)
#                func +=math.exp(logfunc)
#                func +=math.exp(sp.special.logsumexp([logpow,-loggamma,logfunc]))
                 
        loglik += sp.special.logsumexp(logfuncarray)
            
    lag_values[i]=t
    llk_values[i]=loglik
        

# array is now full compute maximum using numpy polynomial class
# and also find maximum of the array 
max_index = llk_values.argmax()
max_t = lag_values[max_index]


# one could select fewer of them ...
#select_lag = [lag_values[m] for m in range(max_index-45, max_index+45)]
#select_llk = [llk_values[m] for m in range(max_index-45, max_index+45)]
select_lag = lag_values
select_llk = llk_values
wl = min(select_lag)
wh = max(select_lag)
p9 = np.polynomial.polynomial.Polynomial.fit(select_lag,select_llk,9,domain=[wl,wh], window=[wl,wh])
# p5e = np.polynomial.polynomial.Polynomial.fit(lag_values,llk_values,5)
xp = np.linspace(p9.domain[0],p9.domain[1],1000)
#    print(" First derivative -data ", p5.deriv()(max_t))

# find all real roots of the first derivative 
realroots=[rr for rr in (p9.deriv()).roots() if np.iscomplex(rr)==False]
rootFound = len(realroots)> 0
    
closest_value = BADVALUE

if rootFound:
    closest_value = min(realroots, key=lambda rr: abs(max_t - rr))
    closest_value = np.real(closest_value)
    closest_value= float(closest_value)
    lagfit_at_max_llk=closest_value
    
        
print(" Lag at max likelihood ", max_t)
print(" real roots ", realroots)
print(" closest_value ", closest_value)
print(" 1st derivative ", p9.deriv()(closest_value))
print(" 2nd derivative ", (p9.deriv()).deriv()(closest_value))
closest_value = np.real(closest_value)
closest_value= float(closest_value)
ddt1 = p9.deriv()(closest_value)
ddt2 = (p9.deriv()).deriv()(closest_value)
ddt3 = ((p9.deriv()).deriv()).deriv()(closest_value)
ddt4 = (((p9.deriv()).deriv()).deriv()).deriv()(closest_value)
ddt5 = ((((p9.deriv()).deriv()).deriv()).deriv()).deriv()(closest_value)
ddt6 = (((((p9.deriv()).deriv()).deriv()).deriv()).deriv()).deriv()(closest_value)
ddt7 = ((((((p9.deriv()).deriv()).deriv()).deriv()).deriv()).deriv()).deriv()(closest_value)
ddt8 = (((((((p9.deriv()).deriv()).deriv()).deriv()).deriv()).deriv()).deriv()).deriv()(closest_value)

# first initialize fit_error
fit_error=BADVALUE
if rootFound:
    fit_error =math.sqrt(abs(1.0/((p9.deriv()).deriv()(closest_value))))
else:
    fit_error = BADVALUE
    # log a bad fit to see what it looks like 
    figerr, axerr = plterr.subplots(1,1)
    axerr.plot(lag_values, llk_values)
    axerr.plot(xp, p9(xp),color="red")
    axerr.axvline(max_t, color="red")
    figname = "polyfiterror" + + ".png"
    figerr.savefig(output_dir + "/" + prename+figname)

# show the likelihood profile and best time with fit and
# with just scanning 
axlags.plot(lag_values, llk_values)
axlags.axvline(max_t, color="red")
axlagf.plot(lag_values, llk_values)
axlagf.plot(xp, p9(xp),color="red")
axlagf.axvline(closest_value, color="red")



plottimes = [time + max_t for time in c_event_times]
lp, lpe = np.histogram(plottimes, num_bins, [0.0, 0.2])
# make another copy with finer binning
tb, tb_e = np.histogram(b_event_times, num_bins, [0.0, 0.2])
pedges =0.5*(lpe[1:] + lpe[:-1])
axshifta.set_yscale("log")
axshifta.plot(pedges,lp,color="red")
axshifta.plot(pedges,tb,color="blue")   
# make a histogram to display the shift - you have to bin pretty fine so
# as to see shift 
shft, shft_e = np.histogram(c_event_times, num_bins, [0., 0.2])
edges = 0.5*(shft_e[1:]+shft_e[:-1])
axshiftu.set_yscale("log")
axshiftu.plot(edges,shft,color="red")
axshiftu.plot(edges,tb,color="blue")
# post run processing - clean out bad values





# print stats 
        

figlag.savefig(output_dir + "/" + prename + "llscan.png")
figscanrange.savefig(output_dir + "/" + prename + "displayscanrange.png")
figshift.savefig(output_dir + "/" + prename + "shiftedback.png")
closest_value = np.real(closest_value)
closest_value= float(closest_value)
print("Type of closest value " + str(type(closest_value)))
print("Type of closest value " + str(type(closest_value)))
# dump json
dictionary = {
    "host": host,
    "big_det": bdetname,
    "small_det":cdetname,
    "scan_step": scan_step_size,
    "bin_size": bin_size,
    "bback": bb,
    "cback": bc,
    "scan_lag": max_t,
    "fit_lag": closest_value,
    "fit_error": fit_error,
    "scan_pull": (max_t + true_value)/fit_error,
    "fit_pull": (closest_value+true_value)/fit_error,
    "true_lag": true_value,
    "coeff0": p9.coef[0],
    "coeff1": p9.coef[1],
    "coeff2": p9.coef[2],
    "coeff3": p9.coef[3],
    "coeff4": p9.coef[4],
    "coeff5": p9.coef[5],
    "coeff6": p9.coef[6],
    "coeff7": p9.coef[7],
    "coeff8": p9.coef[8],
    "coeff9": p9.coef[9],
    "ddt1": ddt1,
    "ddt2": ddt2,
    "ddt3": ddt3,
    "ddt4": ddt4,
    "ddt5": ddt5,
    "ddt6": ddt6,
    "ddt7": ddt7,
    "ddt8": ddt8
}

json_object = json.dumps(dictionary, indent=4)
with open(output_dir + "/" + prename + "fits.json", "w") as outfile:
    outfile.write(json_object)

end= timer()

time_in_seconds = end-start
time_in_minutes = int(time_in_seconds/60)
time_in_hours = int(time_in_seconds/3600)

print("Time taken: " + str(time_in_hours)+ " Hours, " + str(time_in_minutes) + " Minutes, " + str(time_in_seconds) + " Seconds")