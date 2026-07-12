import csv
import numpy as np
import matplotlib.pyplot as plt

dels = ','
tt = [] # lower edges of time bins
nn = [] # number of events in this time bin
with open('/home/tseng/dev/snews/numodels/Bollig_2016/ibd-s27-nmo-wc.data', newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter='\t' if dels == '' else dels)
  for row in reader:
    tt.append(float(row[0]))
    nn.append(float(row[1]))

t = np.array(tt[:-1])
t1 = np.array(tt[1:])
dt = t1 - t
thi = tt[-1]
# use the last bin edge as the maximum.
# This amounts to cutting off the last n element.
mu = np.array(nn[:-1]) # number of events in the bin
rate = mu / dt # rate

fig, ax = plt.subplots()
ax.plot(t, rate)
ax.grid()
plt.savefig('ibd-s27-nmo-wc.png')

# scintillator

tt = [] # lower edges of time bins
nn = [] # number of events in this time bin
with open('/home/tseng/dev/snews/numodels/Bollig_2016/ibd-s27-nmo-scint.data', newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter='\t' if dels == '' else dels)
  for row in reader:
    tt.append(float(row[0]))
    nn.append(float(row[1]))

t_s = np.array(tt[:-1])
t1 = np.array(tt[1:])
dt = t1 - t
thi = tt[-1]
# use the last bin edge as the maximum.
# This amounts to cutting off the last n element.
mu = np.array(nn[:-1]) # number of events in the bin
rate_s = mu / dt # rate

print('t   = {}'.format(t[:5]))
print('t_s = {}'.format(t_s[:5]))

fig, ax = plt.subplots()
ax.plot(t, rate)
ax.plot(t_s, rate_s)
ax.grid()
plt.savefig('ibd-s27-nmo-scint.png')

fig, ax = plt.subplots()
ax.plot(t, rate / rate_s)
ax.grid()
plt.savefig('ibd-s27-nmo-ratio.png')
plt.show()

