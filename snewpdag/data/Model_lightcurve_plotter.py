"""
Plot one model lightcurve file.

Run from the repository root:
  python snewpdag/data/plot-model-lightcurve.py
"""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


MODEL_FILE = Path('models/nc-s27-nmo-scint.data')
OUTPUT_FILE = Path('output/nc-s27-nmo-scint-lightcurve.png')


def main():
  data = np.loadtxt(MODEL_FILE, delimiter=',')
  time = data[:, 0]
  rate = data[:, 1]

  OUTPUT_FILE.parent.mkdir(exist_ok=True)

  fig = Figure(figsize=(8, 6))
  canvas = FigureCanvas(fig)

  ax_full = fig.add_subplot(211)
  ax_full.plot(time, rate, lw=1.2)
  ax_full.axvline(0.0, color='tab:red', linestyle='--', lw=1.0,
                  label='t = 0')
  ax_full.set_title('nc-s27-nmo-scint model lightcurve')
  ax_full.set_xlabel('Model time [s]')
  ax_full.set_ylabel('Model bin weight')
  ax_full.legend()

  ax_zoom = fig.add_subplot(212)
  zoom = (time >= -0.05) & (time <= 1.0)
  ax_zoom.plot(time[zoom], rate[zoom], lw=1.2)
  ax_zoom.axvline(0.0, color='tab:red', linestyle='--', lw=1.0)
  ax_zoom.set_title('Zoom: early 1 second')
  ax_zoom.set_xlabel('Model time [s]')
  ax_zoom.set_ylabel('Model bin weight')

  fig.tight_layout()
  canvas.print_png(OUTPUT_FILE)

  print('model:', MODEL_FILE)
  print('time range: {:.6f} to {:.6f} s'.format(time.min(), time.max()))
  print('peak time: {:.6f} s'.format(time[np.argmax(rate)]))
  print('plot:', OUTPUT_FILE)


if __name__ == '__main__':
  main()
