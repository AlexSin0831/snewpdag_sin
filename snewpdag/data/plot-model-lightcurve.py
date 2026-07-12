"""
Plot a model lightcurve file.

Run from the repository root:
  python snewpdag/data/plot-model-lightcurve.py

Or choose another model:
  python snewpdag/data/plot-model-lightcurve.py models/ibd-s27-nmo-wc.data
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def output_path_for(model_file, output_dir):
  return output_dir / '{}-lightcurve.png'.format(model_file.stem)


def plot_lightcurve(model_file, output_file):
  data = np.loadtxt(model_file, delimiter=',')
  time = data[:, 0]
  rate = data[:, 1]

  output_file.parent.mkdir(exist_ok=True)

  fig = Figure(figsize=(8, 6))
  canvas = FigureCanvas(fig)

  ax_full = fig.add_subplot(211)
  ax_full.plot(time, rate, lw=1.2)
  ax_full.axvline(0.0, color='tab:red', linestyle='--', lw=1.0,
                  label='t = 0')
  ax_full.set_title('{} model lightcurve'.format(model_file.stem))
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
  canvas.print_png(output_file)

  return {
      'time_min': float(time.min()),
      'time_max': float(time.max()),
      'peak_time': float(time[np.argmax(rate)]),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_file', nargs='?',
                      default='models/nc-s27-nmo-scint.data')
  parser.add_argument('--output-dir', default='output')
  args = parser.parse_args()

  model_file = Path(args.model_file)
  output_dir = Path(args.output_dir)
  output_file = output_path_for(model_file, output_dir)
  stats = plot_lightcurve(model_file, output_file)

  print('model: {}'.format(model_file))
  print('time range: {:.6f} to {:.6f} s'.format(
      stats['time_min'], stats['time_max']))
  print('peak time: {:.6f} s'.format(stats['peak_time']))
  print('plot: {}'.format(output_file))


if __name__ == '__main__':
  main()
