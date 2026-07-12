"""NumPy interface to the compiled Poisson-recursion kernel."""

import numpy as np


def buildLogJTable(nmax, mmax, a, b, p, q):
  """Return the float64 ln(J_nm) table used by PoissonLagLikelihood_v4."""
  nmax = int(nmax)
  mmax = int(mmax)
  if nmax < 0 or mmax < 0:
    raise ValueError('table dimensions must be non-negative')

  try:
    from . import _recursion_integral
  except ImportError as error:
    raise ImportError(
        'The recursion extension is not built. Run '
        'cmake -S snewpdag/plugins -B snewpdag/plugins/build '
        '&& cmake --build snewpdag/plugins/build --config Release.'
    ) from error

  buffer = _recursion_integral.buildLogJTableRaw(
      nmax, mmax, float(a), float(b), float(p), float(q))
  return np.frombuffer(buffer, dtype=np.float64).reshape((nmax + 1, mmax + 1))
