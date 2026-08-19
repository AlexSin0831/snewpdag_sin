import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


# Change this value to choose the parameter n.
n = 1

# The CSV file is in the same folder as this script.
csv_file = Path(__file__).with_name("continuous_poisson_integral.csv")

mu_values = []
integral_values = []

# Read mu and I(mu) from the CSV file.
with csv_file.open() as file:
    reader = csv.DictReader(file)

    for row in reader:
        mu_values.append(float(row["mu"]))
        integral_values.append(float(row["integral"]))

# Calculate exp(-mu) * mu^n / Gamma(n+1).
poisson_values = [
    math.exp(-mu) * mu**n / math.gamma(n + 1)
    for mu in mu_values
]

# Divide each Poisson value by I(mu).
normalized_values = [
    poisson / integral
    for poisson, integral in zip(poisson_values, integral_values)
]

# Plot both sets of data points on the same graph.
plt.plot(mu_values, poisson_values, "o", label="Poisson value")
plt.plot(mu_values, normalized_values, "x", label="Poisson value / I(mu)")
plt.xlabel("mu")
plt.ylabel("Value")
plt.title(f"Poisson integrand for n = {n}")
plt.legend()
plt.grid()
plt.show()
