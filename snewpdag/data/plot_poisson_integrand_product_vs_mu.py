import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


# Change these values to choose the parameters n and m.
n = 1
m = 10

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

# Calculate the Poisson values for n and m.
poisson_n_values = [
    math.exp(-mu) * mu**n / math.gamma(n + 1)
    for mu in mu_values
]

poisson_m_values = [
    math.exp(-mu) * mu**m / math.gamma(m + 1)
    for mu in mu_values
]

# Multiply the two Poisson values at each mu.
product_values = [
    poisson_n * poisson_m
    for poisson_n, poisson_m in zip(poisson_n_values, poisson_m_values)
]

# Divide each product by I(mu).
normalized_product_values = [
    product / integral
    for product, integral in zip(product_values, integral_values)
]

# Plot both sets of data points on the same graph.
plt.plot(mu_values, product_values, "o", label="Poisson product")
plt.plot(
    mu_values,
    normalized_product_values,
    "x",
    label="Poisson product / I(mu)",
)
plt.xlabel("mu")
plt.ylabel("Value")
plt.title(f"Poisson integrand product for n = {n} and m = {m}")
plt.legend()
plt.grid()
plt.show()
