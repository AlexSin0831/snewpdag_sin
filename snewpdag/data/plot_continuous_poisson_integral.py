import csv
from pathlib import Path

import matplotlib.pyplot as plt


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

# Plot I(mu) against mu.
plt.plot(mu_values, integral_values, marker="o")
plt.xlabel("mu")
plt.ylabel("I(mu)")
plt.title("Continuous Poisson integral")
plt.grid()
plt.show()
