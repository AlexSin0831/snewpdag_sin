import csv
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

#detector parameters: 
sen = 280
bg = 0.00001

# The CSV file is in the same folder as this Python script.
csv_file = Path(__file__).with_name("continuous_poisson_integral.csv")

# These lists will store the values read from the CSV file.
mu_values = []
integral_values = []

# Read each row from the table.
with csv_file.open() as file:
    reader = csv.DictReader(file)

    for row in reader:
        mu_values.append(float(row["mu"]))
        integral_values.append(float(row["integral"]))

mu_values = np.array(mu_values)
integral_values = np.array(integral_values)

# Calculate the inverse, 1/I(mu), for every integral value.
x_values = (mu_values - bg) / sen
inverse_values = 1 / integral_values


y = inverse_values - 1

A = np.column_stack([
    1 / x_values,
    1 / x_values**2,
    1 / x_values**3
])

B = np.column_stack([1 / x_values])

coeffs1, *_ = np.linalg.lstsq(A, y, rcond=None)
coeffs2, *_ = np.linalg.lstsq(B, y, rcond=None)

a1, a2, a3 = coeffs1
b1 = coeffs2[0]

fit_values_high = (
    1
    + a1 / x_values
    + a2 / x_values**2
    + a3 / x_values**3
)

fit_values_low = 1 + b1 / x_values

print('sen = {:.1f} ; bg = {:1g}'.format(sen,bg))
print(
    'a0 = 1 (fixed); '
    'a1 = {:.6f}; a2 = {:.6f}; a3 = {:.6f}'
    .format(a1,a2,a3)
)

print(
    'b0 = 1 (fixed); '
    'b1 = {:.6f}'
    .format(b1)
)

# Make the plot.
plt.plot(x_values, inverse_values, "o", label='exact')
plt.plot(x_values, fit_values_high, "x", label=(rf"high-order: "
                                                rf"$1 + \frac{{{a1:.3e}}}{{\lambda}}"
                                                rf" + \frac{{{a2:.3e}}}{{\lambda^2}}"
                                                rf" + \frac{{{a3:.3e}}}{{\lambda^3}}$"))       
                                                
plt.plot(x_values, fit_values_low, "x", label=(rf"low-order: "
                                                rf"$1 + \frac{{{b1:.3e}}}{{\lambda}}$"))
        
plt.xlabel("lambda")
plt.ylabel("1 / I(\lambda)")
plt.title("Inverse of the continuous Poisson integral")
plt.plot([], [], ' ', label=fr'sensitivity={sen}, background rate per bin ={bg:.1e}$')

plt.legend()
plt.grid()
plt.show()
