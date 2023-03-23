import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

power = np.array([64.28737827365903, 68.11066140264327, 75.34909010717473, 84.68024537068402, 86.56279487714502, 104.40010544102306, 121.18074368787791, 131.4059608863366])
mu = [20,40,60,80,100,140,200,300]
powerA2 = np.array([22.64275817833335, 24.48616883273573, 26.67769177728844, 28.932651083176392, 30.940723626376258, 33.8153169079972, 37.17311474835912, 41.08913881350302])

change = power - 40

plt.plot(mu, power)

p = P.fit(mu, power, 2)
print(p)
fx, fy = p.linspace(100)
plt.plot(fx, fy)
plt.plot(mu, powerA2)

p = P.fit(mu, powerA2, 2)

# p = p/3
print(p)
fx, fy = p.linspace(100)
plt.plot(fx, fy)

print(power/powerA2)
plt.show()