import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

power = np.array([69.19915148538834, 73.85515574887992, 88.80004531127683, 108.11312535830946, 119.39331300253284, 127.34891247103539, 135.39999663238834, 131.27172270949356]) #140.00 max
powerA2 = np.array([23.05022378506336, 25.420059552541744, 28.08803248747237, 30.104399796987355, 32.131874884957405, 35.413288082101026, 39.78031785493163, 45.3196376486497]) #60.00 max
powerA100 = np.array([66.84023795399116, 70.10372223432057, 74.90626433401941, 80.8033836370964, 84.68810099633825, 96.03202304502588, 113.3332478303856, 138.53872289602504]) #250
powerA6000 = np.array([96.56324244103033, 104.31696574410243, 118.16806945830271, 130.98906294037735, 143.01341202279326, 165.70648926966473, 185.47145238651296, 217.5334262628919]) #300
mu = [20,40,60,80,100,140,200,300]


plt.plot(mu, power, label="A4000")
plt.plot(mu, powerA2, label="A2")
plt.plot(mu, powerA100, label="A100")
plt.plot(mu, powerA6000, label="A6000")

p = P.fit(mu, power, 2)
print(p)
fx, fy = p.linspace(100)
plt.plot(fx, fy)

p = P.fit(mu, powerA2, 2)
print(p)
fx, fy = p.linspace(100)
plt.plot(fx, fy)

p = P.fit(mu, powerA100, 2)
print(p)
fx, fy = p.linspace(100)
plt.plot(fx, fy)

p = P.fit(mu, powerA6000, 2)
print(p)
fx, fy = p.linspace(100)
plt.plot(fx, fy)
plt.legend()
plt.show()
plt.cla()

# plt.plot(mu, power/140, label="A4000")
# plt.plot(mu, powerA2/60, label="A2")
# plt.plot(mu, powerA100/250, label="A100")
# plt.plot(mu, powerA6000/300, label="A6000")
# plt.legend()
# plt.show()