import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

mu = [20,40,60,80,100,140,200,300]

def power_func():
    power = np.array([69.19915148538834, 73.85515574887992, 88.80004531127683, 108.11312535830946, 119.39331300253284, 127.34891247103539, 135.39999663238834, 131.27172270949356]) #140.00 max
    powerA2 = np.array([23.05022378506336, 25.420059552541744, 28.08803248747237, 30.104399796987355, 32.131874884957405, 35.413288082101026, 39.78031785493163, 45.3196376486497]) #60.00 max
    powerA100 = np.array([66.84023795399116, 70.10372223432057, 74.90626433401941, 80.8033836370964, 84.68810099633825, 96.03202304502588, 113.3332478303856, 138.53872289602504]) #250
    powerA6000 = np.array([96.56324244103033, 104.31696574410243, 118.16806945830271, 130.98906294037735, 143.01341202279326, 165.70648926966473, 185.47145238651296, 217.5334262628919]) #300


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

def diff(pred, measured):
    calc = (pred-measured)/measured
    return np.mean(np.abs(calc)), np.std(calc)

def gpu_model():
    events = 100000
    energy_A4000 = [5185.141970000124, 6605.479396000158, 7541.046152000001, 10284.4495380003, 12313.63820200063, 18743.956063998965, 30710.728255998933, 50585.86805199575]
    power_A4000 = [69.49206488155338, 75.23897785744643, 86.63455776083732, 109.22209721190112, 116.33874833827353, 129.8560494147376, 135.63421362582983, 131.28197788971062]
    idle_power_A4000 = 37
    full_time_A4000 = np.array([79.35139999999998, 93.58619999999999, 93.7246, 102.5918, 115.7058, 155.27519999999998, 239.67919999999998, 401.6022])
    idle_time_A4000 =  np.array([11.5762, 12.336799999999998, 12.0082, 13.261, 14.2168, 15.777, 18.5942, 22.939])
    event_time_A4000 = (full_time_A4000 - idle_time_A4000) / events
    print(f"Event time A4000: {event_time_A4000}")

    energy_calc_A4000 = idle_time_A4000 * idle_power_A4000 + events * (power_A4000 * event_time_A4000)
    print(diff(energy_calc_A4000, energy_A4000))

    plt.plot(mu, energy_calc_A4000, label="Calculated A4000")
    plt.plot(mu, energy_A4000, label="Measured A4000")
    plt.legend()
    plt.show()
    plt.cla()

    util_A4000 = np.array([54.72951516354419, 59.66675941476914, 68.35432941030582, 73.78840462664328, 73.97146590204287, 72.08683302625077, 75.4623763395548, 83.99664412878384])/100

    events = 100000
    energy_A6000 = [7356.15803399984, 9577.64417599995, 10630.831578000178, 12763.58298399983, 15248.828975999946, 20811.024940000152, 34054.66214600057, 68383.37378800075]
    power_A6000 = [99.02450671903631, 108.159657594557, 121.69193569253362, 139.43949618253617, 152.37919731137742, 179.40842790783404, 203.6435966915058, 243.41334391428796]
    idle_power_A6000 = 75

    full_time_A6000 = np.array([76.7484, 91.5892, 91.6092, 97.1044, 106.6344, 124.50900000000001, 178.04979999999998, 295.4224])
    idle_time_A6000 =  np.array([9.914200000000001, 10.8574, 11.8494, 12.9376, 13.930200000000001, 15.6364, 18.335, 22.7316])
    event_time_A6000 = (full_time_A6000 - idle_time_A6000) / events
    print(f"Event time A6000: {event_time_A6000}")

    energy_calc_A6000 = idle_time_A6000 * idle_power_A6000 + events * (power_A6000 * event_time_A6000)
    print(diff(energy_calc_A6000, energy_A6000))

    plt.plot(mu, energy_calc_A6000, label="Calculated A6000")
    plt.plot(mu, energy_A6000, label="Measured A6000")

    plt.legend()
    plt.show()
    plt.cla()

    util_A6000 = np.array([53.411866738294634, 57.06365254836444, 66.48262655044132, 70.40465063519136, 73.01585571216808, 71.85934674676206, 69.81431509599452, 74.26866775375636])/100

    events = 10000
    energy_A2 = [384.17652400000213, 487.5535240000021, 622.3084300000007, 834.6134940000184, 1073.815302000005, 1559.6106959999593, 2627.9985839999217, 5382.034525999556]
    power_A2 = [27.56709468087019, 32.18477293250427, 37.69196914912332, 40.56661368728677, 42.5553539184089, 45.55704621774727, 48.283723683707024, 49.65448352563327]
    idle_power_A2 = 20

    full_time_A2 = np.array([18.220399999999998, 20.952800000000003, 23.808200000000003, 28.955400000000004, 35.5318, 44.90560000000001, 66.09360000000001, 116.9])
    idle_time_A2 =  np.array([9.996799999999999, 10.855799999999999, 11.9746, 13.01, 14.1022, 15.614, 18.511200000000002, 22.988599999999998])
    event_time_A2 = (full_time_A2 - idle_time_A2) / events
    print(f"Event time A2: {event_time_A2}")

    energy_calc_A2 = idle_time_A2 * idle_power_A2 + events * (power_A2 * event_time_A2)
    print(diff(energy_calc_A2, energy_A2))

    plt.plot(mu, energy_calc_A2, label="Calculated A2")
    plt.plot(mu, energy_A2, label="Measured A2")

    plt.legend()
    plt.show()
    plt.cla()

    util_A2 = np.array([66.29334542072488, 73.60778055802147, 81.02431931258695, 84.25543639769866, 87.30113236991528, 92.32593309063594, 96.42980571239556, 98.70714745228379])/100

    events = 100000
    energy_A100 = [5336.30585000017, 6658.783310000084, 7805.92571000006, 9000.7622699997, 10058.567619999569, 12890.141239999954, 20909.339519999474, 41383.51360000024]
    power_A100 = [64.60389071249276, 70.56121225552663, 72.4255319984295, 78.51308716872691, 86.8947898930116, 100.04558075712325, 116.29872694310956, 140.35159322507909]
    idle_power_A100 = 37
    full_time_A100 = np.array([87.057, 99.649, 113.858, 121.742, 124.017, 138.649, 192.227, 311.766])
    idle_time_A100 = np.array([10.496, 11.1, 12.533, 13.551, 14.577, 15.786, 18.444, 23.301])
    event_time_A100 = (full_time_A100 - idle_time_A100)/events
    print(f"Event time A100: {event_time_A100}")

    energy_calc_A100 = idle_time_A100 * idle_power_A100 + events * (power_A100 * event_time_A100)
    print(diff(energy_calc_A100, energy_A100))

    plt.plot(mu, energy_calc_A100, label="Calculated A100")
    plt.plot(mu, energy_A100, label="Measured A100")

    plt.legend()
    plt.show()
    plt.cla()

    util_A100 = np.array([61.61678382491485, 63.37985886855639, 65.85674772988206, 69.01475539710833, 72.52704140866874, 69.36955997104015, 70.94995878959143, 77.6954259576901])/100


def cpu_model():
    events = 100000
    energy_A4000 = [4215.59914, 5007.02456, 5830.225219999999, 6339.711260000001, 7166.98826, 9830.94186, 14257.054320000001, 25233.208880000002]
    power_A4000 = [52.928219999999996, 53.19054, 63.124340000000004, 64.39658, 66.15306000000001, 68.95272, 66.33037999999999, 67.68894]
    mean_power_A4000 = np.mean(power_A4000)
    full_time_A4000 = np.array([79.64636000000002, 94.13456000000001, 92.33902, 98.45388, 108.34312, 142.53462, 214.93676, 372.81278])

    print(f"Mean power A4000: {mean_power_A4000}")
    energy_calc_A4000 = mean_power_A4000 * full_time_A4000
    print(diff(energy_calc_A4000, energy_A4000))

    plt.plot(mu, energy_calc_A4000, label="Calculated A4000")
    plt.plot(mu, energy_A4000, label="Measured A4000")
    plt.legend()
    plt.show()
    plt.cla()

    energy_A6000 = [4103.11122, 4919.12428, 5643.27888, 6372.131659999999, 7082.437340000001, 8365.9697, 12075.78154, 19937.65692]
    power_A6000 = [53.177859999999995, 53.522000000000006, 61.31627999999999, 65.3572, 66.20792, 66.99006, 67.66066, 67.44409999999999]
    mean_power_A6000 = np.mean(power_A6000)
    full_time_A6000 = np.array([77.1602, 91.91346000000001, 92.04448, 97.51254, 106.9795, 124.8998, 178.47881999999998, 295.66472])

    print(f"Mean power A6000: {mean_power_A6000}")
    energy_calc_A6000 = mean_power_A6000 * full_time_A6000
    print(diff(energy_calc_A6000, energy_A6000))

    plt.plot(mu, energy_calc_A6000, label="Calculated A6000")
    plt.plot(mu, energy_A6000, label="Measured A6000")

    plt.legend()
    plt.show()
    plt.cla()

    energy_A2 = [882.4445400000001, 1015.7910599999999, 1203.14842, 1502.7288199999998, 1868.9317199999998, 2552.8258, 4795.849560000001, 9276.615759999999]
    power_A2 = [51.68796, 51.87284, 53.44724, 54.8322, 56.874340000000004, 59.430060000000005, 73.37284, 76.14308]
    mean_power_A2 = np.mean(power_A2)
    full_time_A2 = np.array([18.220399999999998, 20.952800000000003, 23.808200000000003, 28.955400000000004, 35.5318, 44.90560000000001, 66.09360000000001, 116.9])


    print(f"Mean power A2: {mean_power_A2}")
    energy_calc_A2 = mean_power_A2 * full_time_A2
    print(diff(energy_calc_A2, energy_A2))

    plt.plot(mu, energy_calc_A2, label="Calculated A2")
    plt.plot(mu, energy_A2, label="Measured A2")

    plt.legend()
    plt.show()
    plt.cla()

    energy_A100 = [4873.9272, 5689.9703, 6660.4986, 7552.9591, 8158.7129, 9434.7148, 13194.6044, 21001.8713]
    power_A100 = [55.9168, 57.1781, 58.3289, 62.1149, 65.834, 68.0712, 68.6639, 67.3769]
    mean_power_A100 = np.mean(power_A100)
    full_time_A100 = np.array([87.1639, 99.5131, 114.1886, 121.5966, 123.9286, 138.6007, 192.1622, 311.7073])
    print(mean_power_A100)

    energy_calc_A100 = mean_power_A100 * full_time_A100
    print(diff(energy_calc_A100, energy_A100))

    plt.plot(mu, energy_calc_A100, label="Calculated A100")
    plt.plot(mu, energy_A100, label="Measured A100")

    plt.legend()
    plt.show()
    plt.cla()



gpu_model()
cpu_model()
# power_func()