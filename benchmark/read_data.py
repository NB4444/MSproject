
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def get_iterations_change(list):
    first = list[0]
    for i, l in enumerate(list[1:]):
        if l > first:
            return i
    return 0

def first_higher_then(list, n):
    # for A100 it starts on 30% for no reason
    for i, l in enumerate(list[25:]):
        if l > n:
            return i
    return 0

def get_iterations_change_percentage(list, perc):
    first = list[0]
    for i, l in enumerate(list[1:]):
        if l  > first * (1+perc):
            return i
    return 0
def to_ms(diff):
    return int(diff.total_seconds() *1000)

def calculate_energy(data):
    time = data.get("duration")
    p = data.get("power")
    energy = 0
    for i in range(len(p)-1):
        dur = time[i+1] - time[i]
        energy += (dur/1000) * p[i]
    return energy

def sort_keys(l):
    return sorted(l, key=lambda s: int(''.join(filter(str.isdigit, s))))

def power_average_runs(data, args):
    power_list = []
    perc_sm = []
    perc_cores = []
    s_keys = sort_keys(data.keys())
    highest = s_keys[-1].split("_")

    sm_highest = int(highest[0])
    threads_per_sm_highest = int(highest[1])
    tot_highest = sm_highest * threads_per_sm_highest

    for key in data.keys():
        b = key.split("_")
        sm =  int(b[0])
        threads_per_sm = int(b[1])
        tot = sm * threads_per_sm
        pow = []
        for df in data[key]:
            i = first_higher_then(df['util_gpu'], 5)
            pow.append(np.mean(df['power'][i:]))
        power_list.append(np.mean(pow))
        perc_sm.append((sm/sm_highest) * 100)
        perc_cores.append((tot/tot_highest) * 100)

    plt.scatter(perc_sm, power_list)
    # plt.legend()
    plt.title("Power consumption for different utilization workloads")
    plt.xlabel("Utilization of the SMs (%)")
    plt.ylabel("Power")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "power_sm_util.pdf")
    plt.clf()

    plt.scatter(perc_cores, power_list)
    # plt.legend()
    plt.title("Power consumption for different utilization workloads")
    plt.xlabel("Utilization of the cores (%)")
    plt.ylabel("Power")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "power_cores_util.pdf")
    plt.clf()

def duration_runs(data, args):
    power_list = []
    perc_sm = []
    perc_cores = []
    s_keys = sort_keys(data.keys())
    highest = s_keys[-1].split("_")

    sm_highest = int(highest[0])
    threads_per_sm_highest = int(highest[1])
    tot_highest = sm_highest * threads_per_sm_highest

    for key in data.keys():
        b = key.split("_")
        sm =  int(b[0])
        threads_per_sm = int(b[1])
        tot = sm * threads_per_sm
        pow = []
        for df in data[key]:
            pow.append(df['duration'])
        power_list.append(np.mean(pow))
        perc_sm.append((sm/sm_highest) * 100)
        perc_cores.append((tot/tot_highest) * 100)

    plt.scatter(perc_sm, power_list)
    # plt.legend()
    plt.title("Duration for different utilization workloads")
    plt.xlabel("Utilization of the SMs (%)")
    plt.ylabel("Duration (s)")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "dur_sm_util.pdf")
    plt.clf()

    plt.scatter(perc_cores, power_list)
    # plt.legend()
    plt.title("Duration for different utilization workloads")
    plt.xlabel("Utilization of the cores (%)")
    plt.ylabel("Duration (s)")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "dur_cores_util.pdf")
    plt.clf()

def read_data_gpu(input):
    df = pd.read_csv(input, delimiter=', ', engine='python', skipfooter=1)
    df = df.drop('index', axis=1)
    df = df.rename(columns={'power.draw [W]': 'power', 'clocks.current.sm [MHz]': 'sm', 'clocks.current.memory [MHz]' : 'memory', 'clocks.current.graphics [MHz]': 'graphics', 'utilization.gpu [%]': 'util_gpu', 'utilization.memory [%]': 'util_memory'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['power'] = df['power'].str.replace(' W', '').astype(float)
    df['sm'] = df['sm'].str.replace(' MHz', '').astype(int)
    df['memory'] = df['memory'].str.replace(' MHz', '').astype(int)
    df['graphics'] = df['graphics'].str.replace(' MHz', '').astype(int)
    if 'util_gpu' in df:
        df['util_gpu'] = df['util_gpu'].str.replace(' %', '').astype(int)
        df['util_memory'] = df['util_memory'].str.replace(' %', '').astype(int)

    first_timestamp = df.get("timestamp")[0]

    t = df.get("timestamp") - first_timestamp
    df['duration'] = t.apply(to_ms)

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Name of directory of the data")
    parser.add_argument("--output", default="graphs/", help="Name of directory of the graph")

    parser.add_argument("--experiment", type=int, default=0, help="What experiment to create graph for")

    args = parser.parse_args()

    name_exp = args.input.split("/")[-1] if args.input.split("/")[-1] != '' else args.input.split("/")[-2]

    args.output = args.output + name_exp + "/"
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    experiments_gpu = {}
    for (dirpath, dirnames, filenames) in os.walk(args.input):
        if len(filenames) > 1:
            gpu_data = [s for s in filenames if s.startswith('gpu') and s.endswith('.csv')]

            lgpu = []
            name = dirpath.split('/')[-1]

            for name_data in gpu_data:
                lgpu.append(read_data_gpu(f"{dirpath}/{name_data}" ))
            experiments_gpu[name] = lgpu

    power_average_runs(experiments_gpu, args)
    duration_runs(experiments_gpu, args)

if __name__ == "__main__":
    main()