
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def normalize(low, high, list):
    return ((np.array(list) - low) / (high-low)) * 100

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

def sort_keys(keys):
    order_list = []
    for key in keys:
        num = [int(s) for s in key.split() if s.isdigit()]
        if len(num) == 0:
            order_list.append(0)
        else:
            order_list.append(num[0])
    return [x for _, x in sorted(zip(order_list, keys))]

def get_max_power(gpu):
    if gpu == "A4000":
        return 140
    if gpu == "A6000":
        return 300
    if gpu == "A2":
        return 60
    if gpu == "A100":
        return 250
    return None

def norm_power_line(input_list, args):
    for data, gpu_type in input_list:
        power_list = []
        labels = []
        idle_power_list = []
        for key in data.keys():
            labels.append(int(key))
            pow = []
            for df in data[key]:
                i = first_higher_then(df['util_gpu'], 5)
                pow.append(np.mean(df['power'][i:]))
                idle_power_list.append(df['power'][0])
            power_list.append(pow)
        idle = np.mean(idle_power_list)

        #sort for line plot
        power_list = np.array(power_list).flatten()
        sort_list = [x for _, x in sorted(zip(labels, power_list))]
        sort_labels = sorted(labels)

        norm_list = normalize(idle, get_max_power(gpu_type), sort_list)

        # plt.scatter(sort_labels, norm_list, label=gpu_type)
        plt.plot(sort_labels, norm_list, '-o', label=gpu_type)

    plt.xlabel("Number of Streams")
    plt.ylabel("Normalized Average Power (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output + f"power_lineplot.pdf")
    plt.clf()

def time_line(input_list, args):
    for data, gpu_type in input_list:
        time_list = []
        labels = []
        for key in data.keys():
            labels.append(int(key))
            time = []
            for df in data[key]:
                time.append((list(df['duration'])[-1])/1000)
            time_list.append(np.mean(time))

        #sort for line plot
        time_list = np.array(time_list).flatten()
        sort_list = [x for _, x in sorted(zip(labels, time_list))]
        sort_labels = sorted(labels)

        # plt.scatter(sort_labels, norm_list, label=gpu_type)
        plt.plot(sort_labels, np.array(sort_list) / np.array(sort_labels), '-o', label=gpu_type)

    plt.xlabel("Number of Streams")
    plt.ylabel("Time per stream(s)")
    plt.ylim(0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output + f"time_lineplot.pdf")
    plt.clf()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs='+', help="Name of directories of the data")
    parser.add_argument("--output", default="graphs/", help="Name of directory of the graph")


    args = parser.parse_args()

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    args.output = f"{args.output}combined_kernel_graph_{now_str}/"
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    with open(args.output + '/data', 'w') as f:
        f.write(" ".join(args.input))

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    input_list = []
    for input in args.input:
        experiments_gpu = {}
        for (dirpath, dirnames, filenames) in os.walk(input):
            if len(filenames) > 1:
                gpu_data = [s for s in filenames if s.startswith('gpu') and s.endswith('.csv')]
                if len(gpu_data) < 1:
                    continue
                lgpu = []
                name = dirpath.split('/')[-1]
                gpu_type = dirpath.split('/')[-2].split('_')[-3]
                for name_data in gpu_data:
                    lgpu.append(read_data_gpu(f"{dirpath}/{name_data}" ))
                experiments_gpu[name] = lgpu
        input_list.append((experiments_gpu, gpu_type))

    norm_power_line(input_list, args)
    time_line(input_list, args)



if __name__ == "__main__":
    main()