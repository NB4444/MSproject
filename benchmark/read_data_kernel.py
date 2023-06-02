
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

def power_box(data, args):
    power_list = []
    labels = []
    idle_power_list = []
    sorted_keys = sort_keys(data.keys())
    for key in sorted_keys:
        if len(key) > 10:
            words = key.split(' ')
            l = len(words)
            middle = l//2
            first = " ".join(words[:middle])
            last = " ".join(words[middle:])
            labels.append(f"{first}\n{last}")
        else:
            labels.append(key)
        pow = []
        for df in data[key]:
            i = first_higher_then(df['util_gpu'], 5)
            pow.extend(list(df['power'][i:]))
            idle_power_list.append(df['power'][0])
        power_list.append(pow)

    if args.gpu == "A4000":
        max_power = 140
    if args.gpu == "A6000":
        max_power = 300
    if args.gpu == "A2":
        max_power = 60
    if args.gpu == "A100":
        max_power = 250

    idle = np.mean(idle_power_list)

    print(f"Max power: {max_power}, Idle power {idle}")

    for p in power_list:
        print(f"Mean power: {np.mean(p)}")

    plt.axhline(y=max_power, color='black', linestyle='--', label="Maximum Power")
    plt.axhline(y=idle, color='gray', linestyle='--', label="Averaged Idle Power")
    plt.boxplot(power_list, showfliers=False, labels=labels)
    plt.title(f"Power consumption for different kernels for {args.gpu}")
    plt.xlabel("Different Kernels")
    plt.ylabel("Power (Watt)")
    plt.ylim(0)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + f"power_boxplot_{args.gpu}.pdf")
    plt.clf()

def graphics_box(data, args):
    power_list = []
    labels = []
    sorted_keys = sort_keys(data.keys())
    for key in sorted_keys:
        if len(key) > 10:
            words = key.split(' ')
            l = len(words)
            middle = l//2
            first = " ".join(words[:middle])
            last = " ".join(words[middle:])
            labels.append(f"{first}\n{last}")
        else:
            labels.append(key)
        pow = []
        for df in data[key]:
            i = first_higher_then(df['util_gpu'], 5)
            pow.extend(list(df['graphics'][i:]))
        power_list.append(pow)



    for p in power_list:
        print(f"Mean compute frequency: {np.mean(p)}, Median compute frequency: {np.median(p)}, Min compute frequency: {np.min(p)}, Max compute frequency: {np.max(p)}")

    plt.boxplot(power_list, showfliers=False, labels=labels)
    plt.title(f"Compute frequency for different kernels for {args.gpu}")
    plt.xlabel("Different Kernels")
    plt.ylabel("Compute frequency (MHz)")
    plt.ylim(0)
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + f"compute_frequency_boxplot_{args.gpu}.pdf")
    plt.clf()

def memory_box(data, args):
    power_list = []
    labels = []
    sorted_keys = sort_keys(data.keys())
    for key in sorted_keys:
        if len(key) > 10:
            words = key.split(' ')
            l = len(words)
            middle = l//2
            first = " ".join(words[:middle])
            last = " ".join(words[middle:])
            labels.append(f"{first}\n{last}")
        else:
            labels.append(key)
        pow = []
        for df in data[key]:
            i = first_higher_then(df['util_gpu'], 5)
            pow.extend(list(df['memory'][i:]))
        power_list.append(pow)


    for p in power_list:
        print(f"Mean memory frequency: {np.mean(p)}")

    plt.boxplot(power_list, showfliers=False, labels=labels)
    plt.title(f"Memory frequency for different kernels for {args.gpu}")
    plt.xlabel("Different Kernels")
    plt.ylabel("Memory frequency (MHz)")
    plt.ylim(0)
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + f"memory_frequency_boxplot_{args.gpu}.pdf")
    plt.clf()

def time_bar(data, args):
    power_list = []
    labels = []
    sorted_keys = sort_keys(data.keys())
    std_list = []
    for key in sorted_keys:
        if len(key) > 10:
            words = key.split(' ')
            l = len(words)
            middle = l//2
            first = " ".join(words[:middle])
            last = " ".join(words[middle:])
            labels.append(f"{first}\n{last}")
        else:
            labels.append(key)
        pow = []
        for df in data[key]:
            pow.append((list(df['duration'])[-1])/1000)
        power_list.append(np.mean(pow))
        std_list.append(np.std(pow))

    plt.bar(labels, power_list, yerr=std_list)
    plt.title(f"Runtime for different kernels for {args.gpu}")
    plt.xlabel("Different Kernels")
    plt.ylabel("Time (s)")
    plt.ylim(0)
    # plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + f"time_bar_{args.gpu}.pdf")
    plt.clf()

def time_line(data, args):
    power_list = []
    labels = []
    for key in data.keys():
        labels.append(int(key))
        pow = []
        for df in data[key]:
            pow.append((list(df['duration'])[-1])/1000)
        power_list.append(np.mean(pow))


    plt.scatter(labels, power_list)
    # plt.title(f"Power consumption for different kernels for {args.gpu}")
    # plt.xlabel("Number of FP32 instruction per 1 FP64 instruction")
    plt.xlabel("Number of Streams")
    plt.ylabel("Time (s)")
    plt.ylim(0)
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + f"time_lineplot_{args.gpu}.pdf")
    plt.clf()

    plt.scatter(labels, np.array(power_list)/np.array(labels))
    # plt.title(f"Power consumption for different kernels for {args.gpu}")
    # plt.xlabel("Number of FP32 instruction per 1 FP64 instruction")
    plt.xlabel("Number of Streams")
    plt.ylabel("Time per stream(s)")
    plt.ylim(0)
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + f"time_per_stream_lineplot_{args.gpu}.pdf")
    plt.clf()

def power_line(data, args):
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

    if args.gpu == "A4000":
        max_power = 140
    if args.gpu == "A6000":
        max_power = 300
    if args.gpu == "A2":
        max_power = 60
    if args.gpu == "A100":
        max_power = 250

    idle = np.mean(idle_power_list)

    print(f"Max power: {max_power}, Idle power {idle}")
    print(f"Highest power {np.max(power_list)}")
    print(f"Average power {np.mean(power_list)}")
    print([x for _, x in sorted(zip(labels, [p[0] for p in power_list]), key=lambda pair: pair[0])])


    plt.axhline(y=max_power, color='black', linestyle='--', label="Maximum Power")
    plt.axhline(y=idle, color='gray', linestyle='--', label="Averaged Idle Power")
    plt.scatter(labels, power_list)
    # plt.title(f"Power consumption for different kernels for {args.gpu}")
    # plt.xlabel("Number of FP32 instruction per 1 FP64 instruction")
    plt.xlabel("Number of Streams")
    plt.ylabel("Power (Watt)")
    plt.ylim(0)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + f"power_lineplot_{args.gpu}.pdf")
    plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Name of directory of the data")
    parser.add_argument("--output", default="graphs/", help="Name of directory of the graph")
    parser.add_argument("--gpu", default="A4000", help="Type of GPU for titles")
    parser.add_argument("--line", default=0, help="Forced line use")

    args = parser.parse_args()

    name_exp = args.input.split("/")[-1] if args.input.split("/")[-1] != '' else args.input.split("/")[-2]

    args.output = args.output + name_exp + "/"
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # plt.figure(figsize=(5, 4))
    experiments_gpu = {}
    for (dirpath, dirnames, filenames) in os.walk(args.input):
        if len(filenames) > 1:
            gpu_data = [s for s in filenames if s.startswith('gpu') and s.endswith('.csv')]
            if len(gpu_data) < 1:
                continue
            lgpu = []
            name = dirpath.split('/')[-1]
            for name_data in gpu_data:
                lgpu.append(read_data_gpu(f"{dirpath}/{name_data}" ))
            experiments_gpu[name] = lgpu

    if len(experiments_gpu) < 10 and args.line == 0:
        power_box(experiments_gpu, args)
        time_bar(experiments_gpu, args)
        memory_box(experiments_gpu, args)
        graphics_box(experiments_gpu, args)
    else:
        power_line(experiments_gpu, args)
        time_line(experiments_gpu, args)




if __name__ == "__main__":
    main()