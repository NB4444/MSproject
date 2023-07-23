
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import matplotlib.colors as mcolors

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

class Table:
    name = ""
    energy_cores = []
    energy_all = 0
    power_cores = []
    power_all = 0
    call_counts = []
    runtimes = []
    clocks = []

    def print(self):
        print(self.name)
        print(f"Call counts per core: {','.join(map(str,self.call_counts))}")
        print(f"Clocks per core: {','.join(map(str,self.clocks))}")
        print(f"Runtime per core: {','.join(map(str,self.runtimes))}")
        print(f"Energy per core: {','.join(map(str,self.energy_cores))}")
        print(f"Power per core: {','.join(map(str,self.power_cores))}")
        print(f"Energy overal: { self.energy_all }")
        print(f"Power overal: { self.power_all }\n")


def read_data_cpu(input):
    with open(input) as f:
        lines = f.readlines()
    data_dict = {}
    name = ""
    for line in lines:
        words = line.split(',')
        if name != "" and words[0] == "STRUCT":
            data_dict[name] = new
        if words[0] == "TABLE":
            new_name = words[1].split(' ')[1]
            if new_name == name:
                continue
            name = new_name
            new = Table()
            new.name = name

        if words[0] == "Runtime (RDTSC) [s]":
            new.runtimes = [float(n) for n in words[1:-1] if n != '']
        if words[0] == "call count":
            new.call_counts = [float(n) for n in words[1:-1] if n != '']
        if words[0] == "RAPL_CORE_ENERGY":
            new.energy_cores = [float(n) for n in words[2:-1] if n != '']
        if words[0] == "RAPL_PKG_ENERGY":
                new.energy_all = float(words[2])
        if words[0] == "Clock [MHz]":
            new.clocks = [float(n) for n in words[1:-1] if n != 'nil' and n != '']
        if words[0] == "Power Core [W]":
            new.power_cores = [float(n) for n in words[1:-1] if n != 'nil' and n != '']
        if words[0] == "Power PKG [W]":
            if words[1] != 'nil' and words[1] != '0' and words[1] != '-':
                new.power_all = float(words[1])
            else:
                new.power_all = -1

    data_dict[name] = new
    return data_dict

def read_data_program(input):
    with open(input) as f:
        lines = f.readlines()
    data_dict = {}

    for line in lines:
        words = line.split(' ')
        words = [w.replace("\n", "") for w in words if w != '']
        if len(words) == 0:
            continue
        if words[0] == "Processed":
            data_dict["events"] = int(words[-1])
        if words[0] == "Reconstructed":
            data_dict["track_parameters"] = int(words[-1])
        if words[0] == "Using":
            data_dict['gpu'] = words[words.index("[id:")-1]
    return data_dict

def to_ms(diff):
    return int(diff.total_seconds() *1000)

def calculate_energy(data, skip=0):
    time = data.get("duration")
    p = data.get("power")
    energy = 0
    for i in range(skip, len(p)-1):
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

def sort_keys(l):
    return sorted(l, key=lambda s: int(''.join(filter(str.isdigit, s))))

def cpu_plots(input_list, args):
    labels = []
    power_mean =  []
    power_mean_std =  []
    for input in input_list:
        gpu, cpu, program = input
        sorted_keys = sort_keys(cpu.keys())
        inner_keys = cpu[sorted_keys[0]][0].keys()
        exp = []
        all_mean = []
        all_std = []

        for key in sorted_keys:
            s = key.split("_")
            if s[0] == "threads":
                xlabel = "Number of Streams"
                exp.append(int(s[1]))
            if s[0] == "gpu":
                xlabel = "GPUs"
                exp.append(s[1])
            else:
                xlabel = "Input"
                exp.append(key)
            gpu_type = program[key][0]['gpu']
            power_all = []
            power_std = []
            for k in inner_keys:
                l = [dic[k].power_all for dic in cpu[key] if dic[k].energy_all != 281475000000000.0]
                power_all.append(np.mean(l))
                power_std.append(np.std(l))

            all_mean.append(np.mean(power_all))
            all_std.append(np.mean(power_std))

        if args.label == None:
            labels.append(gpu_type.split('-')[0])
        power_mean.append(all_mean)
        power_mean_std.append(all_std)

    if args.label != None:
        labels = args.label
    length = len(power_mean)
    width = 0.8/len(power_mean)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(length) - np.mean(np.arange(length)))*width
    for i, weight in enumerate(power_mean):
        plt.bar(ind-ind_mean[i], \
                    height=weight, \
                    width=width, \
                    yerr=power_mean_std[i], \
                    label=labels[i])
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Average Power (Watt)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_power_cpu.pdf")
    plt.clf()


def bar_plot_energy_time_for_events(input_list, args):
    energy = []
    energy_gpu_full = []
    time = []
    events = []
    labels = []
    cpu_energy = []
    gpu_power_all = []
    for input in input_list:
        gpu, cpu, program = input
        sorted_keys = sort_keys(gpu.keys())
        inner_keys = cpu[sorted_keys[0]][0].keys()
        exp = []
        energy_gpu_mean = []
        energy_gpu_mean_full = []
        time_gpu_mean = []
        events_list = []
        mean_track = []
        cpu_mean = []
        gpu_p = []

        for key in sorted_keys:
            s = key.split("_")
            if s[0] == "threads":
                xlabel = "Number of Streams"
                exp.append(int(s[1]))
            if s[0] == "gpu":
                xlabel = "GPUs"
                exp.append(s[1])
            else:
                xlabel = "Input"
                exp.append(key)
            energy_gpu = []
            energy_gpu_f = []
            time_list = []
            gp = []
            energy_cpu = 0
            mean_track.append(np.mean([run["track_parameters"] for run in program[key]]))
            events_list.append(program[key][0]["events"])
            gpu_type = program[key][0]['gpu']
            for inner_key in inner_keys:
                energy_cpu += np.mean([dic[inner_key].energy_all for dic in cpu[key] if dic[inner_key].energy_all != 281475000000000.0])
            for df in gpu[key]:
                if not "util_gpu" in df:
                    change_i = get_iterations_change(df["graphics"])
                    if (change_i == 0):
                        change_i = get_iterations_change_percentage(df["power"], 0.1)
                else:
                    change_i = first_higher_then(df["util_gpu"], 5)
                gp.append(np.mean(list(df['power'])[change_i:]))
                energy_gpu.append(calculate_energy(df, change_i))
                energy_gpu_f.append(calculate_energy(df, 0))
                time_list.append((list(df['duration'])[-1] - list(df['duration'])[change_i]) / 1000)

            gpu_p.append(np.mean(gp))
            energy_gpu_mean.append(np.mean(energy_gpu))
            energy_gpu_mean_full.append(np.mean(energy_gpu_f))
            time_gpu_mean.append(np.mean(time_list))
            cpu_mean.append(energy_cpu)
        energy.append(energy_gpu_mean)
        energy_gpu_full.append(energy_gpu_mean_full)
        cpu_energy.append(cpu_mean)
        time.append(time_gpu_mean)
        events.append(events_list)
        if args.label == None:
            labels.append(gpu_type.split('-')[0])
        gpu_power_all.append(gpu_p)

    if args.label != None:
        labels = args.label
    colors = list(mcolors.BASE_COLORS.keys())
    width = 0.8/len(energy_gpu_full)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(energy_gpu_full)) - np.mean(np.arange(len(energy_gpu_full))))*width
    for i, weight in enumerate(energy_gpu_full):
        plt.bar(ind-ind_mean[i], height=cpu_energy[i], width=width, label=f"CPU {labels[i]}")
        plt.bar(ind-ind_mean[i], bottom=cpu_energy[i], height=(weight+np.array(cpu_energy[i])), width=width, label=f"GPU {labels[i]}")
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Energy (J)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_full_application.pdf")
    plt.clf()


    width = 0.8/len(energy_gpu_full)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(energy_gpu_full)) - np.mean(np.arange(len(energy_gpu_full))))*width
    for i, weight in enumerate(energy_gpu_full):
        cpu_per_event = cpu_energy[i] / np.array(events[i])
        plt.bar(ind-ind_mean[i], height=cpu_per_event, width=width, label=f"CPU {labels[i]}")
        plt.bar(ind-ind_mean[i], bottom=cpu_per_event, height=(weight / np.array(events[i]) +cpu_per_event), width=width, label=f"GPU {labels[i]}")
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Energy per event (J/event)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_per_event_full_application.pdf")
    plt.clf()

    width = 0.8/len(energy)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(energy)) - np.mean(np.arange(len(energy))))*width
    for i, weight in enumerate(energy):
        plt.bar(ind-ind_mean[i], height=(weight/np.array(events[i])), width=width, label=labels[i])
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Energy per event (J/event)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_gpu_events_per_event.pdf")
    plt.clf()

    width = 0.8/len(gpu_power_all)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(gpu_power_all)) - np.mean(np.arange(len(gpu_power_all))))*width
    for i, weight in enumerate(gpu_power_all):
        plt.bar(ind-ind_mean[i], height=(weight), width=width, label=labels[i])
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Average Power (Watt)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "avg_power.pdf")
    plt.clf()

    width = 0.8/len(energy)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(energy)) - np.mean(np.arange(len(energy))))*width
    for i, weight in enumerate(energy):
        plt.bar(ind-ind_mean[i], height=(weight/weight[0]), width=width, label=labels[i])
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Normalized energy")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_normalized_gpu_events_per_event.pdf")
    plt.clf()

    width = 0.8/len(energy)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(time)) - np.mean(np.arange(len(time))))*width
    for i, weight in enumerate(time):
        plt.bar(ind-ind_mean[i], height=(weight/np.array(events[i])), width=width, label=labels[i])
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Runtime per event (s/event)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "time_gpu_events_per_event.pdf")
    plt.clf()

    width = 0.8/len(energy)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(time)) - np.mean(np.arange(len(time))))*width
    for i, weight in enumerate(time):
        plt.bar(ind-ind_mean[i], height=(weight/weight[0]), width=width, label=labels[i])
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Normalized Runtime")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "time_normalized_gpu_events_per_event.pdf")
    plt.clf()


    width = 0.8/len(energy)
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(time)) - np.mean(np.arange(len(time))))*width
    for i, weight in enumerate(time):
        plt.bar(ind-ind_mean[i], height=(weight/np.array(mean_track)), width=width, label=labels[i])
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Runtime per track (s/event)")
    plt.xlabel(xlabel)
    plt.legend()
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "time_track_gpu_events_per_event.pdf")
    plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs='+', help="Name of directories of the data")
    parser.add_argument("--output", default="graphs/", help="Name of directory of the graph")
    parser.add_argument("--label", nargs='+', default=None, help="Label names if necessary")

    parser.add_argument("--experiment", type=int, default=0, help="What experiment to create graph for")

    args = parser.parse_args()
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    args.output = f"{args.output}combined_graph_{now_str}/"
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    with open(args.output + '/data', 'w') as f:
        f.write(" ".join(args.input))

    input_list = []
    for input in args.input:
        experiments_gpu = {}
        experiments_cpu = {}
        experiments_program = {}
        for (dirpath, dirnames, filenames) in os.walk(input):
            if len(filenames) > 1:
                gpu_data = [s for s in filenames if s.startswith('gpu') and s.endswith('.csv')]
                cpu_data = [s for s in filenames if s.startswith('cpu') and s.endswith('.csv')]
                program_data = [s for s in filenames if s.startswith('job') and s.endswith('.out')]

                lgpu = []
                lcpu = []
                lprogram = []
                name = dirpath.split('/')[-1]

                for name_data in gpu_data:
                    lgpu.append(read_data_gpu(f"{dirpath}/{name_data}" ))
                experiments_gpu[name] = lgpu

                for name_data in cpu_data:
                    lcpu.append(read_data_cpu(f"{dirpath}/{name_data}" ))
                experiments_cpu[name] = lcpu

                for name_data in program_data:
                    lprogram.append(read_data_program(f"{dirpath}/{name_data}" ))
                experiments_program[name] = lprogram
        input_list.append((experiments_gpu, experiments_cpu, experiments_program))

    bar_plot_energy_time_for_events(input_list, args)
    cpu_plots(input_list, args)



if __name__ == "__main__":
    main()