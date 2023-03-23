
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import math

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
    return data_dict

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

def power(data, args, experiment, run):
    ax = data.plot(x='duration', y='power', label="Power")
    ax.set_xlabel("Time(ms)")
    ax.set_ylabel("Power(W)")
    ax2 = ax.twinx()
    ax2.set_ylabel("Frequency(MHz)")
    data.plot(x='duration', y='graphics', ax=ax2, color='r', label="Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output + f"power_{experiment}_{run}.pdf")
    plt.clf()

def power_plot_multiple_runs(data, args):
    for key in data.keys():
        for i, df in enumerate(data[key]):
            plt.plot(df["duration"], df["power"], label="Frequency")
        plt.ylim(0)
        plt.xlabel("Time(ms)")
        plt.ylabel("Power(W)")
        plt.tight_layout()
        plt.savefig(args.output + f"power_{key}.pdf")
        plt.clf()

def mem_freq_multiple_runs(data, args):
    for key in data.keys():
        for i, df in enumerate(data[key]):
            plt.plot(df["duration"], df["memory"], label="Frequency")
        plt.ylim(0)
        plt.xlabel("Time(ms)")
        plt.ylabel("Memory Frequency(MHz)")
        plt.tight_layout()
        plt.savefig(args.output + f"memory_freq_{key}.pdf")
        plt.clf()

def sort_keys(l):
    return sorted(l, key=lambda s: int(''.join(filter(str.isdigit, s))))

def bar_plot_energy_cpu(data, args):
    all = []
    exp = []
    sorted_keys = sort_keys(data.keys())
    inner_keys = data[sorted_keys[0]][0].keys()

    for key in sorted_keys:
        exp.append(key)
        areas = []
        energy = []
        for k in inner_keys:
            areas.append(k)
            l = [dic[k].energy_all for dic in data[key] if dic[k].energy_all != 281475000000000.0]
            energy.append(np.mean(l))
        all.append(energy)
    d = np.transpose(all)

    width = 0.5
    bottom = np.zeros(len(exp))
    for i, weight in enumerate(d):
        plt.bar(exp, weight, width=width, label=areas[i], bottom=bottom)
        bottom += weight

    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Energy (J)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_cpu.pdf")
    plt.clf()

def bar_plot_avg_power_cpu(data, args):
    all_mean = []
    all_std = []
    exp = []
    sorted_keys = sort_keys(data.keys())
    inner_keys = data[sorted_keys[0]][0].keys()

    for key in sorted_keys:
        exp.append(key)
        areas = []
        power_all = []
        power_std = []
        for k in inner_keys:
            areas.append(k)
            l = [dic[k].power_all for dic in data[key] if dic[k].energy_all != 281475000000000.0]
            power_all.append(np.mean(l))
            power_std.append(np.std(l))

        all_mean.append(power_all)
        all_std.append(power_std)
    d = np.transpose(all_mean)
    d_std = np.transpose(power_std)

    width = 0.1
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(d)) - np.mean(np.arange(len(d))))*width
    for i, weight in enumerate(d):
        plt.bar(ind-ind_mean[i], height=weight, width=width, yerr=d_std[i], label=areas[i])

    plt.xticks(ind, exp, rotation=30)
    plt.legend()
    plt.ylabel("Average Power (Watt)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_power_cpu.pdf")
    plt.clf()

def bar_plot_avg_freq_cpu(data, args):
    all_mean = []
    all_std = []
    exp = []
    sorted_keys = sort_keys(data.keys())
    inner_keys = data[sorted_keys[0]][0].keys()

    for key in sorted_keys:
        exp.append(key)
        areas = []
        freq_all = []
        freq_std = []
        for k in inner_keys:
            areas.append(k)
            l = [dic[k].clocks for dic in data[key]]
            freq_all.append(np.mean(l))
            freq_std.append(np.std(l))

        all_mean.append(freq_all)
        all_std.append(freq_std)
    d = np.transpose(all_mean)
    d_std = np.transpose(freq_std)

    width = 0.1
    ind = np.arange(len(exp))
    ind_mean = (np.arange(len(d)) - np.mean(np.arange(len(d))))*width
    for i, weight in enumerate(d):
        plt.bar(ind-ind_mean[i], height=weight, width=width, yerr=d_std[i], label=areas[i])

    plt.xticks(ind, exp, rotation=30)
    plt.legend()
    plt.ylabel("Average Clocks (MHz)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_clocks_cpu.pdf")
    plt.clf()

def plot_energy_gpu(data, args):
    exp = []
    energy_mean = []
    energy_std = []
    sorted_keys = sort_keys(data.keys())
    for key in sorted_keys:
        exp.append(key)
        energy = []
        for df in data[key]:
            energy.append(calculate_energy(df))
        energy_mean.append(np.mean(energy))
        energy_std.append(np.std(energy))

    plt.xticks(rotation=30)
    plt.bar(exp, energy_mean, yerr=energy_std)
    plt.tight_layout()
    plt.ylabel("Energy (J)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_gpu.pdf")
    plt.clf()

    if "mu" in args.output:
        try:
            n = [int(''.join(filter(str.isdigit, key))) for key in sorted_keys]
        except:
            return

        plt.xticks(rotation=30)
        plt.plot(n, energy_mean)
        plt.ylim(0)
        plt.tight_layout()
        plt.ylabel("Energy (J)")
        plt.xlabel("mu")
        plt.tight_layout()
        plt.savefig(args.output + "energy_line_gpu.pdf")
        plt.clf()

def bar_plot_energy_total(cpu, gpu, program, args):
    exp = []
    sorted_keys = sort_keys(cpu.keys())
    inner_keys = cpu[sorted_keys[0]][0].keys()
    gpu_mean = []
    cpu_mean = []
    events = []
    mean_track = []

    for key in sorted_keys:
        exp.append(key)
        energy_gpu = []
        energy_cpu = 0
        mean_track.append(np.mean([run["track_parameters"] for run in program[key]]))
        events.append(program[key][0]["events"])
        for inner_key in inner_keys:
            energy_cpu += np.mean([dic[inner_key].energy_all for dic in cpu[key] if dic[inner_key].energy_all != 281475000000000.0])
        for df in gpu[key]:
            energy_gpu.append(calculate_energy(df))
        gpu_mean.append(np.mean(energy_gpu))
        cpu_mean.append(energy_cpu)

    width = 0.5
    bottom = np.zeros(len(exp))

    plt.bar(exp, cpu_mean, width=width, label="CPU", bottom=bottom)
    bottom += np.array(cpu_mean)
    plt.bar(exp, gpu_mean, width=width, label="GPU", bottom=bottom)

    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Energy (J)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy.pdf")
    plt.clf()

    bottom = np.zeros(len(exp))
    cpu_per_event = np.array(cpu_mean) / np.array(events)
    plt.bar(exp, cpu_per_event, width=width, label="CPU", bottom=bottom)
    bottom += cpu_per_event
    plt.bar(exp, np.array(gpu_mean) / np.array(events), width=width, label="GPU", bottom=bottom)
    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Energy per event (J/event)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_per_event.pdf")
    plt.clf()

    bottom = np.zeros(len(exp))
    cpu_per_event = np.array(cpu_mean) / np.array(mean_track)
    plt.bar(exp, cpu_per_event, width=width, label="CPU", bottom=bottom)
    bottom += cpu_per_event
    plt.bar(exp, np.array(gpu_mean) / np.array(mean_track), width=width, label="GPU", bottom=bottom)
    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Energy per track (J/track)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_per_track.pdf")
    plt.clf()

def bar_plot_runtime_per_track(data, program, args):
    exp = []
    sorted_keys = sort_keys(data.keys())
    inner_keys = data[sorted_keys[0]][0].keys()
    cpu_mean = []
    mean_track = []

    for key in sorted_keys:
        exp.append(key)
        runtime_cpu = 0
        mean_track.append(np.mean([run["track_parameters"] for run in program[key]]))
        for inner_key in inner_keys:
            runtime_cpu += np.mean([dic[inner_key].runtimes for dic in data[key]])
        cpu_mean.append(runtime_cpu)

    bottom = np.zeros(len(exp))
    width = 0.5
    cpu_per_event = np.array(cpu_mean) / np.array(mean_track)
    plt.bar(exp, cpu_per_event, width=width, label="CPU", bottom=bottom)
    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Runtime per track (s/track)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "runtime_per_track.pdf")
    plt.clf()

def bar_plot_track_paramaters(program, args):
    sorted_keys = sort_keys(program.keys())
    exp = []
    mean_track = []
    std_track = []
    events = []
    for key in sorted_keys:
        exp.append(key)
        runs = [run["track_parameters"] for run in program[key]]
        events.append(program[key][0]["events"])
        mean_track.append(np.mean(runs))
        std_track.append(np.std(runs))

    events = np.array(events)
    width = 0.5

    plt.bar(exp, mean_track, yerr=std_track, width=width)
    plt.xticks(rotation=30)
    plt.ylabel("Reconstructed track parameters")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "reconstructed_track_parameters.pdf")
    plt.clf()

    plt.bar(exp, mean_track / events, width=width)
    plt.xticks(rotation=30)
    plt.ylabel("Reconstructed track parameters per event")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "reconstructed_track_parameters_per_event.pdf")
    plt.clf()

    if "mu" in args.output:
        try:
            n = np.array([int(''.join(filter(str.isdigit, key))) for key in sorted_keys])
        except:
            return
        plt.plot(n, mean_track)
        plt.xticks(rotation=30)
        plt.ylabel("Reconstructed track parameters")
        plt.xlabel("mu")
        plt.plot()
        plt.tight_layout()
        plt.savefig(args.output + "reconstructed_track_parameters_per_mu.pdf")
        plt.clf()

def bar_plot_runtime(cpu, gpu, args):
    exp = []
    sorted_keys = sort_keys(cpu.keys())
    inner_keys = cpu[sorted_keys[0]][0].keys()
    gpu_mean = []
    cpu_mean = []

    for key in sorted_keys:
        exp.append(key)
        runtime_gpu = []
        runtime_cpu = 0
        for inner_key in inner_keys:
                runtime_cpu += np.mean([dic[inner_key].runtimes for dic in cpu[key]])
        for df in gpu[key]:
            runtime_gpu.append(list(df['duration'])[-1] / 1000)
        gpu_mean.append(np.mean(runtime_gpu))
        cpu_mean.append(runtime_cpu)


    width = 0.4
    ind = np.arange(len(exp))

    plt.bar(ind, height=(np.array(gpu_mean)/np.array(cpu_mean)), width=width)
    plt.axhline(y = 1, color = 'b', linestyle = 'dotted')

    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Runtime GPU/CPU")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "runtime_compare.pdf")
    plt.clf()

    plt.bar(ind-width/2, height=cpu_mean, width=width, label="CPU")
    plt.bar(ind+width/2, height=gpu_mean, width=width, label="GPU")

    plt.xticks(ind, exp, rotation=30)
    plt.legend()
    plt.ylabel("Runtime (s)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "runtime.pdf")
    plt.clf()

def bar_plot_average_power_gpu(data, args):
    exp = []
    power_mean = []
    power_std = []
    sorted_keys = sort_keys(data.keys())
    for key in sorted_keys:
        exp.append(key)
        energy = []
        for df in data[key]:
            dur = (list(df['duration'])[-1]) / 1000
            energy.append(calculate_energy(df)/ dur)
        power_mean.append(np.mean(energy))
        power_std.append(np.std(energy))

    plt.xticks(rotation=30)
    plt.bar(exp, power_mean, yerr=power_std)
    plt.tight_layout()
    plt.ylabel("Average Power (W)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_power_gpu.pdf")
    plt.clf()

def bar_plot_runtime_cpu(data, args):
    all = []
    exp = []

    sorted_keys = sort_keys(data.keys())
    inner_keys = data[sorted_keys[0]][0].keys()

    for key in sorted_keys:
        exp.append(key)
        areas = []
        runtime = []
        for k in inner_keys:
            areas.append(k)
            runtime.append(np.mean([dic[k].runtimes for dic in data[key]]))
        all.append(runtime)
    d = np.transpose(all)

    width = 0.5
    bottom = np.zeros(len(exp))
    for i, weight in enumerate(d):
        plt.bar(exp, weight, width=width, label=areas[i], bottom=bottom)
        bottom += weight

    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Runtime (s)")
    plt.xlabel("Input")
    plt.plot()
    plt.savefig(args.output + "runtime_cpu.pdf")
    plt.clf()

def read_data_gpu(input):
    df = pd.read_csv(input, delimiter=', ', engine='python', skipfooter=1)
    df = df.drop('index', axis=1)
    df = df.rename(columns={'power.draw [W]': 'power', 'clocks.current.sm [MHz]': 'sm', 'clocks.current.memory [MHz]' : 'memory', 'clocks.current.graphics [MHz]': 'graphics'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['power'] = df['power'].str.replace(' W', '').astype(float)
    df['sm'] = df['sm'].str.replace(' MHz', '').astype(int)
    df['memory'] = df['memory'].str.replace(' MHz', '').astype(int)
    df['graphics'] = df['graphics'].str.replace(' MHz', '').astype(int)

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
    experiments_cpu = {}
    experiments_program = {}
    for (dirpath, dirnames, filenames) in os.walk(args.input):
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

    if args.experiment == 0:
        bar_plot_energy_cpu(experiments_cpu, args)
        bar_plot_avg_power_cpu(experiments_cpu, args)
        bar_plot_avg_freq_cpu(experiments_cpu, args)
        bar_plot_runtime_cpu(experiments_cpu, args)

        plot_energy_gpu(experiments_gpu, args)
        bar_plot_average_power_gpu(experiments_gpu, args)

        bar_plot_energy_total(experiments_cpu, experiments_gpu, experiments_program, args)
        bar_plot_runtime(experiments_cpu, experiments_gpu, args)

        bar_plot_track_paramaters(experiments_program, args)
        bar_plot_runtime_per_track(experiments_cpu, experiments_program, args)

        args.output = args.output + "/power/"
        if not os.path.exists(args.output):
            os.mkdir(args.output)

        power_plot_multiple_runs(experiments_gpu, args)
        mem_freq_multiple_runs(experiments_gpu, args)
        for key in experiments_gpu.keys():
            for i, df in enumerate(experiments_gpu[key]):
                power(df, args, key, i)
    elif(args.experiment == 1):
        bar_plot_energy_total(experiments_cpu, experiments_gpu, experiments_program, args)
        bar_plot_runtime(experiments_cpu, experiments_gpu, args)

        bar_plot_track_paramaters(experiments_program, args)
        bar_plot_avg_freq_cpu(experiments_cpu, args)

if __name__ == "__main__":
    main()