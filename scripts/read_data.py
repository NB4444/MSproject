
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

    for key in data.keys():
        for i, df in enumerate(data[key]):
            l = np.array(df["power"])
            plt.plot(df["duration"], l - l[0], label="Frequency")
        plt.ylim(0)
        plt.xlabel("Time(ms)")
        plt.ylabel("Power(W)")
        plt.tight_layout()
        plt.savefig(args.output + f"power_shifted_{key}.pdf")
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


def bar_plot_energy_time_for_events(gpu, cpu, program, args):
    sorted_keys = sort_keys(gpu.keys())
    inner_keys = cpu[sorted_keys[0]][0].keys()
    exp = []
    energy_gpu_mean = []
    energy_gpu_std = []
    time_gpu_mean = []
    time_gpu_std = []
    events = []
    mean_track = []
    time_mean_cpu = []
    energy_mean_cpu = []

    for key in sorted_keys:
        exp.append(key)
        energy_gpu = []
        time_list = []
        energy_cpu = 0
        runtime_cpu = 0
        mean_track.append(np.mean([run["track_parameters"] for run in program[key]]))
        events.append(program[key][0]["events"])
        for inner_key in inner_keys:
            energy_cpu += np.mean([dic[inner_key].energy_all for dic in cpu[key] if dic[inner_key].energy_all != 281475000000000.0])
            runtime_cpu += np.mean([dic[inner_key].runtimes for dic in cpu[key]])
        for df in gpu[key]:
            if not "util_gpu" in df:
                change_i = get_iterations_change(df["graphics"])
                if (change_i == 0):
                    change_i = get_iterations_change_percentage(df["power"], 0.1)
            else:
                change_i = first_higher_then(df["util_gpu"], 5)
            energy_gpu.append(calculate_energy(df, change_i))
            time_list.append((list(df['duration'])[-1] - list(df['duration'])[change_i]) / 1000)

        energy_gpu_mean.append(np.mean(energy_gpu))
        energy_gpu_std.append(np.std(energy_gpu))
        time_gpu_mean.append(np.mean(time_list))
        time_gpu_std.append(np.std(time_list))
        time_mean_cpu.append(runtime_cpu)
        energy_mean_cpu.append(energy_cpu)

    plt.bar(exp, np.array(energy_gpu_mean), yerr=energy_gpu_std, label="GPU")
    plt.xticks(rotation=30)
    plt.ylabel("Energy (J)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_gpu_events.pdf")
    plt.clf()

    plt.bar(exp, np.array(energy_gpu_mean) / np.array(events), label="GPU")
    plt.xticks(rotation=30)
    plt.ylabel("Energy per event (J/event)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_gpu_events_per_event.pdf")
    plt.clf()

    plt.bar(exp, np.array(energy_gpu_mean) / np.array(mean_track), label="GPU")
    plt.xticks(rotation=30)
    plt.ylabel("Energy per track (J/track)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_gpu_events_per_track.pdf")
    plt.clf()

    plt.bar(exp, np.array(time_gpu_mean), yerr=time_gpu_std, label="GPU")
    plt.xticks(rotation=30)
    plt.ylabel("Runtime (s)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "runtime_gpu_events.pdf")
    plt.clf()

    plt.bar(exp, np.array(time_gpu_mean) / np.array(events), label="GPU")
    plt.xticks(rotation=30)
    plt.ylabel("Runtime per event (s/event)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "runtime_gpu_events_per_event.pdf")
    plt.clf()

    plt.bar(exp, np.array(time_gpu_mean) / np.array(mean_track), label="GPU")
    plt.xticks(rotation=30)
    plt.ylabel("Runtime per track (s/track)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "runtime_gpu_events_per_track.pdf")
    plt.clf()


def bar_plot_energy_cpu(data, args):
    all = []
    exp = []
    sorted_keys = sort_keys(data.keys())
    inner_keys = data[sorted_keys[0]][0].keys()
    energy_norm = []

    for key in sorted_keys:
        exp.append(key)
        areas = []
        energy = []

        for k in inner_keys:
            areas.append(k)
            l = [dic[k].energy_all for dic in data[key] if dic[k].energy_all != 281475000000000.0]
            energy.append(np.mean(l))
        energy_norm.append(energy / np.sum(energy))
        all.append(energy)
    d = np.transpose(all)
    d_norm = np.transpose(energy_norm)

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

    width = 0.5
    bottom = np.zeros(len(exp))
    for i, weight in enumerate(d_norm):
        plt.bar(exp, weight, width=width, label=areas[i], bottom=bottom)
        bottom += weight

    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Energy normalized")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "energy_cpu_norm.pdf")
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

    if len(inner_keys) == 1:
        print(f"Average power CPU: {[l[0] for l in all_mean]}")
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

    print("Energy GPU: ", energy_mean)

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

    print(f"CPU measured energy: {cpu_mean}")
    print(f"GPU measured energy: {gpu_mean}")

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

    print(f"Track parameters: {mean_track}")

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
    duration_idle = []

    for key in sorted_keys:
        exp.append(key)
        runtime_gpu = []
        gpu_std = []
        runtime_cpu = 0
        duration_idle_temp = []
        for inner_key in inner_keys:
                runtime_cpu += np.mean([dic[inner_key].runtimes for dic in cpu[key]])
        for df in gpu[key]:
            runtime_gpu.append(list(df['duration'])[-1] / 1000)
            if not "util_gpu" in df:
                change_i = get_iterations_change(df["graphics"])
                if (change_i == 0):
                    change_i = get_iterations_change_percentage(df["power"], 0.1)
            else:
                change_i = first_higher_then(df["util_gpu"], 5)
            duration_idle_temp.append(df.get("duration")[change_i])

        duration_idle.append(np.mean(duration_idle_temp)/1000)
        gpu_mean.append(np.mean(runtime_gpu))
        gpu_std.append(np.std(runtime_gpu))
        cpu_mean.append(runtime_cpu)

    print(f"GPU idle time: {duration_idle}")
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

    print("CPU Runtime: ", cpu_mean)
    print("GPU Runtime: ", gpu_mean)

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

    plt.bar(ind, height=np.array(gpu_mean), yerr=gpu_std)
    plt.xticks(ind, exp, rotation=30)
    plt.ylabel("Runtime (s)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "runtime_gpu.pdf")
    plt.clf()



def box_plot_average_power_gpu(data, program, args):
    sorted_keys = sort_keys(data.keys())
    exp = []
    power_all = []
    gpu_type = ""
    idle_power_list = []

    for key in sorted_keys:
        exp.append(key)
        power = []
        for df in data[key]:
            power.extend(list(df['power']))
            idle_power_list.append(df['power'][0])
        power_all.append(power)
        gpu_type = program[key][0]['gpu']

    if gpu_type == "A4000":
        max_power = 140
    elif gpu_type == "A6000":
        max_power = 300
    elif gpu_type == "A2":
        max_power = 60
    elif gpu_type == "A100-PCIE-40GB":
        max_power = 250

    idle = np.mean(idle_power_list)

    plt.axhline(y=max_power, color='black', linestyle='--', label="Maximum Power")
    plt.axhline(y=idle, color='gray', linestyle='--', label="Averaged Idle Power")
    plt.legend()
    plt.xticks(rotation=30)
    plt.boxplot(power_all, showfliers=False, labels=exp)
    plt.tight_layout()
    plt.ylabel("Average Power (W)")
    plt.xlabel("Input")
    plt.ylim(0)
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_power_gpu.pdf")
    plt.clf()


    exp = []
    power_mean = []
    power_std = []
    power_idle_mean = []
    power_all = []


    for key in sorted_keys:
        exp.append(key)
        power = []
        power_idle = []
        power2 = []
        for df in data[key]:
            if not "util_gpu" in df:
                change_i = get_iterations_change(df["graphics"])
                if (change_i == 0):
                    change_i = get_iterations_change_percentage(df["power"], 0.1)
            else:
                change_i = first_higher_then(df["util_gpu"], 5)
            power2.extend(list(df['power'])[change_i:])
            power.append(np.mean(list(df['power'])[change_i:]))
            power_idle.append(np.mean(list(df['power'])[:change_i]))

        power_idle_mean.append(np.mean(power_idle))
        power_mean.append(np.mean(power))
        power_all.append(power2)
        power_std.append(np.std(power))

    plt.xticks(rotation=30)
    print("Mean power gpu: ", power_mean)
    print("Average idle power gpu:", power_idle_mean)
    plt.axhline(y=max_power, color='black', linestyle='--', label="Maximum Power")
    plt.axhline(y=idle, color='gray', linestyle='--', label="Averaged Idle Power")
    plt.legend()
    # plt.bar(exp, power_mean, yerr=power_std)
    plt.boxplot(power_all, showfliers=False, labels=exp)
    plt.tight_layout()
    plt.ylabel("Average Power (W)")
    plt.xlabel("Input")
    plt.ylim(0)
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_power_during_events_gpu.pdf")
    plt.clf()

def box_plot_average_freq_gpu(data, args):
    sorted_keys = sort_keys(data.keys())
    exp = []
    freq_all_events = []
    freq_all = []

    for key in sorted_keys:
        exp.append(key)
        freq = []
        freq_events = []
        for df in data[key]:
            freq.extend(list(df['graphics']))
            if not "util_gpu" in df:
                change_i = get_iterations_change(df["graphics"])
                if (change_i == 0):
                    change_i = get_iterations_change_percentage(df["power"], 0.1)
            else:
                change_i = first_higher_then(df["util_gpu"], 5)
            freq_events.extend(list(df['graphics'])[change_i:])

        freq_all.append(freq)
        freq_all_events.append(freq_events)


    plt.xticks(rotation=30)
    plt.boxplot(freq_all, showfliers=False, labels=exp)
    plt.tight_layout()
    plt.ylabel("Frequency (MHz)")
    plt.xlabel("Input")
    plt.ylim(0, 2000)
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_freq_gpu.pdf")
    plt.clf()


    plt.xticks(rotation=30)
    plt.boxplot(freq_all_events, showfliers=False, labels=exp)
    plt.tight_layout()
    plt.ylabel("Frequency (MHz)")
    plt.xlabel("Input")
    plt.ylim(0, 2000)
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_freq_during_events_gpu.pdf")
    plt.clf()

def bar_plot_util_gpu(data, args):
    sorted_keys = sort_keys(data.keys())
    exp = []
    util_mean = []
    util_std = []

    for key in sorted_keys:
        exp.append(key)
        util = []
        for df in data[key]:
            change_i = first_higher_then(df["util_gpu"], 5)
            util.append(np.mean(list(df['util_gpu'])[change_i:]))

        util_mean.append(np.mean(util))
        util_std.append(np.std(util))

    plt.xticks(rotation=30)
    print("Mean util gpu: ", util_mean)
    plt.bar(exp, util_mean, yerr=util_std)
    plt.tight_layout()
    plt.ylabel("Average utilization GPU (%)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "average_utilization_gpu.pdf")
    plt.clf()

def bar_plot_runtime_cpu(data, args):
    all = []
    exp = []
    runtime_norm = []

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
        runtime_norm.append(runtime / np.sum(runtime))
    d = np.transpose(all)
    d_norm = np.transpose(runtime_norm)

    width = 0.5
    bottom = np.zeros(len(exp))
    for i, weight in enumerate(d):
        if (len(d) > 1):
            plt.bar(exp, weight, width=width, label=areas[i], bottom=bottom)
        else:
            plt.bar(exp, weight)
        bottom += weight

    plt.xticks(rotation=30)
    if (len(d) > 1):
        plt.legend()
    plt.ylabel("Runtime (s)")
    plt.xlabel("Input")
    plt.plot()
    plt.savefig(args.output + "runtime_cpu.pdf")
    plt.clf()

    width = 0.5
    bottom = np.zeros(len(exp))

    for i, weight in enumerate(d_norm):
        plt.bar(exp, weight, width=width, label=areas[i], bottom=bottom)
        bottom += weight

    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Runtime normalized")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "runtime_cpu_norm.pdf")
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
        box_plot_average_power_gpu(experiments_gpu, experiments_program, args)
        box_plot_average_freq_gpu(experiments_gpu, args)
        bar_plot_energy_time_for_events(experiments_gpu, experiments_cpu, experiments_program, args)

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

        box_plot_average_power_gpu(experiments_gpu, experiments_program, args)
        box_plot_average_freq_gpu(experiments_gpu, args)

        bar_plot_track_paramaters(experiments_program, args)
        bar_plot_avg_freq_cpu(experiments_cpu, args)
        bar_plot_avg_power_cpu(experiments_cpu, args)
    elif(args.experiment == 2):
        bar_plot_util_gpu(experiments_gpu, args)

if __name__ == "__main__":
    main()