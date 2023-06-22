
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime


def take_first(str):
    return int(str.split(',')[0])

def read_data_gpu(input):
    df = pd.read_csv(input, delimiter=',')
    df = df.rename(columns={'Function Name': 'function_name', 'Cycles [cycle]': 'cycles', 'Compute Throughput [%]': 'compute_throughput',
                            'Memory Throughput [%]': 'memory_throughput', "# Registers [register/thread]": "register/thread", "Grid Size": "grid_size", "Block Size": "block_size"})
    df = df.drop('Demangled Name', axis=1)
    df = df.drop('Process', axis=1)
    device_name = list(df['Device Name'])[0]
    df = df.drop('Device Name', axis=1)
    df = df.drop('ID', axis=1)
    df = df.drop('Issues Detected', axis=1)
    # df = df.drop('Duration [msecond]', axis=1)

    if df['cycles'].dtype == object:
        df['cycles'] = df['cycles'].str.replace('.', '').astype(int)
    df['grid_size'] = df['grid_size'].apply(take_first)
    df['block_size'] = df['block_size'].apply(take_first)
    df['compute_throughput'] = df['compute_throughput'].str.replace(',', '.').astype(float)
    df['memory_throughput'] = df['memory_throughput'].str.replace(',', '.').astype(float)
    df['register/thread'] = df['register/thread'].astype(int)

    df_mean = df.groupby(df['function_name']).aggregate("mean")
    df_std = df.groupby(df['function_name']).aggregate("std")
    print(df_mean)
    weight = np.array(df_mean['grid_size']) / np.sum(df_mean['grid_size'])
    print(weight)
    print(f"Size weighted: {np.average(np.array(df_mean['grid_size'])*np.array(df_mean['block_size']), weights=weight)}")
    print(f"Size sum: {np.average(np.array(df_mean['grid_size'])*np.sum(df_mean['block_size']))}")
    print(f"Compute weighted: {np.average(np.array(df_mean['compute_throughput']), weights=weight)}")
    print(f"Memory weighted: {np.average(np.array(df_mean['memory_throughput']), weights=weight)}")


    return (df_mean, df_std), device_name

def sort_keys(l):
    return sorted(l, key=lambda s: int(''.join(filter(str.isdigit, s))))

def create_bar_plot(inverted_mean, inverted_std, sorted_keys, labels, sorted_bool=True):

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    length = len(inverted_mean)
    width = 0.8/length
    ind = np.arange(len(sorted_keys))
    ind_mean = np.flip((np.arange(length) - np.mean(np.arange(length)))*width)

    if sorted_bool:
        order = np.mean(inverted_mean, axis=1)
        sorted_inverted_mean = [x for _,x in sorted(zip(order,inverted_mean), reverse=True)]
        sorted_inverted_std = [x for _,x in sorted(zip(order,inverted_std), reverse=True)]
        sorted_labels = [x for _,x in sorted(zip(order,labels), reverse=True)]
        sorted_colors = [x for _,x in sorted(zip(order,colors), reverse=True)]
    else:
        sorted_inverted_mean = inverted_mean
        sorted_inverted_std = inverted_std
        sorted_labels = labels
        sorted_colors = colors


    for i, weight in enumerate(sorted_inverted_mean):
        plt.bar(ind-ind_mean[i], height=weight, yerr=sorted_inverted_std[i], width=width, label=sorted_labels[i], color=sorted_colors[i])

    plt.xticks(ind, sorted_keys, rotation=30)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')


def bar_plot_cycles(args, data, device_name):
    sorted_keys = sort_keys(data.keys())

    mean_dfs = []
    std_dfs = []
    for key in sorted_keys:
        mean_dfs.append(data[key][0])
        std_dfs.append(data[key][1])

    labels = mean_dfs[0].index.values.tolist()

    inverted_cycles_mean = np.transpose([list(df['cycles']) for df in mean_dfs])
    inverted_cycles_std = np.transpose([list(df['cycles']) for df in std_dfs])

    create_bar_plot(inverted_cycles_mean, inverted_cycles_std, sorted_keys, labels)
    plt.ylabel("Number of cycles")
    plt.xlabel("Input")
    plt.yscale('log')
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_cycles_plot.pdf")
    plt.clf()

    inverted_size_mean = np.transpose([np.array(df['grid_size'])*np.array(df['block_size']) for df in mean_dfs])
    inverted_size_std = np.transpose([np.array(df['grid_size'])*np.array(df['block_size']) for df in std_dfs])

    create_bar_plot(inverted_size_mean, inverted_size_std, sorted_keys, labels)
    plt.ylabel("Grid size * Block size")
    plt.xlabel("Input")
    plt.yscale('log')
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_gridsize*blocksize_plot.pdf")
    plt.clf()

    cycles_per_size = np.array(inverted_cycles_mean) / np.array(inverted_size_mean)

    create_bar_plot(cycles_per_size, np.zeros(len(inverted_size_std)), sorted_keys, labels)
    plt.ylabel("Cycles / (Grid size * Block size)")
    plt.xlabel("Input")
    plt.yscale('log')
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_cycle_per_size_plot.pdf")
    plt.clf()

    inverted_compute_mean = np.transpose([list(df['compute_throughput']) for df in mean_dfs])
    inverted_compute_std = np.transpose([list(df['compute_throughput']) for df in std_dfs])


    create_bar_plot(inverted_compute_mean, inverted_compute_std, sorted_keys, labels)
    plt.ylabel("Compute Throughput (%)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_compute_throughput_plot.pdf")
    plt.clf()

    norm_compute = [np.array(x)/x[0] for x in inverted_compute_mean]

    create_bar_plot(norm_compute, np.zeros(len(norm_compute)), sorted_keys, labels)
    plt.ylabel("Compute Throughput normalized to mu20")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_compute_norm_throughput_plot.pdf")
    plt.clf()

    inverted_memory_mean = np.transpose([list(df['memory_throughput']) for df in mean_dfs])
    inverted_memory_std = np.transpose([list(df['memory_throughput']) for df in std_dfs])

    create_bar_plot(inverted_memory_mean, inverted_memory_std, sorted_keys, labels)
    plt.ylabel("Memory Throughput (%)")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_memory_throughput_plot.pdf")
    plt.clf()

    norm_memory = [np.array(x)/x[0] for x in inverted_memory_mean]

    create_bar_plot(norm_memory, np.zeros(len(norm_memory)), sorted_keys, labels)
    plt.ylabel("Memory Throughput normalized to mu20")
    plt.xlabel("Input")
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_memory_norm_throughput_plot.pdf")
    plt.clf()


    if device_name == "NVIDIA RTX A4000":
        cores = 6144
    elif device_name == "NVIDIA RTX A6000":
        cores = 10752
    elif device_name == "NVIDIA A2":
        cores = 1280
    elif device_name == "NVIDIA A100-PCIE-40GB":
        cores = 6912
    else:
        return

    create_bar_plot(inverted_cycles_mean / cores, inverted_cycles_std / cores, sorted_keys, labels)
    plt.ylabel("Number of cycles / # cores")
    plt.xlabel("Input")
    plt.yscale('log')
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_cycles_per_core_plot.pdf")
    plt.clf()

    create_bar_plot(inverted_size_mean / cores, np.zeros(len(inverted_size_std)), sorted_keys, labels)
    plt.ylabel("(Grid size * Block size) / # cores")
    plt.xlabel("Input")
    plt.yscale('log')
    plt.plot()
    plt.tight_layout()
    plt.savefig(args.output + "bar_size_per_core_plot.pdf")
    plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs='+', help="Name of directories of the data")
    parser.add_argument("--output", default="graphs/", help="Name of directory of the graph")

    parser.add_argument("--experiment", type=int, default=0, help="What experiment to create graph for")

    args = parser.parse_args()

    if len(args.input) <= 1:
        input = args.input[0]
        name_exp = input.split("/")[-1] if input.split("/")[-1] != '' else input.split("/")[-2]

        args.output = args.output + name_exp + "/"
        if not os.path.exists(args.output):
            os.mkdir(args.output)

        experiments_gpu = {}
        for (dirpath, dirnames, filenames) in os.walk(input):
            if len(filenames) > 1:
                gpu_data = [s for s in filenames if s.endswith('.csv')]
                name = dirpath.split('/')[-1]
                experiments_gpu[name], device_name = read_data_gpu(f"{dirpath}/{gpu_data[0]}" )

        bar_plot_cycles(args, experiments_gpu, device_name)
    # else:
    #     now = datetime.now()
    #     now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    #     args.output = f"{args.output}combined_ncu_graph_{now_str}/"
    #     if not os.path.exists(args.output):
    #         os.mkdir(args.output)

    #     with open(args.output + '/data', 'w') as f:
    #         f.write(" ".join(args.input))

    #     if not os.path.exists(args.output):
    #         os.mkdir(args.output)\

    #     input_list = []
    #     for input in args.input:
    #         experiments_gpu = {}
    #         for (dirpath, dirnames, filenames) in os.walk(input):
    #             if len(filenames) > 1:
    #                 gpu_data = [s for s in filenames if s.endswith('.csv')]
    #                 name = dirpath.split('/')[-1]
    #                 experiments_gpu[name], _ = read_data_gpu(f"{dirpath}/{gpu_data[0]}" )
    #         input_list.append(experiments_gpu)


if __name__ == "__main__":
    main()