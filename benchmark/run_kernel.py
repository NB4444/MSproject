#!/usr/bin/python3

import argparse
import os
import json
import datetime
import subprocess

MAX_THREADS = 24

def create_core_string(threads):
    threads = min(threads, MAX_THREADS)
    if threads == 0:
        return "0"
    else:
        return f"0-{threads-1}"

def run(runs, program, node=None, experiment=None, time_limit=15, numBlocks=0, numThreads=0, gpu=None, output_dir="", filename="run", num_runs=1000, type=0):
    for i in range(runs):
        with open(filename, "w") as f:
            setup = f"""#!/bin/bash
#SBATCH --job-name={experiment.replace(' ', '_')}_{i}
#SBATCH --account=nbreed
#SBATCH --nodes=1
#SBATCH --time=00:{time_limit}:00
#SBATCH --error="{output_dir}job{i}.err"
#SBATCH --output="{output_dir}job{i}.out"
#SBATCH -C {gpu}
#SBATCH --gres=gpu:1
{f"#SBATCH --nodelist={node}" if node is not None else ""}
{"#SBATCH -p fatq" if gpu == "A100" else ""}

module load cuda11.7/toolkit
sleep 5
"""
            if type == 0:
                exp = f"""
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr,utilization.gpu,utilization.memory --format=csv --id=0 -lms 1 -f {output_dir}gpu{i}.csv &
./{program} {numBlocks} {numThreads} {num_runs}
                """
            elif type == 1:
                last_dir_name = output_dir.split('/')[-2]
                exp = f"""
ncu -o {output_dir}profile_{last_dir_name}_{i} ./{program} {numBlocks} {numThreads} {num_runs}
                """
            else:
                exp = ""

            f.write(setup + exp)
        os.chmod(filename, 0o777)
        result = subprocess.call(f"sbatch ./{filename}", shell=True)

def experiment_comp(args, output, total_sm, cores_per_sm):
    name = f"Compute intensive kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute", node=args.node, experiment=name, output_dir=output_dir, filename=f"run1", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000)

    name = f"Memory intensive kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000)

    name = f"compute_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000, type=1)

    name = f"compute_memory_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, type=1)

def experiment_comp_bigger_blocks(args, output, total_sm, cores_per_sm):
    name = f"Compute intensive kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute", node=args.node, experiment=name, output_dir=output_dir, filename=f"run1", gpu=args.gpu, numBlocks=total_sm*4, numThreads=cores_per_sm, num_runs=1000000000)

    name = f"Memory intensive kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm*4, numThreads=cores_per_sm, num_runs=10000000000)

    name = f"compute_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm*4, numThreads=cores_per_sm, num_runs=10000000, type=1)

    name = f"compute_memory_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm*4, numThreads=cores_per_sm, num_runs=100000000, type=1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="A4000", help="The type of gpu")
    parser.add_argument("--node", type=str, default="node005", help="The node it is run on")
    parser.add_argument("--experiment", type=int, default=0, help="Type of experiment")

    args = parser.parse_args()

    if args.gpu == "A4000":
        total_sm = 48
        cores_per_sm = 128
    if args.gpu == "A6000":
        total_sm = 84
        cores_per_sm = 128
    if args.gpu == "A2":
        total_sm = 10
        cores_per_sm = 128
    if args.gpu == "A100":
        total_sm = 108
        cores_per_sm = 64

    now = datetime.datetime.now()

    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")


    if args.experiment == 0:
        output = f"comp_bench_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp(args, output, total_sm, cores_per_sm)
    elif args.experiment == 1:
        output = f"comp_bench_bigger_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_bigger_blocks(args, output, total_sm, cores_per_sm)




if __name__ == "__main__":
    main()