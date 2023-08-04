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

def run(runs, program, node=None, experiment=None, time_limit=15, numBlocks=0, numThreads=0, gpu=None, output_dir="", filename="run", num_runs=1000, type=0, n=None, m=None):
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
./{program} {numBlocks} {numThreads} {num_runs} {n if n is not None else ""} {m if m is not None else ""}
                """
            elif type == 1:
                last_dir_name = output_dir.split('/')[-2]
                exp = f"""
ncu --set full -o {output_dir}profile_{last_dir_name}_{i} ./{program} {numBlocks} {numThreads} {num_runs} {n if n is not None else ""} {m if m is not None else ""}
                """
            elif type == 2:
                last_dir_name = output_dir.split('/')[-2]
                exp = f"""
nsys profile --output={output_dir}report_{last_dir_name}_{i}.nsys-rep ./{program} {numBlocks} {numThreads} {num_runs} {n if n is not None else ""} {m if m is not None else ""}
                """
            elif type == 3:
                exp = f"""
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr,utilization.gpu,utilization.memory --format=csv --id=0 -lms 1 -f {output_dir}gpu{i}.csv &
likwid-perfctr -C { create_core_string(numThreads) } -g ENERGY -o {output_dir}cpu{i}.csv ./{program} {numBlocks} {numThreads} {num_runs} {n if n is not None else ""} {m if m is not None else ""}
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
    run(5, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run1", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=50000000000)

    name = f"Memory intensive kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000)

    name = f"compute_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000, type=1)

    name = f"compute_memory_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, type=1)

def experiment_comp_ncu(args, output, total_sm, cores_per_sm):
    name = f"compute_float_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, type=1)

    name = f"compute_double_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, type=1)

    name = f"compute_int_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_int", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, type=1)

    name = f"compute_memory_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000, type=1)

    name = f"memory_worse_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_memory_offset", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000, type=1)

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

def experiment_comp_types(args, output, total_sm, cores_per_sm):
    name = f"Floating point kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=50000000000)

    name = f"Double precision kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000)

    name = f"Integer kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_int", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000000)

    name = f"compute_float_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000, type=1)

    name = f"compute_double_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, type=1)

    name = f"compute_int_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_int", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000, type=1)

def experiment_comp_float_double(args, output, total_sm, cores_per_sm, float_frac):
    name = f"1 FP32 and 1 FP64 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_float_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000)

    name = f"{float_frac} FP32 and 1 FP64 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_floatN_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=500000000, n=float_frac)

    name = f"FP32 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=50000000000)

    name = f"FP64 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000)

    name = f"compute_float_double_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_float_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, type=1)

    name = f"compute_floatN_double_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_floatN_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=50000000, type=1, n=float_frac)

    name = f"compute_float_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, type=1)

    name = f"compute_double_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_compute_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, type=1)

def experiment_comp_float_double_2(args, output, total_sm, cores_per_sm):
    name = f"1 FP32 and 1 FP64 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(3, "benchmark_compute_floatN_doubleM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, n=1, m=1)

    name = f"1 FP32 and 2 FP64 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(3, "benchmark_compute_floatN_doubleM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, n=1, m=2)

    name = f"2 FP32 and 2 FP64 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(3, "benchmark_compute_floatN_doubleM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, n=2, m=2)

    name = f"64 FP32 and 2 FP64 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(3, "benchmark_compute_floatN_doubleM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, n=64, m=2)

    name = f"1 FP32 and 3 FP64 kernel"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(3, "benchmark_compute_floatN_doubleM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=100000000, n=1, m=3)


def experiment_mem(args, output, total_sm, cores_per_sm):
    name = f"Bad coalescing memory"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memory_offset", node=args.node, experiment=name, output_dir=output_dir, filename=f"run1", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=500000000)

    name = f"Good coalescing memory"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000)

    name = f"memory_worse_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_memory_offset", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=50000000, type=1)

    name = f"memory_ncu"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    run(1, "benchmark_memory", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, type=1)

def experiment_mem_stride(args, output, total_sm, cores_per_sm):
    for i in [1, 16, 32, 64, 128, 256]:
        name = f"Stride {i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(5, "benchmark_memory_offset_dyn", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, n=i, num_runs=100000000)
        run(1, "benchmark_memory_offset_dyn", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, n=i, num_runs=1000000, type=1)

def experiment_mem_stride2(args, output, total_sm, cores_per_sm):
    for i in [1, 32, 128, 256, 512, 2048]:
        name = f"Stride {i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(5, "benchmark_memory_offset_dyn2", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, n=i, num_runs=100000000)
        run(1, "benchmark_memory_offset_dyn2", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, n=i, num_runs=1000000, type=1)

def experiment_mem_stride3(args, output, total_sm, cores_per_sm):
    for i in [1, 32, 128, 256, 512, 2048]:
        name = f"Stride {i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(5, "benchmark_memory_offset_dyn3", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, n=i, num_runs=100000000)
        run(1, "benchmark_memory_offset_dyn3", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, n=i, num_runs=1000000, type=1)


def experiment_mem_compute_balance(args, output, total_sm, cores_per_sm):
    name = f"1 FP32"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=5000000000, n=0, m=1)

    name = f"1 Memory"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, n=1, m=0)

    name = f"1 FP32 and 1 Memory"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, n=1, m=1)

    name = f"1 FP32 and 2 Memory"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=50000000, n=2, m=1)

    name = f"2 FP32 and 1 Memory"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=50000000, n=1, m=2)

    name = f"1fp32"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(1, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=50000000, n=0, m=1, type=1)

    name = f"1mem"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(1, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000, n=1, m=0, type=1)

    name = f"1fp32_1mem"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(1, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000, n=1, m=1, type=1)

    name = f"1fp32_2mem"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(1, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=5000000, n=2, m=1, type=1)

    name = f"2fp32_1mem"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(1, "benchmark_memoryN_computeM", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=5000000, n=1, m=2, type=1)

def experiment_comp_scaling_float_double(args, output, total_sm, cores_per_sm, float_frac):
    for i in range(float_frac*2+1):
        name = f"{i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(1, "benchmark_compute_floatN_double", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=500000000, n=i)


def experiment_comp_float_streams(args, output, total_sm, cores_per_sm):
    for i in range(1, 20):
        name = f"{i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(1, "benchmark_compute_float_streams", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000, n=i)
        run(1, "benchmark_compute_float_streams", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000, n=i, type=2)

def experiment_comp_float_streams_cpu(args, output, total_sm, cores_per_sm):
    for i in range(1, 20):
        name = f"{i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(1, "benchmark_compute_float_streams", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000, n=i, type=3)

def experiment_comp_float_streams_cpu_threads(args, output, total_sm, cores_per_sm):
    for i in range(1, 20):
        name = f"{i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(1, "benchmark_compute_float_streams_threads", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=10000000000, n=i, type=3)


def experiment_comp_float_cpu_threads(args, output, total_sm, cores_per_sm):
    for i in range(1, 20):
        name = f"{i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(1, "benchmark_compute_float_threads", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, n=i, type=3)


def experiment_comp_memory_streams(args, output, total_sm, cores_per_sm):
    for i in range(1, 20):
        name = f"{i}"
        output_dir = f"{output}{name}/"
        os.mkdir(output_dir)
        output_dir = output_dir.replace(' ', '\ ')
        run(1, "benchmark_compute_memory_streams", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, n=i)
        run(1, "benchmark_compute_memory_streams", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, n=i, type=2)

def experiment_comp_balance_streams(args, output, total_sm, cores_per_sm):
    name = f"2 Streams {total_sm} blocks {cores_per_sm} blocksize"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_float_streams", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm, num_runs=1000000000, n=2)

    name = f"1 Stream {total_sm*2} blocks {cores_per_sm} blocksize"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm*2, numThreads=cores_per_sm, num_runs=1000000000)

    name = f"1 Stream {total_sm} blocks {cores_per_sm*2} blocksize"
    output_dir = f"{output}{name}/"
    os.mkdir(output_dir)
    output_dir = output_dir.replace(' ', '\ ')
    run(5, "benchmark_compute_float", node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=total_sm, numThreads=cores_per_sm*2, num_runs=1000000000)

8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="A4000", help="The type of gpu")
    parser.add_argument("--node", type=str, default="node005", help="The node it is run on")
    parser.add_argument("--experiment", type=int, default=0, help="Type of experiment")

    args = parser.parse_args()

    if args.gpu == "A4000":
        total_sm = 48
        cores_per_sm = 128
        float_frac = 64
    if args.gpu == "A6000":
        total_sm = 84
        cores_per_sm = 128
        float_frac = 64
    if args.gpu == "A2":
        total_sm = 10
        cores_per_sm = 128
        float_frac = 64
    if args.gpu == "A100":
        total_sm = 108
        cores_per_sm = 64
        float_frac = 2

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
    elif args.experiment == 2:
        output = f"comp_bench_types_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_types(args, output, total_sm, cores_per_sm)
    elif args.experiment == 3:
        output = f"mem_bench_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_mem(args, output, total_sm, cores_per_sm)
    elif args.experiment == 4:
        output = f"comp_bench_only_ncu_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_ncu(args, output, total_sm, cores_per_sm)
    elif args.experiment == 5:
        output = f"comp_bench_float_double_{args.gpu}_{now_str}/"
        os.mkdir(output)
        # experiment_comp_float_double(args, output, total_sm, cores_per_sm, float_frac)
        experiment_comp_float_double_2(args, output, total_sm, cores_per_sm)
    elif args.experiment == 6:
        output = f"comp_mem_balance_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_mem_compute_balance(args, output, total_sm, cores_per_sm)
    elif args.experiment == 7:
        output = f"comp_bench_scaling_float_double_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_scaling_float_double(args, output, total_sm, cores_per_sm, float_frac)
    elif args.experiment == 8:
        output = f"comp_float_streams_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_float_streams(args, output, total_sm, cores_per_sm)
    elif args.experiment == 9:
        output = f"comp_memory_streams_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_memory_streams(args, output, total_sm, cores_per_sm)
    elif args.experiment == 10:
        output = f"comp_balance_streams_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_balance_streams(args, output, total_sm, cores_per_sm)
    elif args.experiment == 11:
        output = f"comp_memory_offset_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_mem_stride(args, output, total_sm, cores_per_sm)
    elif args.experiment == 12:
        output = f"comp_memory_offsets2_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_mem_stride2(args, output, total_sm, cores_per_sm)
    elif args.experiment == 13:
        output = f"comp_memory_offsets3_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_mem_stride3(args, output, total_sm, cores_per_sm)
    elif args.experiment == 14:
        output = f"comp_float_streams_cpu_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_float_streams_cpu(args, output, total_sm, cores_per_sm)
    elif args.experiment == 15:
        output = f"comp_float_streams_cpu_threads_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_float_streams_cpu_threads(args, output, total_sm, cores_per_sm)
    elif args.experiment == 16:
        output = f"comp_float_cpu_threads_{args.gpu}_{now_str}/"
        os.mkdir(output)
        experiment_comp_float_cpu_threads(args, output, total_sm, cores_per_sm)




if __name__ == "__main__":
    main()