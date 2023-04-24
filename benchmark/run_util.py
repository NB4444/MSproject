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

def run(runs, node=None, experiment=None, time_limit=15, numBlocks=0, numThreads=0, gpu=None, output_dir="", filename="run"):
    for i in range(runs):
        with open(filename, "w") as f:
            setup = f"""#!/bin/bash
#SBATCH --job-name={experiment}_{i}
#SBATCH --account=nbreed
#SBATCH --nodes=1
#SBATCH --time=00:{time_limit}:00
#SBATCH --error={output_dir}job{i}.err
#SBATCH --output={output_dir}job{i}.out
#SBATCH -C {gpu}
#SBATCH --gres=gpu:1
{f"#SBATCH --nodelist={node}" if node is not None else ""}
{"#SBATCH -p fatq" if gpu == "A100" else ""}

module load cuda11.7/toolkit
sleep 5
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr,utilization.gpu,utilization.memory --format=csv --id=0 -lms 1 -f {output_dir}gpu{i}.csv &
./benchmark {numBlocks} {numThreads}
                """


            f.write(setup)
        os.chmod(filename, 0o777)
        result = subprocess.call(f"sbatch ./{filename}", shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="A4000", help="The type of gpu")
    parser.add_argument("--node", type=str, default="node005", help="The node it is run on")

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

    total = total_sm*cores_per_sm

    now = datetime.datetime.now()

    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    output = f"util_bench_{args.gpu}_{now_str}/"

    print(output)

    os.mkdir(output)


    for num_blocks in range(2, total_sm+1, 2):
        perc_sm = (num_blocks / total_sm) * 100
        print(f"Percentage of sms: {perc_sm}")
        for threads in [cores_per_sm//4, cores_per_sm//2, cores_per_sm]:
            perc_threads = ((threads*num_blocks) / total) * 100
            print(f"Percentage of cores: {perc_threads}")
            name = f"{num_blocks}_{threads}"
            output_dir = f"{output}{name}/"
            os.mkdir(output_dir)
            run(1, node=args.node, experiment=name, output_dir=output_dir, filename=f"run", gpu=args.gpu, numBlocks=num_blocks, numThreads=threads)

if __name__ == "__main__":
    main()