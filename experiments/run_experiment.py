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

def run(exp, node=None, experiment=None, time_limit=None, gpu=None, input_dir=None, exec=None, events=None, threads=None, marker=None, output_dir="", t=0, filename="run"):
    for i in range(exp['runs']):
        with open(filename, "w") as f:
            setup = f"""#!/bin/bash
#SBATCH --job-name={exp['experiment_name'] if experiment == None else experiment}_{i}
#SBATCH --account=nbreed
#SBATCH --nodes=1
#SBATCH --time=00:{exp['time_limit'] if time_limit == None else time_limit}:00
#SBATCH --error={output_dir}job{i}.err
#SBATCH --output={output_dir}job{i}.out
#SBATCH -C {exp['gpu'] if gpu == None else gpu}
#SBATCH --gres=gpu:1
{f"#SBATCH --nodelist={node}" if node is not None else ""}
{"#SBATCH -p fatq" if (exp['gpu'] == "A100" and gpu == None) or gpu == "A100" else ""}

module load cuda11.7/toolkit
sleep 5"""

            if t == 0:
                r = f"""
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv --id=0 -lms 1 -f {output_dir}gpu{i}.csv &
likwid-perfctr -C { create_core_string(exp['threads'] if threads == None else threads) } -g ENERGY {'-m' if (exp['marker'] if marker == None else marker) else ''} -o {output_dir}cpu{i}.csv {exp['exec'] if exec == None else exec} --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory={exp['input_dir'] if input_dir == None else input_dir}/ --loaded_events 10 --processed_events {exp['events'] if events == None else events} --cold_run_events 0 --threads {exp['threads'] if threads == None else threads}
                """
            if t == 1:
                last_dir_name = output_dir.split('/')[-2]
                r = f"""
nsys profile --output={output_dir}report_{last_dir_name}_{i}.nsys-rep {exp['exec'] if exec == None else exec} --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory={exp['input_dir'] if input_dir == None else input_dir}/ --loaded_events 10 --processed_events {exp['events'] if events == None else events} --cold_run_events 0 --threads {exp['threads'] if threads == None else threads}
                """

            if t == 2:
                last_dir_name = output_dir.split('/')[-2]
                r = f"""
ncu -o {output_dir}profile_{last_dir_name}_{i} {exp['exec'] if exec == None else exec} --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory={exp['input_dir'] if input_dir == None else input_dir}/ --loaded_events 10 --processed_events {exp['events'] if events == None else events} --cold_run_events 0 --threads {exp['threads'] if threads == None else threads}
                """

            f.write(setup + r)
        os.chmod(filename, 0o777)
        result = subprocess.call(f"sbatch ./{filename}", shell=True)

def start_experiment(exp, t):
    now = datetime.datetime.now()

    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    output = f"{exp['experiment_name']}_{now_str}/"

    print(output)

    os.mkdir(output)

    with open(output + '/parameters', 'w') as f:
        f.write(json.dumps(exp))

    if "nodes" in exp:
        node = exp['nodes']
    else:
        node = None

    if type(exp['input_dir']) == list:
        for i, input_dir in enumerate(exp['input_dir']):
            input_name = input_dir.split('/')[1]
            name = exp['experiment_name'] + "-" + input_name
            output_dir = f"{output}{input_name}/"
            os.mkdir(output_dir)
            run(exp, node, experiment=name, input_dir=input_dir, output_dir=output_dir, t=t, filename=f"run{i}")
    elif(type(exp['threads']) == list):
        for i, n_threads in enumerate(exp['threads']):
            input_name = f"threads_{n_threads}"
            name = exp['experiment_name'] + "-" + input_name
            output_dir = f"{output}{input_name}/"
            os.mkdir(output_dir)
            run(exp, node, experiment=name, threads=n_threads, output_dir=output_dir, t=t, filename=f"run{i}")
    elif(type(exp['events']) == list):
        for i, event in enumerate(exp['events']):
            input_name = f"events_{event}"
            name = exp['experiment_name'] + "-" + input_name
            output_dir = f"{output}{input_name}/"
            os.mkdir(output_dir)
            run(exp, node, experiment=name, events=event, output_dir=output_dir, t=t, filename=f"run{i}")
    elif(type(exp['gpu']) == list):
        for i, gpu in enumerate(exp['gpu']):
            input_name = f"gpu_{gpu}"
            name = exp['experiment_name'] + "-" + input_name
            output_dir = f"{output}{input_name}/"
            os.mkdir(output_dir)
            run(exp, node, experiment=name, gpu=gpu, output_dir=output_dir, t=t, filename=f"run{i}")
    elif("nodes" in exp and type(exp['nodes']) == list):
        for i, node in enumerate(exp['nodes']):
            input_name = f"{node}"
            name = exp['experiment_name'] + "-" + input_name
            output_dir = f"{output}{input_name}/"
            os.mkdir(output_dir)
            run(exp, node, experiment=name, output_dir=output_dir, t=t, filename=f"run{i}")
    else:
        run(exp, node, output_dir=output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="experiments/example.json", help="JSON file with experiment")
    parser.add_argument("--type", type=int, default=0, help="The type of experiment")

    args = parser.parse_args()

    with open(args.experiment, "r") as f:
        exp = json.loads(f.read())

    start_experiment(exp, args.type)

if __name__ == "__main__":
    main()