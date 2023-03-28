import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def to_ms(diff):
    return int(diff.total_seconds() *1000)

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

    args = parser.parse_args()

    dataframes = []
    names = []
    for (dirpath, dirnames, filenames) in os.walk(args.input):
        if len(filenames) > 1:
            for filename in filenames:
                dataframes.append(read_data_gpu(f"{dirpath}/{filename}"))
                names.append(filename)

    print("================================")
    for i, df in enumerate(dataframes):
        print(names[i])
        print("Max", np.max(df['power']))
        print("Min", np.min(df['power']))
        print("Mean", np.mean(df['power']))
        print("25 percentile", np.quantile(df['power'], 0.25))
        print("75% percentile", np.quantile(df['power'], 0.75))
        print("90% percentile", np.quantile(df['power'], 0.90))
        print("Frequency Max", np.max(df['graphics']))
        print("Frequency Min", np.min(df['graphics']))
        print("Frequency Median", np.median(df['graphics']))
        print("Frequency Mean", np.mean(df['graphics']))
        print("================================")

    print("\nRemoved values below start values")
    print("================================")
    for i, df in enumerate(dataframes):
        start_power = df['power'][0]
        df = df[df.power >= start_power]
        start_freq = df['graphics'][0]
        df = df[df.graphics >= start_freq]
        print(names[i])
        print("Max", np.max(df['power']))
        print("Min", np.min(df['power']))
        print("Mean", np.mean(df['power']))
        print("25 percentile", np.quantile(df['power'], 0.25))
        print("75% percentile", np.quantile(df['power'], 0.75))
        print("90% percentile", np.quantile(df['power'], 0.90))
        print("Frequency Max", np.max(df['graphics']))
        print("Frequency Min", np.min(df['graphics']))
        print("Frequency Median", np.median(df['graphics']))
        print("Frequency Mean", np.mean(df['graphics']))
        print("================================")

if __name__ == "__main__":
    main()