import gc
import sys
import time
import torch
import numpy as np
import lightning as pl
import pandas as pd
from itertools import product
from argparse import ArgumentParser

from nablafx.models import GreyBoxModel
from nablafx.processors import *

processors = [
    ParametricEQ(
        sample_rate=48000,
        control_type="static",
    ),
    Gain(
        sample_rate=48000,
        control_type="static",
    ),
    DCOffset(
        sample_rate=48000,
        control_type="static",
    ),
    StaticRationalNonlinearity(sample_rate=48000, degrees=[6, 5], init_approx_func="tanh"),
    Gain(
        sample_rate=48000,
        control_type="static",
    ),
    ParametricEQ(
        sample_rate=48000,
        control_type="static",
    ),
    Lowpass(sample_rate=48000, control_type="static"),
]


def run(
    processors=processors,
    num_controls=0,
    stat_control_params_initial=0.0,
    stat_cond_num_layers=3,
    stat_cond_hidden_dim=16,
    dyn_block_size=128,
    dyn_num_layers=1,
    dyn_cond_block_size=128,
    dyn_cond_num_layers=1,
    N=48000,
):
    dict_args = {}
    dict_args["processors"] = processors
    dict_args["num_controls"] = num_controls
    dict_args["stat_control_params_initial"] = stat_control_params_initial
    dict_args["stat_cond_num_layers"] = stat_cond_num_layers
    dict_args["stat_cond_hidden_dim"] = stat_cond_hidden_dim
    dict_args["dyn_block_size"] = dyn_block_size
    dict_args["dyn_num_layers"] = dyn_num_layers
    dict_args["dyn_cond_block_size"] = dyn_cond_block_size
    dict_args["dyn_cond_num_layers"] = dyn_cond_num_layers

    sr = 48000
    duration = N / sr  # seconds
    n_iters = 500
    timings = []

    model = GreyBoxModel(**dict_args)  # create the model with args

    samples = N
    input = (torch.rand(1, 1, samples) * 2) - 1

    if dict_args["num_controls"] > 0:
        controls = torch.rand(1, 1, dict_args["num_controls"])
    else:
        controls = None

    gc.disable()  # disable garbage collection

    ### CRITICAL SECTION

    model.eval()
    with torch.no_grad():
        for n in range(n_iters):
            tic = time.perf_counter()
            output = model(input, controls)
            toc = time.perf_counter()
            timings.append(toc - tic)
            # sys.stdout.write(f"{n+1:3d}/{n_iters:3d}\r")
            # sys.stdout.flush()

    ### END CRITICAL SECTION

    gc.enable()  # enable garbage collection

    mean_time_s = np.mean(timings)
    mean_time_ms = mean_time_s * 1e3
    sec_sec = (1 / duration) * mean_time_s
    rtf = duration / mean_time_s
    print(f"Avg. time: {mean_time_ms:0.1f} ms  | sec/sec {sec_sec:0.3f} |  RTF: {rtf:0.2f}x")

    return rtf


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parser.parse_args()

    time.sleep(10)  # give time to write psw if running with priority

    pl.seed_everything(42)  # set the seed

    candidates = []
    frame_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # , 32768, 65536]
    # frame_sizes = [128, 256, 512, 1024]
    hidden_size = [32, 96]
    num_layers = [1, 1]

    model_id = ["GB-DIST-RNL"]

    for mid in model_id:

        for N in frame_sizes:
            print()
            print(f"{mid}, frame size: {N}")
            rtf = run(
                N=N,
            )
            # if c:   mid += "-C"
            # else:   mid += "-N"
            candidates.append({"model_id": mid, "rtf": rtf, "N": N})

    df = pd.DataFrame(candidates)
    df.to_csv("tests/speed_cpu_gb-dist-rnl.csv")

    print("-" * 120)
    print("\tID\t\tN\tRTF")
    print("-" * 120)
    for n, c in enumerate(candidates):
        print(f"{n: 3d}\t{c['model_id']}\t\t{c['N']}\t{c['rtf']: 2.2f}x")
    print("-" * 120)
