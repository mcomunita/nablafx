import gc
import sys
import time
import torch
import numpy as np
import lightning as pl
import pandas as pd
from itertools import product
from argparse import ArgumentParser

from nablafx.lstm import LSTM


def run(
    num_inputs=1,
    num_outputs=1,
    num_controls=0,
    hidden_size=1,
    num_layers=1,
    residual=False,
    direct_path=False,
    cond_type=None,  # null, fixed, tvcond
    cond_block_size=128,  # block size for tvcond
    cond_num_layers=1,  # number of lstm layers for tvcond
    N=48000,
):
    dict_args = {}
    dict_args["num_inputs"] = num_inputs
    dict_args["num_outputs"] = num_outputs
    dict_args["num_controls"] = num_controls
    dict_args["hidden_size"] = hidden_size
    dict_args["num_layers"] = num_layers
    dict_args["residual"] = residual
    dict_args["direct_path"] = direct_path
    dict_args["cond_type"] = cond_type
    dict_args["cond_block_size"] = cond_block_size
    dict_args["cond_num_layers"] = cond_num_layers

    sr = 48000
    duration = N / sr  # seconds
    n_iters = 1000
    timings = []

    model = LSTM(**dict_args)  # create the model with args

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

    model_id = ["LSTM-32", "LSTM-96"]

    for mid, h, l in zip(
        model_id,
        hidden_size,
        num_layers,
    ):

        for N in frame_sizes:
            print()
            print(f"{mid}: {h}, {l}, frame size: {N}")
            rtf = run(
                hidden_size=h,
                num_layers=l,
                N=N,
            )
            # if c:   mid += "-C"
            # else:   mid += "-N"
            candidates.append({"model_id": mid, "hidden_size": h, "num_layers": l, "rtf": rtf, "N": N})

    df = pd.DataFrame(candidates)
    df.to_csv("tests/speed_cpu_lstm.csv")

    print("-" * 120)
    print("\tID\t\t\tHid.\tLay.\tN\tRTF")
    print("-" * 120)
    for n, c in enumerate(candidates):
        print(f"{n: 3d}\t{c['model_id']}\t\t{c['hidden_size']}\t{c['num_layers']}\t{c['N']}\t{c['rtf']: 2.2f}x")
    print("-" * 120)
