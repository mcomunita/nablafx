# import os
import sys
import time
import torch
import numpy as np
import lightning as pl
from nablafx.tcn import TCN
import pandas as pd
from itertools import product
from argparse import ArgumentParser


def run(
    num_inputs=1,
    num_outputs=1,
    num_controls=0,
    num_blocks=10,
    kernel_size=5,
    dilation_growth=3,
    channel_width=16,
    channel_growth=1,
    stack_size=12,
    groups=1,
    bias=True,
    causal=True,
    batchnorm=False,  # active only when cond_type = None
    residual=True,
    direct_path=False,
    cond_type=None,  # null, film, tfilm, tvfilm
    cond_block_size=128,  # block size for tfilm or tvfilm
    cond_num_layers=1,  # number of lstm layers for tfilm or tvfilm
    N=48000,
    gpu=False,
):

    pl.seed_everything(42)  # set the seed

    dict_args = {}
    dict_args["num_inputs"] = num_inputs
    dict_args["num_outputs"] = num_outputs
    dict_args["num_controls"] = num_controls
    dict_args["num_blocks"] = num_blocks
    dict_args["kernel_size"] = kernel_size
    dict_args["dilation_growth"] = dilation_growth
    dict_args["channel_width"] = channel_width
    dict_args["channel_growth"] = channel_growth
    dict_args["stack_size"] = stack_size
    dict_args["groups"] = groups
    dict_args["bias"] = bias
    dict_args["causal"] = causal
    dict_args["batchnorm"] = batchnorm
    dict_args["residual"] = residual
    dict_args["direct_path"] = direct_path
    dict_args["cond_type"] = cond_type
    dict_args["cond_block_size"] = cond_block_size
    dict_args["cond_num_layers"] = cond_num_layers

    sr = 48000
    duration = N / sr  # seconds
    n_iters = 100
    timings = []

    model = TCN(**dict_args)  # create the model with args

    rf = model.rf
    samples = N + rf
    input = (torch.rand(1, 1, samples) * 2) - 1

    if dict_args["num_controls"] > 0:
        controls = torch.rand(1, 1, dict_args["num_controls"])
    else:
        controls = None

    if gpu:
        model.cuda()
        input = input.to("cuda:0")
        if controls is not None:
            controls = controls.to("cuda:0")

    model.eval()
    with torch.no_grad():
        for n in range(n_iters):
            tic = time.perf_counter()
            output = model(input, controls)
            toc = time.perf_counter()
            timings.append(toc - tic)
            sys.stdout.write(f"{n+1:3d}/{n_iters:3d}\r")
            sys.stdout.flush()

    mean_time_s = np.mean(timings)
    mean_time_ms = mean_time_s * 1e3
    sec_sec = (1 / duration) * mean_time_s
    rtf = duration / mean_time_s
    rf_ms = (rf / sr) * 1e3
    print(f"Avg. time: {mean_time_ms:0.1f} ms  | sec/sec {sec_sec:0.3f} |  RTF: {rtf:0.2f}x")

    return rf_ms, rtf


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    candidates = []

    # frame_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    frame_sizes = [128, 256, 512]
    causal = [True]
    for c, N in product(causal, frame_sizes):
        model_id = ["TCN-45-S-16"]  # , "TCN-45-L-16", "TCN-250-S-16", "TCN-250-L-16", "TCN-2500-S-16", "TCN-2500-L-16"]
        num_blocks = [5]  # , 10, 6, 11, 5, 10]
        dilation_factors = [4]  # , 2, 4, 2, 10, 3]
        kernels = [7]  # , 3, 11, 7, 13, 5]
        channel_width = [16]  # , 16, 16, 16, 16, 16]
        for mid, b, d, k, ch in zip(model_id, num_blocks, dilation_factors, kernels, channel_width):
            print(b, d, k, ch)
            rf, rtf = run(num_blocks=b, dilation_growth=d, kernel_size=k, channel_width=ch, N=N, gpu=args.gpu)
            if c:
                mid += "-C"
            else:
                mid += "-N"
            candidates.append(
                {"model_id": mid, "causal": c, "kernel": k, "dilation": d, "blocks": b, "channel_width": ch, "rf": rf, "rtf": rtf, "N": N}
            )

    df = pd.DataFrame(candidates)
    print(df)
    if args.gpu:
        df.to_csv("speed_tcn_gpu.csv")
    else:
        df.to_csv("speed_tcn_cpu.csv")

    print("-" * 120)
    print("\tID\t\t\tBlk.\tDil.\tKer.\tN\tRTF\tRF")
    print("-" * 120)
    for n, c in enumerate(candidates):
        print(f"{n: 3d}\t{c['model_id']}\t\t{c['blocks']}\t{c['dilation']}\t{c['kernel']}\t{c['N']}\t{c['rtf']: 2.2f}x\t{c['rf']:0.1f} ms")
    print("-" * 120)
