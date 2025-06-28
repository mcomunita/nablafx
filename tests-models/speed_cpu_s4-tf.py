import gc
import sys
import time
import torch
import numpy as np
import lightning as pl
import pandas as pd
from itertools import product
from argparse import ArgumentParser

from nablafx.s4 import S4


def run(
    num_inputs=1,
    num_outputs=1,
    num_controls=0,
    num_blocks=4,
    s4_state_dim=4,
    channel_width=16,
    batchnorm=False,
    residual=True,
    direct_path=False,
    cond_type=None,
    cond_block_size=128,
    cond_num_layers=1,
    act_type="tanh",
    s4_learning_rate=0.01,
    N=48000,
):
    dict_args = {}
    dict_args["num_inputs"] = num_inputs
    dict_args["num_outputs"] = num_outputs
    dict_args["num_controls"] = num_controls
    dict_args["num_blocks"] = num_blocks
    dict_args["s4_state_dim"] = s4_state_dim
    dict_args["channel_width"] = channel_width
    dict_args["batchnorm"] = batchnorm
    dict_args["residual"] = residual
    dict_args["direct_path"] = direct_path
    dict_args["cond_type"] = cond_type
    dict_args["cond_block_size"] = cond_block_size
    dict_args["cond_num_layers"] = cond_num_layers
    dict_args["act_type"] = act_type
    dict_args["s4_learning_rate"] = s4_learning_rate

    sr = 48000
    duration = N / sr  # seconds
    n_iters = 500
    timings = []

    model = S4(**dict_args)  # create the model with args

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

    model_id = ["S4-TF-S-16", "S4-TF-L-16"]
    num_controls = [0, 0]
    num_blocks = [4, 8]
    s4_state_dim = [4, 32]
    channel_width = [16, 16]
    cond_type = ["tfilm", "tfilm"]

    for mid, nc, b, s, ch, ct in zip(model_id, num_controls, num_blocks, s4_state_dim, channel_width, cond_type):

        for N in frame_sizes:
            print()
            print(f"{mid}: {nc}, {b}, {s}, {ch}, {ct}, frame size: {N}")
            rtf = run(
                num_controls=nc,
                num_blocks=b,
                s4_state_dim=s,
                channel_width=ch,
                cond_type=ct,
                N=N,
            )
            # if c:   mid += "-C"
            # else:   mid += "-N"
            candidates.append(
                {
                    "model_id": mid,
                    "num_controls": nc,
                    "num_blocks": b,
                    "s4_state_dim": s,
                    "channel_width": ch,
                    "cond_type": ct,
                    "rtf": rtf,
                    "N": N,
                }
            )

    df = pd.DataFrame(candidates)
    df.to_csv("tests/speed_cpu_s4-tf.csv")

    print("-" * 120)
    print("\tID\t\t\tBlk.\tSt.\tCh.\tN\tRTF")
    print("-" * 120)
    for n, c in enumerate(candidates):
        print(f"{n: 3d}\t{c['model_id']}\t\t{c['num_blocks']}\t{c['s4_state_dim']}\t{c['channel_width']}\t{c['N']}\t{c['rtf']: 2.2f}x")
    print("-" * 120)
