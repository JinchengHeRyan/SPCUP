import os
import json
import time
import GPUtil
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
from utils.utils import get_instance
from utils.average import AverageVal
import librosa
import pysptk

import dataset
import dataloader
import model as module_model

output_path = "answer.txt"

output_file = open(output_path, "w")


def main(config, args):
    model = get_instance(module_model, config["model"])
    chkpt = torch.load(args.chkpt_path)
    model.load_state_dict(chkpt["model"])
    model = model.cuda()
    model.eval()

    testset = get_instance(dataset, config["testset"])
    # testloader = get_instance(dataloader, config["testloader"], testset)
    testloader = dataloader.SimpleDataLoader(
        testset, shuffle=False, batch_size=1, num_workers=1, drop_last=False
    )

    os.makedirs(args.output, exist_ok=True)
    for i, (wav_utt, fbank, wav_spec, wav_phase, wav, target_method) in enumerate(
        tqdm(testloader, ncols=80)
    ):
        fbank = (
            fbank.view(fbank.shape[0], 1, fbank.shape[1], fbank.shape[2]).float().cuda()
        )

        with torch.no_grad():
            pred = model(fbank)
            pred = torch.argmax(pred, dim=1)
            pred = int(pred.to("cpu"))
        output_file.write("{}, {}\n".format(wav_utt[0], pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Speech Separation")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-p",
        "--chkpt-path",
        type=str,
        required=True,
        help="path to the chosen checkpoint",
    )
    parser.add_argument("output")
    args = parser.parse_args()

    for f in args.config, args.chkpt_path:
        assert os.path.isfile(f), "No such file: %s" % f

    # Read config of the whole system.
    with open(args.config) as rfile:
        config = json.load(rfile)

    # deviceIDs = GPUtil.getAvailable(limit=1, maxMemory=0.8, maxLoad=0.8)
    # assert deviceIDs != [], "No GPUs available!"
    # print("Use GPU:", deviceIDs)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, deviceIDs))
    # os.environ["OMP_NUM_THREADS"] = "1"

    main(config, args)
