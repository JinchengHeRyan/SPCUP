import torch
import torch.nn as nn
import numpy as np
import logging
import os
import sys
import argparse
import time
import json
import GPUtil
import random

from utils.utils import get_instance
import trainer as trainer_
import model as module_model

import dataset
import dataloader

# fix random seeds for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config, resume_path, name, print_freq):
    trainset = get_instance(dataset, config["trainset"])
    trainloader = get_instance(dataloader, config["trainloader"], trainset)

    # testset = get_instance(dataset, config["testset"])
    # testloader = get_instance(dataloader, config["testloader"], testset)

    model = get_instance(module_model, config["model"])
    model = nn.DataParallel(model).cuda()

    chkpt_dir = os.path.join("chkpt", name)
    os.makedirs(chkpt_dir, exist_ok=True)

    # make a copy of config in chkpt_dir
    with open(os.path.join(chkpt_dir, "config.json"), "w") as wfile:
        json.dump(config, wfile, indent=4, sort_keys=False)

    model_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = get_instance(torch.optim, config["optimizer"], model_params)
    lr_scheduler = get_instance(
        torch.optim.lr_scheduler, config["lr_scheduler"], optimizer
    )
    criterion = get_instance(torch.nn, config["loss"]).cuda()

    # set logger for printing information.
    log_dir = os.path.join("logs", name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, time.strftime("%Y-%m-%d-%H%M.log", time.localtime(time.time()))
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    logger.info(model)
    logger.info("-" * 50)

    trainer = get_instance(
        trainer_,
        config["Trainer"],
        chkpt_dir=chkpt_dir,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        logger=logger,
        epochs=config["epochs"],
        trainloader=trainloader,
        # testloader=testloader,
        resume_path=resume_path,
        print_freq=print_freq,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Speech Separation: VoiceFilter")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        required=True,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        type=str,
        required=True,
        help="name of the saved model",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="The frequency of printing information.",
    )
    args = parser.parse_args()

    # Read config of the whole system.
    assert os.path.isfile(args.config), "No such file: %s" % args.config
    with open(args.config) as rfile:
        config = json.load(rfile)

    main(config, args.resume, args.name, args.print_freq)
