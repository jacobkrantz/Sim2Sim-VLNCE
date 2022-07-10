#!/usr/bin/env python3

import argparse
import os
import random

import MatterSim  # noqa: F401
import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import sim2sim_vlnce  # noqa: F401
from sim2sim_vlnce.config.default import get_config


def set_seed(seed: int, is_eval: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = is_eval
    if torch.cuda.is_available():
        torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["collect", "train", "eval", "inference"],
        default="eval",
        help="run type of the experiment",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    set_seed(config.TASK_CONFIG.SEED, is_eval=run_type == "eval")

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "collect":
        trainer.collect_sgm_imitation()
    elif run_type == "train":
        trainer.train(config)
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()
