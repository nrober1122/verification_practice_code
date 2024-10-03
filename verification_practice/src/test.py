"""Runs a closed-loop reachability experiment according to a param file."""
import argparse  # noqa: E402
import ast  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import tracemalloc

import torch

import dynamics
import cl_systems
from utils.robust_training_utils import Analyzer, ReachableSet

from utils.nn import *

dir_path = os.path.dirname(os.path.realpath(__file__))


def main_forward_nick(params: dict) -> Tuple[Dict, Dict]:
    device = 'cpu'
    torch.no_grad()

    def di_condition(input_range):
            delta = 0.0
            return input_range[1, 0] >= -1 and input_range[0, 0] >= 0 + delta

    if params["system"]["type"] == 'DoubleIntegrator':
        controller = load_controller(params['system']['type'], params['system']['controller'], params["system"]["dagger"], device=device)
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.ClosedLoopDynamics(controller, ol_dyn, device=device)
        
        init_range = np.array([
            [2.5, 3.],
            [-0.25, 0.25]
        ])

        time_horizon = 30

        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=30, device=device)

        tstart = time.time()
        if params["analysis"]["mode"] == 'symbolic':
            reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=di_condition)
        elif params["analysis"]["mode"] == 'concrete':
            # we do some partitioning to keep things reasonable
            analyzer.set_partition_strategy(0, np.array([4,4]))
            reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=False, visualize=False, condition=di_condition)
        elif params["analysis"]["mode"] == 'CARV':
            reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=False, visualize=False, condition=di_condition)
        else:
            return NotImplementedError
        tend = time.time()
        print('Calculation Time: {}'.format(tend-tstart))

        for i in [-1]:
            analyzer.another_plotter(info, [i])  


def setup_parser() -> dict:
    """Load yaml config file with experiment params."""
    parser = argparse.ArgumentParser(
        description="Analyze a closed loop system w/ NN controller."
    )

    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Absolute or relative path to yaml file describing experiment"
            " configuration. Note: if this arg starts with 'example_configs/',"
            " the configs in the installed package will be used (ignoring the"
            " pwd)."
        ),
    )

    args = parser.parse_args()

    if args.config.startswith("configs/"):
        # Use the config files in the pip-installed package
        param_filename = f"{dir_path}/_static/{args.config}"
    else:
        # Use the absolute/relative path provided in args.config
        param_filename = f"{args.config}"

    with open(param_filename, mode="r", encoding="utf-8") as file:
        params = yaml.load(file, yaml.Loader)

    return params


if __name__ == "__main__":
    experiment_params = setup_parser()
    main_forward_nick(experiment_params)
