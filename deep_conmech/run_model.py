# from conmech.helpers.config import SET_ENV
# if name == "__main__":
#     SET_ENV()

# import multiprocessing

import multiprocessing
from dotenv import load_dotenv

load_dotenv()

import argparse
import os
from argparse import ArgumentParser, Namespace
from ctypes import ArgumentError
from pathlib import Path

# Force JAX to initialize once
import jax
jax.devices()

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing

from conmech.helpers import cmh, pca
from conmech.helpers.config import Config, SimulationConfig
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import bunny_fall
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.data.calculator_dataset import CalculatorDataset
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.graph.model_jax import GraphModelDynamicJax, save_tf_model
from deep_conmech.graph.net_jax import CustomGraphNetJax
from deep_conmech.helpers import dch
from deep_conmech.training_config import mtd, TrainingConfig, TrainingData, get_train_config


def setup_distributed(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    # with socketserver.TCPServer(("localhost", 0), None) as s:
    #     free_port = str(s.server_address[1])
    free_port = "12348"
    os.environ["MASTER_PORT"] = free_port
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def get_device_count(config):
    return len(jax.local_devices())


def initialize_data(config: TrainingConfig):
    device_count = get_device_count(config)

    train_dataset = get_train_dataset(
        config.td.dataset, config=config, device_count=device_count
    )
    # if mtd.recreate_training_data:
        # train_dataset.clear_all_data()
    train_dataset.initialize_data(force_recreate=mtd.recreate_training_data)
    all_validation_datasets = get_all_val_datasets(
        config=config, rank=0, world_size=1, device_count=device_count  # 1
    )
    for datasets in all_validation_datasets:
        datasets.initialize_data(force_recreate=False)

    return train_dataset, all_validation_datasets


def train(config: TrainingConfig):
    train_dataset, all_validation_datasets = initialize_data(config=config)

    train_single(
        config,
        train_dataset=train_dataset,
        all_validation_datasets=all_validation_datasets,
    )


def dist_run(
    rank: int,
    world_size: int,
    config: TrainingConfig,
):
    setup_distributed(rank=rank, world_size=world_size)
    train_single(config, rank=rank, world_size=world_size)
    cleanup_distributed()


def train_single(
    config, rank=0, world_size=1, train_dataset=None, all_validation_datasets=None
):
    device_count = get_device_count(config)
    if train_dataset is None:
        train_dataset = get_train_dataset(
            config.td.dataset,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
        )
        train_dataset._load_indices()

    statistics = (
        train_dataset.get_statistics() if config.td.use_dataset_statistics else None
    )
    if config.td.use_dataset_statistics:
        train_dataset.statistics = statistics

    if all_validation_datasets is None:
        all_validation_datasets = get_all_val_datasets(
            config=config, rank=rank, world_size=world_size, device_count=device_count
        )

    all_print_datasets = scenarios.all_print(config.td, config.sc)
    for dataset in all_validation_datasets:
        if config.td.use_dataset_statistics:
            dataset.statistics = statistics

    model = GraphModelDynamicJax(
        train_dataset=train_dataset,
        all_validation_datasets=all_validation_datasets,
        print_scenarios=all_print_datasets,
        config=config,
        statistics=statistics,
    )

    
    state = None
    # if config.load_newest_train:
    # path = '/home/mjureczka/Desktop/conmech3d/output/25.06.03-20.54.25 - JAX GRAPH MODELS/17492983130616 - EPOCH 9 - MODEL'
    # state = model.get_checkpointed_net(path=path)
    # model.epoch = 9
    model.train(state=state)


def visualize(config: TrainingConfig):
    import netron

    checkpoint_path = get_checkpoint_path(config)
    dataset = get_train_dataset(config.td.dataset, config=config)
    dataset.initialize_data()

    model_path = "log/jax_model.tflite"
    state = GraphModelDynamicJax.get_checkpointed_net(path=checkpoint_path)
    save_tf_model(model_path, state, dataset)

    netron.start(model_path)


def plot(config: TrainingConfig):
    if config.td.use_dataset_statistics:
        train_dataset = get_train_dataset(config.td.dataset, config=config)
        statistics = train_dataset.get_statistics()
    else:
        statistics = None
    all_print_scenaros = scenarios.all_print(config.td, config.sc)

    checkpoint_path = get_checkpoint_path(config)
    state = GraphModelDynamicJax.get_checkpointed_net(path=checkpoint_path)
    GraphModelDynamicJax.plot_all_scenarios(state, all_print_scenaros, config)


def run_pca(config: TrainingConfig):
    dataset = get_train_dataset(
        dataset_type=config.td.dataset, config=config, device_count=1
    )
    dataset.initialize_data()
    dataloader = base_dataset.get_train_dataloader(dataset)
    scene = dataset.get_sample_scene()

    pca.run(dataloader, latent_dim=200, scene=scene)

    # simulation_runner.run_examples(
    #     all_scenarios=all_scenarios,
    #     file=__file__,
    #     plot_animation=True,
    #     config=Config(shell=False),
    #     save_all=True,
    # )


def get_train_dataset(
    dataset_type,
    config: TrainingConfig,
    rank: int = 0,
    world_size: int = 1,
    device_count=None,
    item_fn=None,
):
    if device_count is None:
        device_count = get_device_count(config)
    if dataset_type == "synthetic":
        train_dataset = SyntheticDataset(
            description="train",
            load_data_to_ram=config.load_training_data_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize=True,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
            item_fn=item_fn,
        )
    elif dataset_type == "calculator":
        train_dataset = CalculatorDataset(
            description="train",
            all_scenarios=None,
            all_scenarios_fun=scenarios.all_train(config.td, config.sc),
            load_data_to_ram=config.load_training_data_to_ram,
            with_scenes_file=config.with_train_scenes_file,
            randomize=True,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
            item_fn=item_fn,
        )
    else:
        raise ValueError("Wrong dataset type")
    return train_dataset


def get_all_val_datasets(
    config: TrainingConfig, rank: int, world_size: int, device_count: int
):
    all_val_datasets = []
    for all_scenarios in scenarios.all_validation(config.td, config.sc):
        description = "validation_" + str.join(
            "/", [scenario.name for scenario in all_scenarios]
        )
        all_val_datasets.append(
            CalculatorDataset(
                description=description,
                all_scenarios = all_scenarios,
                all_scenarios_fun=None,
                load_data_to_ram=config.load_validation_data_to_ram,
                with_scenes_file=False,
                randomize=False,
                config=config,
                rank=rank,
                world_size=world_size,
                device_count=device_count,
            )
        )
    return all_val_datasets


def get_newest_checkpoint_path_jax(config: TrainingConfig):
    def get_index_jax(path):
        return int(path.split("/")[-2].split(" ")[0])

    all_checkpoint_paths = cmh.find_files_by_name(config.output_catalog, "checkpoint")
    if not all_checkpoint_paths:
        raise ArgumentError("No saved models")
    newest_index = np.argmax(
        np.array([get_index_jax(path) for path in all_checkpoint_paths])
    )

    path = str(Path(all_checkpoint_paths[newest_index]).parent.absolute())
    print(f"============================ Taking saved model {path.split('/')[-1]}")
    return path



def get_checkpoint_path(config: TrainingConfig=None):
    # path = '/output/25.07.30-22.03.40 - JAX GRAPH MODELS/17543001184047 - EPOCH 3 - MODEL'
    path = '/output/25.07.30-22.03.40 - JAX GRAPH MODELS/17544318286450 - EPOCH 4 - MODEL'
    ### Only smaller bunny in training, larger model
    # path = '/output/25.07.28-17.10.33 - JAX GRAPH MODELS/17538806991748 - EPOCH 50 - MODEL'
    ### Bunny and sphere in training, added slide scenario (?)
    # path = '/output/25.05.17-15.40.33 - JAX GRAPH MODELS/17476930222762 - EPOCH 19 - MODEL'
    # path = '/output/25.05.17-15.40.33 - JAX GRAPH MODELS/17475248307028 - EPOCH 1 - MODEL'
    ### Sphere in training, no self collisions
    # path = '/output/25.05.15-08.21.41 - JAX GRAPH MODELS/17474288569620 - EPOCH 15 - MODEL'
    # path = '/output/25.05.15-08.21.41 - JAX GRAPH MODELS/17473225290601 - EPOCH 2 - MODEL'
    # path = '/output/25.05.15-08.21.41 - JAX GRAPH MODELS/17473797451665 - EPOCH 9 - MODEL'
    # path = '/output/25.05.15-08.21.41 - JAX GRAPH MODELS/17473225290601 - EPOCH 2 - MODEL'
    # path = '/output/25.05.15-08.21.41 - JAX GRAPH MODELS/17473143989471 - EPOCH 1 - MODEL'
    ### Denser fine mesh - density proportion 4->2 (final time 6), refactored node features (?)
    # path = '/output/25.05.06-21.25.33 - JAX GRAPH MODELS/17470487201675 - EPOCH 21 - MODEL'
    # path = '/output/25.05.06-21.25.33 - JAX GRAPH MODELS/17466828441649 - EPOCH 5 - MODEL'
    # path = '/output/25.05.06-21.25.33 - JAX GRAPH MODELS/17466312998837 - EPOCH 1 - MODEL'
    ### Longer training examples (final time 7)
    # path = '/output/25.05.01-14.28.00 - JAX GRAPH MODELS/17464255545828 - EPOCH 23 - MODEL'
    # path = '/output/25.05.01-14.28.00 - JAX GRAPH MODELS/17463273115004 - EPOCH 16 - MODEL'
    # path = '/output/25.05.01-14.28.00 - JAX GRAPH MODELS/17462431605234 - EPOCH 10 - MODEL'
    # path = '/output/25.05.01-14.28.00 - JAX GRAPH MODELS/17462011019684 - EPOCH 7 - MODEL'
    # path = '/output/25.05.01-14.28.00 - JAX GRAPH MODELS/17461589058378 - EPOCH 4 - MODEL'
    ### Fixed training data
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17459852938233 - EPOCH 43 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17459008092399 - EPOCH 36 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17457799327246 - EPOCH 26 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17457317066645 - EPOCH 22 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17456954864402 - EPOCH 19 - MODEL'
    # path ='/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17456714615382 - EPOCH 17 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17456473298535 - EPOCH 15 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17456352313850 - EPOCH 14 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17455869178494 - EPOCH 10 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17455627442429 - EPOCH 8 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17455143250332 - EPOCH 4 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17454902142944 - EPOCH 2 - MODEL'
    # path = '/output/25.04.23-18.00.54 - JAX GRAPH MODELS/17454781883695 - EPOCH 1 - MODEL'
    ### New data (larger range)
    # return '/home/michal/Desktop/conmech3d/output/MODELS NEW FEATURES NEW DATA/17442777618789 - EPOCH 16 - MODEL'
    ### Readded input displacement and velocity
    # return '/home/michal/Desktop/conmech3d/output/MODELS NEW FEATURES OLD DATA/17439153689612 - EPOCH 15 - MODEL'
    # return '/home/michal/Desktop/conmech3d/output/MODELS NEW FEATURES OLD DATA/17439394419766 - EPOCH 17 - MODEL'
    # return '/home/michal/Desktop/conmech3d/output/MODELS NEW FEATURES OLD DATA/17439274217021 - EPOCH 16 - MODEL'
    ### Simplified input, norm by current timestep
    # return '/home/michal/Desktop/conmech3d/output/25.03.29-14.23.40 - JAX GRAPH MODELS/17433991184666 - EPOCH 8 - MODEL'
    # path = '/output/25.03.29-14.23.40 - JAX GRAPH MODELS/17434958282739 - EPOCH 16 - MODEL'
    # return '/home/michal/Desktop/conmech3d/output/25.03.29-14.23.40 - JAX GRAPH MODELS/17435684218629 - EPOCH 22 - MODEL'
    ### Longest run
    # return '/home/michal/Desktop/conmech3d/output/25.02.23-15.49.20 - JAX GRAPH MODELS/17405623655142 - EPOCH 16 - MODEL'
    absolute_path = '/home/michal/Desktop/conmech3d' + path
    return absolute_path

def main(args: Namespace):
    cmh.print_jax_configuration()
    print(f"MODE: {args.mode}, PID: {os.getpid()}")
    # dch.cuda_launch_blocking()
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)

    config = get_train_config(shell=args.shell, mode="normal")

    # dch.set_torch_sharing_strategy()
    dch.set_memory_limit(config=config)
    print(f"Running using {config.device}")

    if args.mode == "train":
        train(config)
    if args.mode == "profile":
        config.max_epoch_number = 2
        train(config)
    if args.mode == "plot":
        plot(config)
    if args.mode == "visualize":
        visualize(config)
    if args.mode == "pca":
        run_pca(config)


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_start_method("spawn")  # forkserver")
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "plot", "profile", "visualize", "pca"],
        default="plot",
        help="Running mode of aplication",
    )
    parser.add_argument(
        "--shell", action=argparse.BooleanOptionalAction, default=False
    )  # Python 3.9+
    args = parser.parse_args()
    # with jax.disable_jit():
    main(args)
