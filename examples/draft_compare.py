from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from conmech.helpers import cmh
from conmech.helpers.config import Config
from conmech.scenarios import scenarios
from conmech.simulations import simulation_runner
from deep_conmech.graph.model_jax import RMSE
from deep_conmech.run_model import get_checkpoint_path
from deep_conmech.training_config import get_train_config

# TODO:
# test with uninitialized network
# check difference between exact and lifted in plotted data

# TODO:
# test with uninitialized network
# check difference between exact and lifted in plotted data


def main():
    load_dotenv()
      
    base_mode = "compare_reduced" # "skinning_backwards",
    other_modes = [
        "skinning",
        "net",
        "pca",
    ]

    config = get_train_config(shell=False, mode=None)
    remove_old_simulation(config=config)

    skip_base=False
    run_all_simulations(base_mode=base_mode, other_modes=other_modes, config=config, skip_base=skip_base)
    compare_latest(base_mode=base_mode, other_modes=other_modes, config=config)

    input("Press Enter to continue...")



def _get_scene_files(config):
    checkpoint_path = get_checkpoint_path()
    input_path = "output"
    all_scene_files = cmh.find_files_by_extension(input_path, "scenes_comparer")
    label = checkpoint_path.split("/")[-1]
    main_path = f"output/{config.current_time}_COMPARE_{label}"

    path_id = "/scenarios/"
    scene_files = [f for f in all_scene_files if path_id in f]
    return main_path, scene_files
    # all_arrays_path = max(scene_files, key=os.path.getctime)



def remove_old_simulation(config):
    _, scene_files = _get_scene_files(config)
    for scene_file in scene_files:
        directory = Path(scene_file).parent.parent
        cmh.clear_folder(directory)


def run_examples(config, mode, additional_args = {}):
    # all_print_scenaros = scenarios.all_print(config.td, config.sc)
    # all_scenarios = scenarios.all_validation(config.td, config.sc)
    # GraphModelDynamicJax.plot_all_scenarios(state, all_print_scenaros, training_config)
    config.sc.mode = mode
    all_scenarios = scenarios.all_compare(config.td, config.sc)
    scenes = simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=False,
        config=Config(shell=False),
        save_all=True,
        additional_args=additional_args
    )
    return scenes

def run_all_simulations(base_mode, other_modes, config, skip_base=False):
    cmh.print_jax_configuration()

    print("MODE: ", base_mode)
    if not skip_base:
        run_examples(config=config, mode=base_mode)
    reduced_exact_accelerations = get_reduced_exact_acceleration(config=config, base_mode=base_mode)
    
    for mode in other_modes:
        print("MODE: ", mode)
        run_examples(config=config, mode=mode, additional_args={'reduced_exact_accelerations': reduced_exact_accelerations})

def get_error(simulation_1, simulation_2, index, key):
    return RMSE(simulation_1[index][key], simulation_2[index][key])

def get_reduced_exact_acceleration(config, base_mode):
    main_path, scene_files = _get_scene_files(config)
    base = cmh.get_simulation(scene_files, base_mode)
    return [s['reduced_exact_acceleration'] for s in base]




def compare_latest(base_mode, other_modes, config):
    main_path, scene_files = _get_scene_files(config)
    base = cmh.get_simulation(scene_files, base_mode)

    cmh.create_folder(main_path)
    for key in [
        # "norm_lifted_new_displacement",
        "recentered_norm_lifted_new_displacement",
        "normalized_nodes",
        # "new_displacement",
        # "displacement_old",
        "exact_acceleration",
        "normalized_nodes",
        # "lifted_acceleration",
    ]:
        errors_df = pd.DataFrame()
        for mode in other_modes:
            pretendent = cmh.get_simulation(scene_files, mode)

            simulation_len = min(len(base), len(pretendent))

            errors = []
            for index in tqdm(range(simulation_len)):
                errors.append(get_error(base, pretendent, index=index, key=key))
            errors_df[mode] = np.array(errors)
            print(f"Error {mode} {key}: ", np.mean(errors))
            print()

        plot = errors_df.plot()
        fig = plot.get_figure()
        fig.savefig(f"{main_path}/{key}.png")


if __name__ == "__main__":
    main()
