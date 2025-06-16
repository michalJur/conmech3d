import copy
import json
import os
from ctypes import ArgumentError
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

from conmech.helpers import cmh, pkh
from conmech.helpers.config import Config
from conmech.helpers.tmh import Timer
from conmech.plotting import plotter_functions
from conmech.scenarios.scenarios import Scenario
from conmech.scene.energy_functions import EnergyFunctions
from conmech.scene.scene import Scene
from conmech.scene.scene_temperature import SceneTemperature
from conmech.solvers.calculator import Calculator


def get_solve_function(simulation_config):
    if simulation_config.mode in ["normal", 'normal_with_reduced', 'pca']:
        return Calculator.solve
    # if simulation_config.mode == "pca":
        # return Calculator.solve_compare_reduced
        # return Calculator.solve_skinning_backwards
    # if simulation_config.mode == "compare_reduced":
        # return Calculator.solve_compare_reduced
    if simulation_config.mode == "skinning":
        return Calculator.solve_skinning
    if simulation_config.mode == "skinning_backwards":
        return Calculator.solve_skinning_backwards
    if simulation_config.mode == "temperature":
        return Calculator.solve_with_temperature
    if "net" in simulation_config.mode:
        from deep_conmech.graph import model_jax
        from deep_conmech.graph.model_jax import GraphModelDynamicJax
        from deep_conmech.run_model import get_checkpoint_path, get_train_dataset
        from deep_conmech.training_config import TrainingConfig

        training_config = TrainingConfig(shell=False)
        training_config.sc = simulation_config
        checkpoint_path = get_checkpoint_path(training_config)
        state = GraphModelDynamicJax.get_checkpointed_net(path=checkpoint_path)

        if training_config.td.use_dataset_statistics:
            train_dataset = get_train_dataset(
                training_config.td.dataset, config=training_config
            )
            train_dataset._load_indices()

        if "compare" in simulation_config.mode:
            return partial(
                model_jax.solve_compare, apply_net=model_jax.get_apply_net(state)
            )
        return partial(model_jax.solve, apply_net=model_jax.get_apply_net(state))

    raise ArgumentError


def create_scene(scenario):
    print("Creating scene...")
    create_in_subprocess = False

    def get_scene():
        if scenario.simulation_config.mode in ["normal"]:
            scene = Scene(
                mesh_prop=scenario.mesh_prop,
                body_prop=scenario.body_prop,
                obstacle_prop=scenario.obstacle_prop,
                schedule=scenario.schedule,
                create_in_subprocess=create_in_subprocess,
                simulation_config=scenario.simulation_config,
            )
        elif scenario.simulation_config.mode in [
            "skinning",
            "skinning_backwards",
            "pca",
        ]:
            from deep_conmech.scene.scene_layers import SceneLayers

            scene = SceneLayers(
                mesh_prop=scenario.mesh_prop,
                body_prop=scenario.body_prop,
                obstacle_prop=scenario.obstacle_prop,
                schedule=scenario.schedule,
                create_in_subprocess=create_in_subprocess,
                simulation_config=scenario.simulation_config,
            )
        elif scenario.simulation_config.mode in [
            "net",
            "compare_net",
            "normal_with_reduced",
        ]:
            from deep_conmech.scene.scene_input import SceneInput

            randomize = False
            scene = SceneInput(
                mesh_prop=scenario.mesh_prop,
                body_prop=scenario.body_prop,
                obstacle_prop=scenario.obstacle_prop,
                schedule=scenario.schedule,
                simulation_config=scenario.simulation_config,
                create_in_subprocess=create_in_subprocess,
            )
            if randomize:
                scene.set_randomization(scenario.simulation_config)
            else:
                scene.unset_randomization()
        elif scenario.simulation_config.mode == "temperature":
            scene = SceneTemperature(
                mesh_prop=scenario.mesh_prop,
                body_prop=scenario.body_prop,
                obstacle_prop=scenario.obstacle_prop,
                schedule=scenario.schedule,
                create_in_subprocess=create_in_subprocess,
                simulation_config=scenario.simulation_config,
            )
        else:
            raise ArgumentError

        scene.normalize_and_set_obstacles(
            scenario.linear_obstacles, scenario.mesh_obstacles
        )
        return scene

    scene = cmh.profile(
        get_scene,
        baypass=True,
    )
    # np.save("./pt-jax/bunny_boundary_nodes2.npy", scene.boundary_nodes)
    # np.save("./pt-jax/contact_boundary2.npy", scene.boundaries.contact_boundary)
    return scene


def run_examples(
    all_scenarios,
    file,
    plot_animation,
    config: Config,
    simulate_dirty_data=False,
    save_all=False,
    additional_args = {},
):
    scenes = []
    final_catalogs = []
    for i, scenario in enumerate(all_scenarios):
        print(f"-----EXAMPLE {i + 1}/{len(all_scenarios)}-----")
        
        catalog = os.path.splitext(os.path.basename(file))[0].upper()
        # if save_files:
        if 'main_dir' in additional_args:
            main_dir = additional_args['main_dir']
        else:
            main_dir = f"{config.output_catalog}/{config.current_time} - {catalog}"

        final_catalog = f"{main_dir}/{scenario.name}/{scene.simulation_config.mode}"
        # _{scene.mesh_prop.mesh_type}

        scene, final_catalog = run_scenario(
            solve_function=get_solve_function(scenario.simulation_config),
            scenario=scenario,
            config=config,
            run_config=RunScenarioConfig(
                final_catalog=final_catalog,
                save_all=save_all,
                simulate_dirty_data=simulate_dirty_data,
                plot_animation=plot_animation,
            ),
            scene=create_scene(scenario),
            additional_args=additional_args
        )
        scenes.append(scene)
        final_catalogs.append(final_catalog)
        print()
    print("DONE")
    return scenes, final_catalogs

def run_example(
    scenario,
    final_catalog,
    plot_animation,
    config: Config,
    simulate_dirty_data=False,
    save_all=False,
    additional_args = {},
):
    run_scenario(
        solve_function=get_solve_function(scenario.simulation_config),
        scenario=scenario,
        config=config,
        run_config=RunScenarioConfig(
            final_catalog=final_catalog,
            save_all=save_all,
            simulate_dirty_data=simulate_dirty_data,
            plot_animation=plot_animation,
        ),
        scene=create_scene(scenario),
        additional_args=additional_args
    )
    print("DONE")
    return

@dataclass
class RunScenarioConfig:
    final_catalog: Optional[str] = None
    simulate_dirty_data: bool = False
    plot_animation: bool = False
    save_all: bool = False


def save_scene(scene: Scene, final_catalog: str, save_animation: bool):
    # Blender
    if False:
        blender_data_path = f"{final_catalog}/blender/data.blender"
        
        blender_data = (scene.boundary_nodes, scene.boundaries.boundary_surfaces)
        if isinstance(scene, SceneTemperature):
            blender_data += (scene.t_old,)
        else:
            blender_data += (None,)

        for obs in scene.mesh_obstacles:
            # TODO: Mesh obstacles and temperature - create dataclass
            blender_data += (obs.boundary_nodes, obs.boundaries.boundary_surfaces)

        pkh.append_data(data=blender_data, data_path=blender_data_path, lock=None)

    # Comparer
    if hasattr(scene, "reduced"):
        comparer_data_path = f"{final_catalog}/scenes/data.scene"

        comparer_data = {
            "exact_acceleration": scene.exact_acceleration,
            # "lifted_acceleration": scene.lifted_acceleration,
            # "lifted_acceleration": scene.lifted_acceleration,
            "displacement_old": scene.displacement_old,
            # "norm_new_displacement": scene.to_normalized_displacement(
            #     scene.lifted_acceleration
            # ),
            # "norm_reduced": scene.new_displacement_norm_by_reduced_new_displacement,
            # "normalized_by_reduces_nodes": scene.normalized_initial_nodes
            # + scene.new_displacement_norm_by_reduced_new_displacement,
            # # "norm_lifted_new_displacement": scene.norm_lifted_new_displacement,
            # # "normalized_nodes": scene.normalized_initial_nodes
            # # + scene.norm_lifted_new_displacement,
            # "recentered_norm_lifted_new_displacement": scene.recentered_norm_lifted_new_displacement,
            "reduced_exact_acceleration": scene.reduced.exact_acceleration
        }

        pkh.append_data(data=comparer_data, data_path=comparer_data_path, lock=None)

    # Matplotlib
    if save_animation:
        # scenes_file, indices_file = pkh.open_files_append(scenes_path)
        # with scenes_file, indices_file:
        scene_copy = copy.copy(scene)
        scene_copy.prepare_to_save()

        pkh.append_data(data=scene_copy, data_path=f"{final_catalog}/matplotlib/data.matplotlib", lock=None)


def run_scenario(
    solve_function: Callable,
    scenario: Scenario,
    config: Config,
    run_config: RunScenarioConfig,
    scene: Scene,
    additional_args = {}
) -> Tuple[Scene, str, float]:
    time_skip = config.print_skip
    ts = int(time_skip / scenario.time_step)
    plot_scenes_count = [0]
    # with_reduced = hasattr(scene, "reduced")
    save_files = run_config.plot_animation or run_config.save_all
    save_animation = run_config.plot_animation

    step = [0]  # TODO: #65 Clean

    def operation_save(scene: Scene):
        if config.animation_backend is None:
            return
        plot_index = step[0] % ts == 0
        if "three" in config.animation_backend:
            plotter_functions.save_three(
                scene=scene, step=step[0], folder=f"{run_config.final_catalog}/three", skip=10
            )
        if run_config.save_all or plot_index:
            # scenes_path_reduced = f"{final_catalog}/scenarios_reduced/{label}.scenes"

            save_scene(
                scene=scene, final_catalog=run_config.final_catalog, save_animation=save_animation
            )
            # if with_reduced:
            #     cmh.create_folders(f"{final_catalog}/scenarios_reduced")
            #     save_scene(
            #         scene=scene.reduced,
            #         scenes_path=scenes_path_reduced,
            #         save_animation=save_animation,
            #     )
        if plot_index:
            plot_scenes_count[0] += 1
        step[0] += 1

    def fun_sim():
        return simulate(
            scene=scene,
            solve_function=solve_function,
            scenario=scenario,
            simulate_dirty_data=run_config.simulate_dirty_data,
            config=config,
            operation=operation_save if save_files else None,
            additional_args=additional_args
        )

    # cmh.profile(fun_sim)
    scene = fun_sim()

    if run_config.plot_animation and config.animation_backend is not None:
        if "blender" in config.animation_backend:
            plotter_functions.plot_using_blender(output=config.blender_output)
        if "matplotlib" in config.animation_backend:
            plotter_functions.plot_scenario_animation(
                scenario=scenario,
                config=config,
                animation_path=f"{run_config.final_catalog}/{scenario.name}.gif",
                time_skip=time_skip,
                index_skip=ts if run_config.save_all else 1,
                plot_scenes_count=plot_scenes_count[0],
                all_scenes_path=run_config.final_catalog, ###
            )

    return scene


def prepare(scenario, scene: Scene, current_time, with_temperature):
    forces = scenario.get_forces_by_function(scene, current_time)
    if with_temperature:
        heat = scenario.get_heat_by_function(scene, current_time)
        scene.prepare_tmp(forces, heat)
    else:
        scene.prepare(forces)


def print_mesh_data(scene):
    print(f"Mesh type: {scene.mesh_prop.mesh_type}")
    print(
        f" Elements: {scene.elements_count}\
 | Boundary surfaces: {scene.boundary_surfaces_count} | Nodes: {scene.nodes_count}"
    )
    print()


def prepare_energy_functions(
    scenario, scene, solve_function, with_temperature, precompile
):
    pca=(scene.simulation_config.mode == "pca")
    energy_functions = EnergyFunctions(simulation_config=scene.simulation_config, pca=pca)
    reduced_energy_functions = EnergyFunctions(
        simulation_config=scene.simulation_config
    )

    if not precompile:
        return [energy_functions, reduced_energy_functions]

    print("Precompiling...")  # TODO: Copy can be expensive
    with cmh.HiddenPrints():
        prepare(scenario, scene, 0, with_temperature)
        scene_copy = copy.deepcopy(scene)  # copy to not change initial vectors
        for mode in EnergyFunctions.get_manual_modes():
            energy_functions.set_manual_mode(mode)
            try:
                _ = solve_function(
                    scene=scene_copy,
                    energy_functions=energy_functions,
                    initial_a=None,
                    initial_t=None,
                )
            except AssertionError:
                pass

        energy_functions.set_automatic_mode()
    return energy_functions


def simulate(
    scene,
    solve_function,
    scenario: Scenario,
    simulate_dirty_data: bool,
    config: Config,
    operation: Optional[Callable] = None,
    additional_args={}
) -> Tuple[Scene, float]:
    with_temperature = isinstance(scene, SceneTemperature)

    print_mesh_data(scene)
    energy_functions = prepare_energy_functions(
        scenario, scene, solve_function, with_temperature, precompile=False  # True
    )

    acceleration, temperature = (None,) * 2
    time_tqdm = scenario.get_tqdm(desc="Simulating", config=config)
    steps = len(time_tqdm)
    timer = additional_args['timer'] if 'timer' in additional_args else Timer()

    if 'reduced_exact_accelerations' in additional_args:
        assert len(time_tqdm) == len(additional_args['reduced_exact_accelerations'])
    scene.reduced_exact_accelerations = []

    for step, time_step in enumerate(time_tqdm):
        current_time = (time_step) * scene.time_step

        with timer["all_prepare"]:
            prepare(scenario, scene, current_time, with_temperature)


        with timer["reduced_solver"]:
            if "reduced_exact_accelerations" in additional_args:
                # print('Taking reduced acceleration')
                scene.reduced.exact_acceleration = additional_args['reduced_exact_accelerations'][step]
            else:
                # print('Calculating reduced acceleration')
                scene.reduced.exact_acceleration, _ = Calculator.solve(
                    scene=scene.reduced,
                    energy_functions=energy_functions[1],
                    initial_a=scene.reduced.exact_acceleration,
                    # timer=timer,
                )

        with timer["solver"]:
            scene.exact_acceleration, temperature = solve_function(
                scene=scene,
                energy_functions=energy_functions,
                initial_a=acceleration,
                initial_t=temperature,
                # timer=timer,
            )
        scene.reduced_exact_accelerations.append(scene.reduced.exact_acceleration)

        scene.reorient_and_set_lifted()

        # scene.norm_lifted_new_displacement = (
        #     scene.get_norm_by_reduced_lifted_new_displacement(scene.lifted_acceleration)
        # )
        # scene.recentered_norm_lifted_new_displacement = scene.recenter_by_current_reduced(
        #     new_displacement=scene.norm_lifted_new_displacement
        # )

        if simulate_dirty_data:
            scene.make_dirty()

        with timer["all_operation"]:
            if operation is not None:
                operation(scene=scene)  # (current_time, scene, a, base_a)

        with timer["all_iterate"]:
            scene.iterate_self(scene.exact_acceleration, temperature=temperature)

    for key in timer:
        all_time = timer.dt[key].sum()
        print(f" {key}: {all_time:.2f}s | {(steps/all_time):.2f}it/s")

    # print("Saving timings")
    # fig = timer.dt["all_solver"].plot.hist(bins=100).get_figure()
    # fig.savefig(f"log/timing-{cmh.get_timestamp(config)}.png")

    return scene
