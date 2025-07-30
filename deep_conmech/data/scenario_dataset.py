from ctypes import ArgumentError
import gc
import tempfile
import os
import pickle
from typing import Callable, List, Optional

import numpy as np

from conmech.helpers import cmh, interpolation_helpers, mph
from conmech.plotting.plotter_functions import save_three
from conmech.scenarios.scenarios import Scenario
from conmech.scene.energy_functions import EnergyFunctions
from conmech.scene.scene import Scene
from conmech.solvers.calculator import Calculator
from deep_conmech.data.base_dataset import BaseDataset
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingConfig



class ScenariosDataset(BaseDataset):
    def __init__(
        self,
        description: str,
        all_scenarios,
        all_scenarios_fun: List[Scenario],
        solve_function: Callable,
        load_data_to_ram: bool,
        with_scenes_file: bool,
        randomize: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
        device_count: int,
        item_fn,
    ):
        self.all_scenarios = all_scenarios
        self.all_scenarios_fun = all_scenarios_fun
        self.data_count = None

        super().__init__(
            description=description,
            dimension=3, ###
            data_count=None,
            solve_function=solve_function,
            load_data_to_ram=load_data_to_ram,
            randomize=randomize,
            num_workers=config.scenario_generation_workers,
            with_scenes_file=with_scenes_file,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
            item_fn=item_fn,
        )
        self.reset(epoch=1)

    def reset(self, epoch):
        self.epoch = epoch
        if self.all_scenarios:
            test_scenarios = self.all_scenarios
            self.scenarios_count = len(test_scenarios)
        else:
            self.generate_scenario, self.scenarios_count = self.all_scenarios_fun
            scenario = self.generate_scenario()
            test_scenarios = [scenario for _ in range(self.scenarios_count)]
        self.data_count=self.get_data_count(test_scenarios)



    def get_data_count(self, scenarios):
        return np.sum([int(s.schedule.episode_steps) for s in scenarios])

    def get_assigned_scenarios(self, num_workers, process_id):
        scenarios_count = len(self.all_scenarios)
        if scenarios_count % num_workers != 0:
            raise Exception("Cannot divide data generation work")
        assigned_scenarios_count = int(scenarios_count / num_workers)
        assigned_scenarios = self.all_scenarios[
            process_id
            * assigned_scenarios_count : (process_id + 1)
            * assigned_scenarios_count
        ]
        return assigned_scenarios

    def get_sample_scene(self):
        assigned_scenarios = self.get_assigned_scenarios(num_workers=1, process_id=0)
        scenario = assigned_scenarios[0]
        print("Taking sample scenario, assuming all scenes are the same")
        scene = self.get_scene(scenario=scenario, config=self.config)
        return scene

    def get_scene(self, scenario: Scenario, config: TrainingConfig) -> Scene:
        scene = SceneInput(
            mesh_prop=scenario.mesh_prop,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            simulation_config=scenario.simulation_config,
            create_in_subprocess=False,
        )
        scene.set_randomization(config)

        scene.normalize_and_set_obstacles(
            scenario.linear_obstacles, scenario.mesh_obstacles
        )
        return scene

    def print_stats(self, scene):
        print(len(scene.initial_nodes))
        print(len(scene.reduced.initial_nodes))
        print(np.min(scene.initial_nodes, axis=0))
        print(np.max(scene.initial_nodes, axis=0))

    def generate_data(self):
        if self.config.generate_data_in_subprocesses:
            # mph.run_process(self.generate_data_process)
            done = mph.run_processes(
                self.generate_data_process, num_workers=self.num_workers
            )
            if not done:
                print("NOT DONE")
        else:
            self.generate_data_process()

    def generate_data_process(self, num_workers: int = 1, process_id: int = 0):
        # tqdm_description = f"Generating data - process {process_id+1}/{num_workers}"
        # simulation_data_count = np.sum(
        #     [s.schedule.episode_steps for s in assigned_scenarios]
        # )
        # start_index = process_id * simulation_data_count
        # current_index = start_index
        # step_tqdm = cmh.get_tqdm(
        #     range(simulation_data_count),
        #     config=self.config,
        #     desc=tqdm_description,
        #     position=process_id,
        # )
        # scenario_id = 0
        # scenario = assigned_scenarios[scenario_id]

        all_steps = 0
        if self.all_scenarios:
            assigned_scenarios = self.get_assigned_scenarios(num_workers, process_id)
        scenario_id = 0
        while scenario_id < self.scenarios_count:
            correct = True
            if self.all_scenarios:
                scenario = assigned_scenarios[scenario_id]
            else:
                scenario = self.generate_scenario()

            scene = self.get_scene(scenario=scenario, config=self.config)
            energy_functions = EnergyFunctions(
                    simulation_config=scene.simulation_config
            )
            reduced_energy_functions = EnergyFunctions(
                simulation_config=scene.simulation_config
            )

            message = f"Simulating {scenario_id+1}/{self.scenarios_count}"
            time_tqdm = scenario.get_tqdm(desc=message, config=self.config)
            cmh.save_to_log(message)
            gc.collect()

            tmp_filename = "output/graph_data_tmp"
            open(tmp_filename, "wb").close()

            for episode_step in time_tqdm:
                current_time = episode_step * scene.time_step

                forces = scenario.get_forces_by_function(scene, current_time)
                scene.prepare(forces)

                reduced_exact_acceleration, _ = Calculator.solve(
                    scene=scene.reduced,
                    initial_a=scene.reduced.exact_acceleration,
                    energy_functions=reduced_energy_functions,
                )
                exact_acceleration, _ = Calculator.solve(
                    scene=scene, energy_functions=energy_functions, initial_a=scene.exact_acceleration
                )
                if reduced_exact_acceleration is None or exact_acceleration is None:
                    cmh.save_to_log(
                        f"Scene {scenario.name} - episode step {episode_step} - NaN in acceleration, skipping",
                        fail=2,
                    )
                    correct = False
                    break
                scene.reduced.exact_acceleration = reduced_exact_acceleration
                scene.exact_acceleration = exact_acceleration

                scene.reorient_and_set_lifted()

                if self.with_scenes_file:
                    self.safe_save_scene(scene=scene, data_path=self.scenes_data_path)
                else:
                    graph_data = self.get_features_and_target(scene=scene, scenario_name=scenario.name, episode_step=episode_step)
                    with open(tmp_filename, "ab") as tmp_file:
                        pickle.dump(graph_data, tmp_file)
                all_steps += 1

                final_catalog = (
                    f"{self.config.output_catalog}/{self.config.current_time} - DATASET"
                )
                label = cmh.get_run_label(self.config, scenario, self.epoch)
                save_three(
                    scene=scene,
                    step=episode_step, #index,
                    # label=label, #f"{self.config.current_time}_dataset_{self.description}_{scene.simulation_config.mode}_{scene.mesh_prop.mesh_type}",  # timestamp
                    # folder="./three",
                    folder=f"{final_catalog}/three/{label}",
                    skip=20,
                )

                scene.iterate_self(scene.exact_acceleration)

            if correct:
                # Read all graph_data from temporary file and append using save_features_and_target
                with open(tmp_filename, "rb") as tmp_file:
                    try:
                        while True:
                            graph_data = pickle.load(tmp_file)
                            self.save_features_and_target(graph_data)
                    except EOFError:
                        pass
                # Clear temporary file
                os.remove(tmp_filename)
                scenario_id += 1

        # step_tqdm.set_description(f"{step_tqdm.desc} - done")
        cmh.save_to_log(
            f"Generated {all_steps} steps in {self.scenarios_count} scenarios"
        )
        return True
