from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from conmech.helpers import cmh
from conmech.helpers.config import Config
from conmech.scenarios import scenarios
from conmech.simulations import simulation_runner
from deep_conmech.graph.model_jax import RMSE
from deep_conmech.run_model import get_checkpoint_path
from deep_conmech.training_config import get_train_config
from conmech.helpers.tmh import Timer
import json

# TODO:
# test with uninitialized network
# check difference between exact and lifted in plotted data

def main():
    load_dotenv()
      
    base_mode = "normal_with_reduced"
    other_modes = [
        "skinning",
        "net",
        "pca",
    ]
    num_runs = 3  # Number of times to run each scenario


    # copy_dir = '/home/michal/Desktop/conmech3d/output/25.04.11-09.46.59 - (17439274217021 - EPOCH6 - MODEL) - compare'
    # copy_dir = '/home/michal/Desktop/conmech3d/output/25.04.11-23.44.22 - (17442777618789 - EPOCH 16 - MODEL) - compare'
    # redo_modes = [base_mode, *other_modes]
    # redo_modes = ['net']
    copy_dir = None
    redo_modes = None

    config = get_train_config(shell=False, mode=None)
    checkpoint_path = get_checkpoint_path(config=config)
    label = checkpoint_path.split("/")[-1]
    main_dir = (
        f"{config.output_catalog}/{config.current_time} - ({label}) - compare"
    )
    # main_dir = '/home/michal/Desktop/conmech3d/output/25.04.05-21.29.31 - (17434958282739 - EPOCH 16 - MODEL) - compare'

    cmh.create_folders(main_dir)
    run_all_simulations(main_dir=main_dir, copy_dir=copy_dir, base_mode=base_mode, other_modes=other_modes, redo_modes=redo_modes, config=config, skip_base=False, num_runs=num_runs)
    create_report(main_dir=main_dir, base_mode=base_mode, other_modes=other_modes)

    # input("Press Enter to continue...")



def copy_previous_results(copy_dir, main_dir, modes_to_copy, scenario_name):
    """Copy previous simulation results for specified modes."""
    source_scenario = Path(copy_dir) / scenario_name
    target_scenario = Path(main_dir) / scenario_name
    
    for mode in modes_to_copy:
        source_mode = source_scenario / mode
        target_mode = target_scenario / mode
        
        if source_mode.exists():
            print(f"Copying {mode} results for {scenario_name}")
            cmh.create_folders(str(target_mode))
            cmh.copy_folder(str(source_mode), str(target_mode))
        else:
            print(f"Warning: Could not find {mode} results in {source_scenario}")


def run_single_scenario_single_mode(scenario, config, mode, additional_args = {}):
    additional_args['timer'] = Timer()
    config.sc.mode = mode

    _, scenario_dirs = simulation_runner.run_examples(
        all_scenarios=[scenario],
        file=__file__,
        plot_animation=False,
        config=config, #Config(shell=False),
        save_all=True,
        additional_args=additional_args
    )
    scenario_dir = scenario_dirs[0]

    save_timer_data(additional_args['timer'], final_catalog=scenario_dir)
    return scenario_dir


def save_timer_data(timer_data, final_catalog):
    if hasattr(timer_data, 'reduced_solver'):
        timer_data.reduced_solver = np.array(timer_data.reduced_solver)
    if hasattr(timer_data, 'solver'):
        timer_data.solver = np.array(timer_data.solver)
        
    timer_path = f"{final_catalog}/timer/data.json"
    cmh.create_folders(timer_path)
    with open(timer_path, "w") as f:
        json.dump(timer_data, f, indent=2)



def run_all_simulations(main_dir, copy_dir, base_mode, other_modes, redo_modes, config, skip_base=False, num_runs=3):
    cmh.print_jax_configuration()
    all_scenarios = scenarios.all_compare(config.td, config.sc)
    for scenario in all_scenarios:
        main_scenario_name = scenario.name 
        for run_idx in range(num_runs):
            scenario.name = f"{main_scenario_name}_run{run_idx+1}"
            print(f"Running scenario: {scenario.name}")
            run_single_scenario(
                main_dir=main_dir,
                copy_dir=copy_dir,
                scenario=scenario,
                base_mode=base_mode,
                other_modes=other_modes,
                redo_modes=redo_modes,
                config=config,
                skip_base=skip_base
            )

def run_single_scenario(main_dir, copy_dir, scenario, base_mode, other_modes, redo_modes, config, skip_base=False):
    additional_args = {'main_dir': main_dir}

    # Determine which modes to copy vs redo
    modes_to_copy = []
    if copy_dir:  # Only try to copy if copy_dir is specified
        if base_mode not in redo_modes:
            modes_to_copy.append(base_mode)
        modes_to_copy.extend([mode for mode in other_modes if mode not in redo_modes])
        
        # Copy results from previous run
        copy_previous_results(copy_dir, main_dir, modes_to_copy, scenario.name)

    # Only run base_mode if it wasn't copied and isn't skipped
    if base_mode not in modes_to_copy and not skip_base:
        print("MODE: ", base_mode)
        final_catalog = run_single_scenario_single_mode(scenario=scenario, config=config, mode=base_mode, additional_args=additional_args)
        additional_args['reduced_exact_accelerations'] = get_reduced_exact_acceleration(final_catalog=final_catalog)

    # Run only the modes that need to be redone
    for mode in other_modes:
        if mode not in modes_to_copy:
            print("MODE: ", mode)
            final_catalog = run_single_scenario_single_mode(scenario=scenario, config=config, mode=mode, additional_args=additional_args)


def get_error(simulation_1, simulation_2, index, key):
    return RMSE(simulation_1[index][key], simulation_2[index][key])


def get_reduced_exact_acceleration(final_catalog):
    simulation = cmh.load_simulation(final_catalog + '/scenes/data.scene')
    return [s['reduced_exact_acceleration'] for s in simulation]


def save_comparison_statistics(scenarios_paths, base_mode, other_modes, report_path, timer_stats):
    """Create markdown report with comparison statistics, plots and timer data for all scenarios."""
    markdown_content = "# Combined Comparison Results\n\n"
    markdown_content += f"Base mode: {base_mode}\n\n"

    # Add timer statistics section
    markdown_content += "## Timer Statistics\n\n"
    for key in ['reduced_solver', 'solver']:
        markdown_content += f"\n### {key} Statistics\n\n"
        markdown_content += "| Mode | Mean (s) | Max (s) | Min (s) | Sum (s) | Sum w/o outliers (s) |\n"
        markdown_content += "|------|----------|---------|---------|---------|--------------------|\n"
        
        plt.figure(figsize=(12, 6))
        plt.clf()
        
        # Sort modes to ensure consistent plotting order
        sorted_modes = sorted(timer_stats.keys())
        for mode in sorted_modes:
            timer_data = timer_stats[mode]
            if key in timer_data:
                times = np.array(timer_data[key])
                if len(times) > 0:
                    mean_time = np.mean(times)
                    max_time = np.max(times)
                    min_time = np.min(times)           
                    sum_time = np.sum(times)
                    sorted_times = np.sort(times)
                    sum_no_outliers = np.sum(sorted_times[1:-1]) if len(times) > 2 else sum_time

                    markdown_content += f"| {mode} | {mean_time:.6f} | {max_time:.6f} | {min_time:.6f} "
                    markdown_content += f"| {sum_time:.6f} | {sum_no_outliers:.6f} |\n"

                    plt.plot(times, label=mode)
        
        plt.title(f'{key} Times per Mode')
        plt.xlabel('Step')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(report_path) / f"{key}_times.png")
        markdown_content += f"\n![{key} Times]({key}_times.png)\n\n"

    markdown_content += "## Scenario Results\n\n"
    
    # Group scenarios by their base name (without _runX suffix)
    scenario_groups = {}
    scenarios_paths = sorted(scenarios_paths)
    for path in scenarios_paths:
        base_name = path.name.split("_run")[0]
        if base_name not in scenario_groups:
            scenario_groups[base_name] = []
        scenario_groups[base_name].append(path)

    # Dictionary to store errors for boxplots
    scenario_errors = {}
    
    # Add a dictionary to store summed errors across all runs
    scenario_summed_errors = {}
    
    for base_name, paths in scenario_groups.items():
        markdown_content += f"## Scenario: {base_name}\n\n"
        scenario_errors[base_name] = {}
        scenario_summed_errors[base_name] = {mode: {'total': 0.0, 'max': 0.0} for mode in other_modes}

        for key in ['displacement_old']:
            markdown_content += f"### {key} Results\n\n"

            # Process all runs and collect their tables
            all_run_tables = []
            max_rows = 0

            for path in paths:
                base = cmh.load_simulation(str(path / base_mode / "scenes" / "data.scene"))
                if base[0][key] is None:
                    continue

                run_rows = []
                run_rows.append("| Mode | Max Error | Total Error | % of Mid Total |")
                run_rows.append("|------|-----------|-------------|----------------|")

                plt.figure(figsize=(10, 6))
                # Calculate total errors first to find middle error
                mode_total_errors = {}
                for mode in other_modes:
                    pretendent = cmh.load_simulation(str(path / mode / "scenes" / "data.scene"))
                    simulation_len = min(len(base), len(pretendent))
                    
                    errors = []
                    for index in range(simulation_len):
                        errors.append(get_error(base, pretendent, index=index, key=key))
                    errors = np.array(errors)
                    mode_total_errors[mode] = np.sum(errors)
                    
                    # Update summed errors
                    scenario_summed_errors[base_name][mode]['total'] += np.sum(errors)
                    scenario_summed_errors[base_name][mode]['max'] = max(
                        scenario_summed_errors[base_name][mode]['max'],
                        np.max(errors)
                    )

                # Sort total errors and get the middle one as reference
                sorted_errors = sorted(mode_total_errors.values())
                middle_idx = len(sorted_errors) // 2
                reference_error = sorted_errors[middle_idx]

                for mode in other_modes:
                    pretendent = cmh.load_simulation(str(path / mode / "scenes" / "data.scene"))
                    simulation_len = min(len(base), len(pretendent))
                    
                    errors = []
                    for index in range(simulation_len):
                        errors.append(get_error(base, pretendent, index=index, key=key))
                    errors = np.array(errors)
                    
                    # Store errors for boxplot
                    if mode not in scenario_errors[base_name]:
                        scenario_errors[base_name][mode] = []
                    scenario_errors[base_name][mode].extend(errors.tolist())
                    
                    max_error = np.max(errors)
                    total_error = mode_total_errors[mode]
                    error_percentage = (total_error / reference_error) * 100

                    run_rows.append(f"| {mode} | {max_error:.6f} | {total_error:.6f} | {error_percentage:.1f}% |")
                    plt.plot(errors, label=mode)

                all_run_tables.append(run_rows)
                max_rows = max(max_rows, len(run_rows))

                plt.title(f'{path.name} - {key} Comparison')
                plt.xlabel('Step')
                plt.ylabel('Error')
                plt.legend()
                plt.grid(True)
                plot_filename = f"{path.name}_{key}.png"
                plt.savefig(str(report_path / plot_filename))
                plt.close()

            # Create a combined table header
            markdown_content += '<table style="border-collapse: collapse; width: 100%;">\n'
            markdown_content += '<tr>\n'
            for path in paths:
                run_name = path.name
                markdown_content += f'<th style="border: 1px solid #ddd; padding: 8px; text-align: center;" colspan="4">Run: {run_name}</th>\n'
            markdown_content += '</tr>\n'

            markdown_content += '<tr>\n'
            for _ in paths:
                markdown_content += '''<td style="border: 1px solid #ddd; padding: 8px;">Mode</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Max Error</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Total Error</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">% of Max Total</td>\n'''
            markdown_content += '</tr>\n'

            for mode in other_modes:
                markdown_content += '<tr>\n'
                for table in all_run_tables:
                    mode_row = next((row for row in table if row.startswith(f"| {mode} |")), None)
                    if mode_row:
                        cells = [cell.strip() for cell in mode_row.split('|')[1:-1]]
                        for cell in cells:
                            markdown_content += f'<td style="border: 1px solid #ddd; padding: 8px;">{cell}</td>\n'
                markdown_content += '</tr>\n'
            
            markdown_content += '</table>\n\n'

            markdown_content += "<div style='display: flex; flex-direction: row;'>\n"
            for path in paths:
                plot_filename = f"{path.name}_{key}.png"
                markdown_content += "<div style='flex: 1; margin: 5px;'>\n"
                markdown_content += f"<img src='{plot_filename}' style='width: 100%;'>\n"
                markdown_content += "</div>\n"
            markdown_content += "</div>\n\n"
            break

    # Add final aggregated table
    markdown_content += "## Aggregated Total Errors Across All Scenarios and Runs\n\n"
    markdown_content += "| Mode | Total Error | Max Error | % of Mid Total |\n"
    markdown_content += "|------|-------------|-----------|----------------|\n"
    total_errors = [sum(scenario_summed_errors[base_name][mode]['total'] for base_name in scenario_summed_errors) for mode in other_modes]
    median_total_error = np.median(total_errors)
    for mode, total_error in zip(other_modes, total_errors):
        max_error = max(scenario_summed_errors[base_name][mode]['max'] for base_name in scenario_summed_errors)
        percentage_of_median = (total_error / median_total_error) * 100 if median_total_error > 0 else 0
        markdown_content += f"| {mode} | {total_error:.6f} | {max_error:.6f} | {percentage_of_median:.1f}% |\n"

    with open(report_path / "comparison_results.md", 'w') as f:
        f.write(markdown_content)


def combine_timer_statistics(scenarios_paths, base_mode, other_modes):
    print("Outlier removed only here!")
    """Combine timer statistics from all scenarios and modes."""
    modes = [base_mode, *other_modes]
    combined_timer_stats = {}
    for scenario_path in scenarios_paths:
        # Load timer data for each mode
        for mode in modes:
            timer_path = scenario_path / mode / "timer" / "data.json"
            if timer_path.exists():
                with open(timer_path) as f:
                    timer_data = json.load(f)
                    if mode not in combined_timer_stats:
                        combined_timer_stats[mode] = {'solver': [], 'reduced_solver': []}
                        # combined_timer_stats[mode] = timer_data
                    # else:
                    # Combine solver times
                    for key in ['solver', 'reduced_solver']:
                        if key in timer_data and key in combined_timer_stats[mode]:
                            data = timer_data[key][1:] # Ignoring first element - outlier
                            combined_timer_stats[mode][key].extend(data)
    return combined_timer_stats


def create_report(main_dir, base_mode, other_modes):
    # Create main report folder
    report_path = Path(main_dir) / "report"
    cmh.recreate_folder(report_path)

    # Load simulation results and extract metadata
    scenarios_paths = [f for f in Path(main_dir).glob("*")     
                      if f.is_dir() and not f.name.startswith('.') and not f == report_path]

    # Combine and save statistics
    combined_timer_stats = combine_timer_statistics(scenarios_paths, base_mode, other_modes)
    save_comparison_statistics(scenarios_paths, base_mode, other_modes, report_path, combined_timer_stats)



if __name__ == "__main__":
    main()
