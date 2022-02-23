from matplotlib.pyplot import draw

import deep_conmech.common.basic_helpers as basic_helpers
import deep_conmech.common.config as config
import deep_conmech.common.examples as examples
import deep_conmech.common.plotter.plotter_mapper as mapper
from deep_conmech.simulator.calculator import Calculator


def run_conmech_dynamic(all_scenatrios):

    for scenario in all_scenatrios:
        path = f"SIMULATOR - {basic_helpers.CURRENT_TIME}"

        mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            path,
            simulate_dirty_data=config.SIMULATE_DIRTY_DATA_SIMULATOR,
            print_base=False,
            description="Printing simulator",
        )


def main():
    scenarios = examples.all_print
    run_conmech_dynamic(scenarios)


if __name__ == "__main__":
    main()
