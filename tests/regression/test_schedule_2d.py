import numpy as np
import pytest

from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import Scenario
from conmech.simulations.simulation_runner import run_scenario


def generate_test_suits():
    scenario = Scenario(
        id="circle_slide_roll",
        mesh_data=MeshProperties(
            dimension=2,
            mesh_type=scenarios.m_circle,
            scale=[1],
            mesh_density=[3],
        ),
        body_prop=scenarios.default_body_prop,
        obstacle_prop=scenarios.default_obstacle_prop,
        schedule=Schedule(final_time=0.2), #1.5),
        forces_function=np.array([0.0, -0.5]),
        obstacles=np.array(
            [[[0.7, 1.0]], [[0.0, 0.1]]]
        ),
    )
    '''
    expected_boundary_nodes = [
        [1.09670439, 0.0700615],
        [0.51874802, 0.7207986],
        [0.26366884, -0.06828732],
        [1.11060891, 0.32856888],
        [0.99402219, 0.56024451],
        [0.77757494, 0.70261436],
        [0.28733568, 0.60572044],
        [0.14549146, 0.39166315],
        [0.13608484, 0.14131729],
        [0.46416333, -0.22844126],
        [0.72052413, -0.25791514],
        [0.95457633, -0.14578926]
    ]
    '''

    expected_boundary_nodes = [
       [ 1.       ,  0.4895   ],
       [ 0.25     ,  0.9225127],
       [ 0.25     ,  0.0564873],
       [ 0.9330127,  0.7395   ],
       [ 0.75     ,  0.9225127],
       [ 0.5      ,  0.9895   ],
       [ 0.0669873,  0.7395   ],
       [-0.       ,  0.4895   ],
       [ 0.0669873,  0.2395   ],
       [ 0.5      , -0.0105   ],
       [ 0.75     ,  0.0564873],
       [ 0.9330127,  0.2395   ]
    ]


    yield scenario, expected_boundary_nodes


@pytest.mark.parametrize("scenario, expected_boundary_nodes", list(generate_test_suits()))
def test_simulation(scenario, expected_boundary_nodes):
    config = Config()
    setting, _ = run_scenario(
        solve_function=scenario.get_solve_function(),
        scenario=scenario,
        catalog=f"TEST_{scenario.id}",
        simulate_dirty_data=False,
        plot_animation=config.PLOT_TESTS,
        config=config,
    )

    np.set_printoptions(precision=8, suppress=True)
    np.testing.assert_array_almost_equal(
        setting.boundary_nodes, expected_boundary_nodes, decimal=3
    )
    
